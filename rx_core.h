/* Packet parsing and image reassembly shared by dpdk_rx.c and pcap_replay_test.c.
 * No DPDK dependency, so the same logic can be tested without DPDK.
 */
#ifndef RX_CORE_H
#define RX_CORE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_MAGIC      0x494D4731u   /* "IMG1" */
#define IMG_HDR_LEN    22
#define MAX_FRAMES     256
#define MAX_FRAME_LEN  (4u << 20)
#define MAX_SEGS       4096

struct frame {
    int       used;
    uint32_t  frame_bytes;
    uint16_t  nseg;
    uint16_t  got;           /* distinct chunks received */
    uint8_t  *data;
    uint8_t  *seen;          /* one byte per chunk */
};

struct rx_stats {
    uint64_t packets, accepted, duplicates;
    uint64_t not_ipv4, not_udp, bad_len, other_port, bad_magic, bad_hdr, too_many_frames;
};

struct rx_state {
    uint16_t        port;            /* UDP destination port to accept */
    struct frame    frames[MAX_FRAMES];
    struct rx_stats st;
};

static inline uint16_t rd16(const uint8_t *p) { return (uint16_t)(p[0] << 8 | p[1]); }
static inline uint32_t rd32(const uint8_t *p)
{ return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

/* Handle one Ethernet frame. Network headers are big-endian, so read them byte by byte. */
static void rx_handle(struct rx_state *s, const uint8_t *pkt, uint32_t len)
{
    s->st.packets++;

    /* Ethernet: 14 bytes, EtherType at 12 */
    if (len < 14 || rd16(pkt + 12) != 0x0800) { s->st.not_ipv4++; return; }
    const uint8_t *ip = pkt + 14;
    uint32_t iplen_avail = len - 14;

    /* IPv4: version 4, header length >= 20 bytes, protocol UDP, not a fragment */
    if (iplen_avail < 20 || (ip[0] >> 4) != 4) { s->st.not_ipv4++; return; }
    uint32_t ihl = (uint32_t)(ip[0] & 0x0F) * 4;
    uint32_t tot = rd16(ip + 2);
    if (ihl < 20 || tot < ihl + 8 || tot > iplen_avail) { s->st.bad_len++; return; }
    if (ip[9] != 17) { s->st.not_udp++; return; }
    if (rd16(ip + 6) & 0x3FFF) { s->st.bad_len++; return; }   /* MF flag or fragment offset */

    /* UDP */
    const uint8_t *udp = ip + ihl;
    uint32_t ulen = rd16(udp + 4);
    if (ulen < 8 || ulen > tot - ihl) { s->st.bad_len++; return; }
    if (rd16(udp + 2) != s->port) { s->st.other_port++; return; }

    /* Our header */
    const uint8_t *h = udp + 8;
    uint32_t plen = ulen - 8;
    if (plen < IMG_HDR_LEN || rd32(h) != IMG_MAGIC) { s->st.bad_magic++; return; }
    uint32_t frame_id = rd32(h + 4);
    uint16_t seq = rd16(h + 8), nseg = rd16(h + 10);
    uint32_t off = rd32(h + 12);
    uint16_t clen = rd16(h + 16);
    uint32_t fbytes = rd32(h + 18);

    if (nseg == 0 || nseg > MAX_SEGS || seq >= nseg || fbytes == 0 || fbytes > MAX_FRAME_LEN ||
        clen > plen - IMG_HDR_LEN || (uint64_t)off + clen > fbytes) { s->st.bad_hdr++; return; }
    if (frame_id >= MAX_FRAMES) { s->st.too_many_frames++; return; }

    struct frame *f = &s->frames[frame_id];
    if (!f->used) {
        f->data = calloc(1, fbytes);
        f->seen = calloc(1, nseg);
        if (!f->data || !f->seen) { fprintf(stderr, "out of memory\n"); exit(1); }
        f->used = 1; f->frame_bytes = fbytes; f->nseg = nseg; f->got = 0;
    } else if (f->frame_bytes != fbytes || f->nseg != nseg) {
        s->st.bad_hdr++; return;                 /* inconsistent with earlier chunks */
    }
    if (f->seen[seq]) { s->st.duplicates++; return; }
    memcpy(f->data + off, h + IMG_HDR_LEN, clen);   /* any arrival order works: offset places it */
    f->seen[seq] = 1;
    f->got++;
    s->st.accepted++;
}

/* Number of frames whose chunks have all arrived. */
static int rx_complete(const struct rx_state *s)
{
    int n = 0;
    for (int i = 0; i < MAX_FRAMES; i++)
        if (s->frames[i].used && s->frames[i].got == s->frames[i].nseg) n++;
    return n;
}

/* Team format: frames_rx.bin = raw RGB bytes of frame 0, 1, ..., n_frames-1, back to back,
 * exactly like frames_original.bin. A missing or incomplete frame is written as zeros (so the
 * offsets of the other frames stay right) and reported. Returns the number of complete frames. */
static int rx_write(struct rx_state *s, const char *path, uint32_t n_frames, uint32_t frame_bytes)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror(path); return -1; }
    uint8_t *zeros = calloc(1, frame_bytes);
    int complete = 0;
    for (uint32_t id = 0; id < n_frames && id < MAX_FRAMES; id++) {
        struct frame *f = &s->frames[id];
        if (f->used && f->got == f->nseg && f->frame_bytes == frame_bytes) {
            fwrite(f->data, 1, frame_bytes, fp);
            complete++;
        } else {
            if (!f->used) printf("frame %u missing\n", id);
            else printf("frame %u incomplete: %u of %u chunks\n", id, f->got, f->nseg);
            fwrite(zeros, 1, frame_bytes, fp);
        }
    }
    for (uint32_t id = n_frames; id < MAX_FRAMES; id++)
        if (s->frames[id].used) printf("frame %u is beyond the expected %u frames, not written\n", id, n_frames);
    free(zeros);
    fclose(fp);
    return complete;
}

static void rx_report(const struct rx_state *s, int complete)
{
    const struct rx_stats *t = &s->st;
    uint64_t dropped = t->packets - t->accepted;
    printf("packets received : %llu\n", (unsigned long long)t->packets);
    printf("chunks accepted  : %llu\n", (unsigned long long)t->accepted);
    printf("packets dropped  : %llu (duplicates %llu, other port %llu, not IPv4 %llu, not UDP %llu,"
           " bad length %llu, bad magic %llu, bad header %llu, frame id too large %llu)\n",
           (unsigned long long)dropped, (unsigned long long)t->duplicates,
           (unsigned long long)t->other_port, (unsigned long long)t->not_ipv4,
           (unsigned long long)t->not_udp, (unsigned long long)t->bad_len,
           (unsigned long long)t->bad_magic, (unsigned long long)t->bad_hdr,
           (unsigned long long)t->too_many_frames);
    printf("complete images  : %d\n", complete);
}

static void rx_free(struct rx_state *s)
{
    for (int i = 0; i < MAX_FRAMES; i++) { free(s->frames[i].data); free(s->frames[i].seen); }
}

#endif
