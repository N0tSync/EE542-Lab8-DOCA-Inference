/* Test for rx_core.h without DPDK: reads a PCAP file directly and feeds each packet to the same
 * rx_handle() that dpdk_rx uses, then writes frames_rx.bin in the same team format.
 *
 *   make test_rx
 *   ./pcap_replay_test images.pcap frames_rx.bin [n_frames] [udp_port]
 */
#include "rx_core.h"

#define FRAME_BYTES (224 * 224 * 3)

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s in.pcap out.bin [n_frames] [udp_port]\n", argv[0]); return 2; }
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror(argv[1]); return 1; }
    uint32_t gh[6];
    if (fread(gh, 4, 6, fp) != 6 || gh[0] != 0xA1B2C3D4u) {
        fprintf(stderr, "not a little-endian microsecond pcap file\n"); return 1;
    }
    uint32_t n_frames = argc > 3 ? (uint32_t)atoi(argv[3]) : 16;
    if (n_frames < 1 || n_frames > MAX_FRAMES) {
        fprintf(stderr, "expected 1-256 frames\n"); return 2;
    }
    static struct rx_state st;
    st.port = argc > 4 ? (uint16_t)atoi(argv[4]) : 5000;
    static uint8_t buf[65536];
    uint32_t rh[4];
    while (fread(rh, 4, 4, fp) == 4) {
        if (rh[2] > sizeof buf || fread(buf, 1, rh[2], fp) != rh[2]) break;
        rx_handle(&st, buf, rh[2]);
        if ((uint32_t)rx_complete(&st, n_frames) >= n_frames) break;
    }
    fclose(fp);
    int complete = rx_write(&st, argv[2], n_frames, FRAME_BYTES);
    rx_report(&st, complete);
    rx_free(&st);
    return complete == (int)n_frames ? 0 : 1;
}
