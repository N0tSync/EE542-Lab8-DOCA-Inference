/* Parts 3-4 (member B): receive image packets with DPDK from a pcap virtual port and rebuild the images.
 *
 * Build from repository root: make
 * Run:
 *   sudo ./build/dpdk_rx -l 0 --no-huge -m 512 --no-pci \
 *        --vdev 'net_pcap0,rx_pcap=data/pcaps/images.pcap,tx_pcap=/dev/null' \
 *        -- data/frames/frames_rx.bin [n_frames] [udp_port] [idle_ms]
 *   defaults: n_frames 16, udp_port 5000, idle_ms 500
 *
 * Output: frames_rx.bin = raw RGB bytes of frames 0..n_frames-1 back to back (224 x 224 x 3 each),
 * the same layout as frames_original.bin, so the two files can be compared byte for byte.
 *
 * When to stop: the pcap port never reports end of file (rx_burst just keeps returning 0), so the
 * receiver stops as soon as all n_frames images are complete, or, if some never complete, after
 * idle_ms with no packets. It then says which frames are missing or incomplete.
 */
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>
#include <rte_errno.h>

#include "rx_core.h"

#define BURST       32
#define FRAME_BYTES (224 * 224 * 3)

int main(int argc, char **argv)
{
    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "EAL init failed\n");
    /* After rte_eal_init, argv[ret] is the program name, so our first argument is argv[ret + 1]. */
    argc -= ret;
    argv += ret;
    const char *out_path = argc > 1 ? argv[1] : "data/frames/frames_rx.bin";
    uint32_t n_frames  = argc > 2 ? (uint32_t)atoi(argv[2]) : 16;
    uint16_t port_udp  = argc > 3 ? (uint16_t)atoi(argv[3]) : 5000;
    unsigned idle_ms   = argc > 4 ? (unsigned)atoi(argv[4]) : 500;
    if (n_frames < 1 || n_frames > MAX_FRAMES || port_udp == 0 || idle_ms == 0)
        rte_exit(EXIT_FAILURE, "expected 1-256 frames, a nonzero UDP port and positive idle timeout\n");

    if (rte_eth_dev_count_avail() == 0)
        rte_exit(EXIT_FAILURE, "no ports: did you pass --vdev?\n");

    struct rte_mempool *pool = rte_pktmbuf_pool_create("mbufs", 8191, 256, 0,
                                   RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!pool)
        rte_exit(EXIT_FAILURE, "mbuf pool: %s\n", rte_strerror(rte_errno));

    struct rte_eth_conf conf;
    memset(&conf, 0, sizeof conf);
    if (rte_eth_dev_configure(0, 1, 1, &conf) < 0 ||
        rte_eth_rx_queue_setup(0, 0, 1024, rte_socket_id(), NULL, pool) < 0 ||
        rte_eth_tx_queue_setup(0, 0, 1024, rte_socket_id(), NULL) < 0 ||
        rte_eth_dev_start(0) < 0)
        rte_exit(EXIT_FAILURE, "port 0 setup failed\n");

    static struct rx_state st;          /* static: zero-initialised, and too big for the stack */
    st.port = port_udp;

    uint64_t hz = rte_get_timer_hz();
    uint64_t idle_cycles = hz * idle_ms / 1000;
    uint64_t last_rx = rte_get_timer_cycles(), t_first = 0, t_last = 0;
    int got_any = 0;
    const char *why = "idle timeout";
    struct rte_mbuf *bufs[BURST];

    for (;;) {
        uint16_t n = rte_eth_rx_burst(0, 0, bufs, BURST);
        uint64_t now = rte_get_timer_cycles();
        if (n == 0) {
            if (got_any && now - last_rx > idle_cycles)
                break;                                   /* file finished, some frames may be incomplete */
            if (!got_any && now - last_rx > 10 * hz) { why = "no packets in 10 s"; break; }
            continue;
        }
        if (!got_any) { got_any = 1; t_first = now; }
        last_rx = t_last = now;
        for (uint16_t i = 0; i < n; i++) {
            struct rte_mbuf *m = bufs[i];
            if (m->nb_segs == 1)
                rx_handle(&st, rte_pktmbuf_mtod(m, const uint8_t *), rte_pktmbuf_data_len(m));
            else {                                       /* multi-segment mbuf: linearise first */
                static uint8_t tmp[16384];
                uint32_t len = rte_pktmbuf_pkt_len(m);
                if (len > sizeof tmp) len = sizeof tmp;
                const uint8_t *p = rte_pktmbuf_read(m, 0, len, tmp);
                if (p) rx_handle(&st, p, len);
            }
            rte_pktmbuf_free(m);
        }
        if ((uint32_t)rx_complete(&st, n_frames) >= n_frames) { why = "all expected frames complete"; break; }
    }

    printf("stopped          : %s\n", why);
    int complete = rx_write(&st, out_path, n_frames, FRAME_BYTES);
    rx_report(&st, complete);
    printf("expected images  : %u\n", n_frames);
    printf("receive time     : %.3f ms (first to last packet)\n",
           (double)(t_last - t_first) * 1000.0 / (double)hz);
    printf("wrote %s (%u x %d bytes)\n", out_path, n_frames, FRAME_BYTES);

    rx_free(&st);
    rte_eth_dev_stop(0);
    rte_eth_dev_close(0);
    rte_eal_cleanup();
    return complete == (int)n_frames ? 0 : 1;
}
