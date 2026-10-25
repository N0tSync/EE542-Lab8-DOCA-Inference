# Parts 3-4 (member B)
CFLAGS ?= -O2 -Wall
DPDK_CFLAGS := $(shell pkg-config --cflags libdpdk 2>/dev/null)
DPDK_LIBS   := $(shell pkg-config --libs libdpdk 2>/dev/null)
EAL := -l 0 --no-huge -m 512 --no-pci

all: dpdk_rx

dpdk_rx: dpdk_rx.c rx_core.h
	$(CC) $(CFLAGS) $(DPDK_CFLAGS) dpdk_rx.c -o $@ $(DPDK_LIBS)

pcap_replay_test: pcap_replay_test.c rx_core.h
	$(CC) $(CFLAGS) pcap_replay_test.c -o $@

pcaps:
	python3 generate_pcap.py -o images_inorder.pcap
	python3 generate_pcap.py --shuffle -o images_shuffled.pcap
	python3 generate_pcap.py --shuffle --dups 50 --noise 20 -o images_messy.pcap

# Same reassembly code without DPDK (works on any Linux/macOS, no sudo)
test_rx: pcap_replay_test pcaps
	for m in inorder shuffled messy; do \
	  ./pcap_replay_test images_$$m.pcap frames_rx_$$m.bin && \
	  python3 verify_frames.py frames_original.bin frames_rx_$$m.bin | tail -1 || exit 1; done

# The real thing with DPDK
run: dpdk_rx pcaps
	for m in inorder shuffled messy; do \
	  sudo ./dpdk_rx $(EAL) --vdev "net_pcap0,rx_pcap=images_$$m.pcap,tx_pcap=/dev/null" -- frames_rx_$$m.bin 16 && \
	  python3 verify_frames.py frames_original.bin frames_rx_$$m.bin | tail -1 || exit 1; done
	cp frames_rx_shuffled.bin frames_rx.bin

clean:
	rm -f dpdk_rx pcap_replay_test *.pcap frames_rx*.bin

.PHONY: all pcaps test_rx run clean
