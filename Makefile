# Parts 3-5 build and integration entry points. Run from the repository root.
CFLAGS ?= -O2 -Wall
PYTHON ?= python3
SUDO ?= sudo
DPDK_CFLAGS := $(shell pkg-config --cflags libdpdk 2>/dev/null)
DPDK_LIBS   := $(shell pkg-config --libs libdpdk 2>/dev/null)
EAL ?= -l 0 --no-huge -m 512 --no-pci
BUILD_DIR := build
PCAP_DIR := data/pcaps
FRAME_DIR := data/frames
PART5_OUTPUT ?= results/part5

all: $(BUILD_DIR)/dpdk_rx

$(BUILD_DIR) $(PCAP_DIR) $(FRAME_DIR):
	mkdir -p $@

$(BUILD_DIR)/dpdk_rx: dpdk/dpdk_rx.c dpdk/rx_core.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(DPDK_CFLAGS) -Idpdk dpdk/dpdk_rx.c -o $@ $(DPDK_LIBS)

$(BUILD_DIR)/pcap_replay_test: tests/pcap_replay_test.c dpdk/rx_core.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Idpdk tests/pcap_replay_test.c -o $@

dpdk_rx: $(BUILD_DIR)/dpdk_rx

pcap_replay_test: $(BUILD_DIR)/pcap_replay_test

pcaps: | $(PCAP_DIR) $(FRAME_DIR)
	$(PYTHON) -m lab8.packets.generate_pcap -o $(PCAP_DIR)/images_inorder.pcap --orig $(FRAME_DIR)/frames_original.bin
	$(PYTHON) -m lab8.packets.generate_pcap --shuffle -o $(PCAP_DIR)/images_shuffled.pcap --orig $(FRAME_DIR)/frames_original.bin
	$(PYTHON) -m lab8.packets.generate_pcap --shuffle --dups 50 --noise 20 -o $(PCAP_DIR)/images_messy.pcap --orig $(FRAME_DIR)/frames_original.bin

# Same reassembly code without DPDK (works on any Linux/macOS, no sudo)
test_rx: $(BUILD_DIR)/pcap_replay_test pcaps
	for m in inorder shuffled messy; do \
	  ./$(BUILD_DIR)/pcap_replay_test $(PCAP_DIR)/images_$$m.pcap $(FRAME_DIR)/frames_rx_$$m.bin && \
	  $(PYTHON) -m lab8.packets.verify_frames $(FRAME_DIR)/frames_original.bin $(FRAME_DIR)/frames_rx_$$m.bin || exit 1; done

# The real thing with DPDK
run: $(BUILD_DIR)/dpdk_rx pcaps
	for m in inorder shuffled messy; do \
	  $(SUDO) ./$(BUILD_DIR)/dpdk_rx $(EAL) --vdev "net_pcap0,rx_pcap=$(PCAP_DIR)/images_$$m.pcap,tx_pcap=/dev/null" -- $(FRAME_DIR)/frames_rx_$$m.bin 16 && \
	  $(PYTHON) -m lab8.packets.verify_frames $(FRAME_DIR)/frames_original.bin $(FRAME_DIR)/frames_rx_$$m.bin || exit 1; done
	cp $(FRAME_DIR)/frames_rx_shuffled.bin $(FRAME_DIR)/frames_rx.bin

part5: run
	$(PYTHON) -u -m lab8.integration.classify_frames --output-dir $(PART5_OUTPUT)

clean:
	rm -f $(BUILD_DIR)/dpdk_rx $(BUILD_DIR)/pcap_replay_test $(PCAP_DIR)/images*.pcap $(FRAME_DIR)/frames_rx*.bin

.PHONY: all dpdk_rx pcap_replay_test pcaps test_rx run part5 clean
