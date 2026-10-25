# Parts 3–4: packet format and file layout (member B)

## Images
- 16 images, frame_id 0–15, each 224 × 224 × 3 RGB, uint8, HWC order = **150,528 bytes**.
- `frames_original.bin` = raw bytes of frame 0, 1, …, 15 back to back (2,408,448 bytes). No header.
- `frames_rx.bin` (written by `dpdk_rx`) = exactly the same layout. A missing or incomplete frame is written as zeros so the other frames stay at the right offset, and the receiver prints which ones.
- To read a frame in Python: `np.fromfile("frames_rx.bin", np.uint8).reshape(16, 224, 224, 3)`.

## Packets
Ethernet (14 B) / IPv4 10.0.0.1 → 10.0.0.2 (20 B) / UDP 40000 → **5000** (8 B) / payload.

Payload = 22-byte header + up to 1,400 bytes of image data. All header fields are **big-endian** (network byte order), packed with no padding:

| Offset | Field | Size | Meaning |
|---|---|---|---|
| 0 | magic | 4 B | `0x494D4731` ("IMG1") |
| 4 | frame_id | 4 B | Image this chunk belongs to (0–15) |
| 8 | seq | 2 B | Chunk number within the image (0–107) |
| 10 | nseg | 2 B | Chunks in the image (108) |
| 12 | offset | 4 B | Byte offset of the chunk in the image (seq × 1,400) |
| 16 | length | 2 B | Image bytes in this packet (1,400; last chunk 728) |
| 18 | frame_bytes | 4 B | Total image size (150,528) |

- Packets per image: 150,528 / 1,400 = 107.5 → **108** (107 × 1,400 + 1 × 728).
- Largest IP packet: 20 + 8 + 22 + 1,400 = **1,450 B ≤ 1,500 MTU**. Ethernet frame 1,464 B.
- 16 images → **1,728 data packets**.

## What the receiver checks
EtherType IPv4, IP version 4, header length, total length, protocol UDP, not a fragment, UDP length, destination port 5000, magic, and the fixed image/chunk format above. In particular, `frame_bytes` must be 150,528, `nseg` must be 108, `offset` must equal `seq * 1400`, and `length` must be exactly 1,400 (728 for the last chunk). The UDP payload must contain exactly the application header and that chunk. Empty, overlapping, short, and inconsistent chunks are rejected before their sequence ID is marked received. A per-chunk "seen" flag drops duplicates. Every drop reason is counted and printed.

## When the receiver stops
The pcap port never signals end of file. `dpdk_rx` stops when all expected frame IDs (default 0-15) are complete, or after 500 ms with no packets, and then reports any missing or incomplete frames. Frames beyond the expected range do not count toward the stop condition. An incomplete result returns a nonzero exit status.

## Run
```bash
pip install scapy numpy            # the lab asks for Scapy; without it the script builds identical bytes itself
make test_rx                       # reassembly test without DPDK: in order, shuffled, shuffled + 50 duplicates + 20 other-port
make run                           # the real DPDK receiver on all three PCAPs (needs DPDK + sudo, Linux)
python test_packet_pipeline.py --dpdk  # malformed chunks, valid retries, and failure propagation
make part5                         # DPDK reassembly followed by GPU classification and H2D timing
```
`make run` leaves `frames_rx.bin` (from the shuffled PCAP) for member C's Part 5.
The verifier runs directly, so a byte mismatch fails both `make test_rx` and
`make run`. Use `SUDO=` when DPDK pcap access works without root, as on the
validated WSL host. `PYTHON` and `EAL` can also be overridden for the target host.
Generated root-level binaries, PCAPs, and frame buffers are ignored by Git.
