"""Parts 3 (member B): split 16 test images (224 x 224 x 3, HWC uint8) into UDP packets and write images.pcap.

Payload = 22-byte header (network byte order) + up to 1,400 bytes of image data:
    magic(4) frame_id(4) seq(2) nseg(2) offset(4) length(2) frame_bytes(4)

150,528 bytes / 1,400 = 107.5 -> 108 packets per image (107 full + one of 728 bytes).
IP packet = 20 (IPv4) + 8 (UDP) + 22 (header) + 1,400 = 1,450 bytes <= 1,500-byte MTU.
Ethernet frame = 14 + 1,450 = 1,464 bytes.

Usage:
    python generate_pcap.py                         # in order        -> images.pcap
    python generate_pcap.py --shuffle -o shuf.pcap  # shuffled order
    python generate_pcap.py --shuffle --dups 50 --noise 20 -o messy.pcap
        (adds 50 duplicate chunks and 20 packets to another UDP port, which the receiver must ignore)

Also writes frames_original.bin (the original images, raw RGB back to back) and, with --png, PNG previews.
"""
import argparse
import random
import struct

import numpy as np

MAGIC = 0x494D4731            # "IMG1"
HDR = struct.Struct("!IIHHIHI")  # 22 bytes, big-endian
H, W, C = 224, 224, 3
FRAME_BYTES = H * W * C       # 150,528
CHUNK = 1400
DPORT = 5000
SPORT = 40000


def make_images(n, seed=0, source="synthetic"):
    """Return a list of n HWC uint8 images."""
    if source == "cifar":
        import torch, torchvision  # noqa: E401
        ds = torchvision.datasets.CIFAR10("./data", train=False, download=True)
        imgs = []
        for i in range(n):
            img = ds[i * 37][0].resize((W, H))   # PIL bilinear upscale 32 -> 224
            imgs.append(np.asarray(img, dtype=np.uint8).copy())
        return imgs
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    imgs = []
    for i in range(n):
        img = np.zeros((H, W, C), np.float32)
        for c in range(C):                       # smooth colour gradients, different per image
            fx, fy, ph = rng.uniform(0.5, 4), rng.uniform(0.5, 4), rng.uniform(0, 6.28)
            img[..., c] = 127 + 120 * np.sin(fx * xx / W * 6.28 + fy * yy / H * 6.28 + ph)
        cy, cx, r = rng.integers(40, 184), rng.integers(40, 184), rng.integers(20, 50)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r   # a solid disc so images look distinct
        img[mask] = rng.integers(0, 256, 3)
        img += rng.normal(0, 6, img.shape)
        imgs.append(np.clip(img, 0, 255).astype(np.uint8))
    return imgs


def chunk_payloads(frame_id, data):
    nseg = (len(data) + CHUNK - 1) // CHUNK
    for seq in range(nseg):
        off = seq * CHUNK
        part = data[off:off + CHUNK]
        yield HDR.pack(MAGIC, frame_id, seq, nseg, off, len(part), len(data)) + part


# ---- Ethernet/IPv4/UDP framing -------------------------------------------------------
def build_frames_scapy(payloads):
    from scapy.all import Ether, IP, UDP, Raw
    return [bytes(Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") /
                  IP(src="10.0.0.1", dst="10.0.0.2") /
                  UDP(sport=SPORT, dport=dport) / Raw(p)) for p, dport in payloads]


def _csum(b):
    if len(b) % 2:
        b += b"\0"
    s = sum(struct.unpack(f"!{len(b) // 2}H", b))
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return ~s & 0xFFFF


def build_frames_manual(payloads):
    """Same bytes as the Scapy version; used only when Scapy is not installed."""
    out, ip_id = [], 1
    src, dst = bytes([10, 0, 0, 1]), bytes([10, 0, 0, 2])
    eth = bytes.fromhex("020000000002" "020000000001") + b"\x08\x00"
    for p, dport in payloads:
        ulen = 8 + len(p)
        pseudo = src + dst + struct.pack("!BBH", 0, 17, ulen)
        udp0 = struct.pack("!HHHH", SPORT, dport, ulen, 0)
        ucs = _csum(pseudo + udp0 + p) or 0xFFFF
        udp = struct.pack("!HHHH", SPORT, dport, ulen, ucs)
        ip0 = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + ulen, ip_id & 0xFFFF, 0, 64, 17, 0, src, dst)
        ip = ip0[:10] + struct.pack("!H", _csum(ip0)) + ip0[12:]
        out.append(eth + ip + udp + p)
        ip_id += 1
    return out


def write_pcap(path, frames):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))   # Ethernet linktype
        for i, fr in enumerate(frames):
            f.write(struct.pack("<IIII", i // 1000000, i % 1000000, len(fr), len(fr)) + fr)


def write_frames_file(path, imgs):
    """Team format: raw RGB bytes (224 x 224 x 3, HWC, uint8) of frame 0, 1, ... back to back.
    frames_rx.bin from dpdk_rx uses exactly the same layout."""
    with open(path, "wb") as f:
        for img in imgs:
            f.write(img.tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--num-images", type=int, default=16)
    p.add_argument("-o", "--output", default="images.pcap")
    p.add_argument("--shuffle", action="store_true", help="shuffle the packet order")
    p.add_argument("--dups", type=int, default=0, help="add N duplicate chunks")
    p.add_argument("--noise", type=int, default=0, help="add N packets for another UDP port")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source", choices=["synthetic", "cifar"], default="synthetic")
    p.add_argument("--orig", default="frames_original.bin")
    p.add_argument("--png", action="store_true", help="also save PNG previews in originals/")
    p.add_argument("--no-scapy", action="store_true")
    a = p.parse_args()

    imgs = make_images(a.num_images, a.seed, a.source)
    payloads = [(pl, DPORT) for i, img in enumerate(imgs) for pl in chunk_payloads(i, img.tobytes())]
    n_data = len(payloads)
    rng = random.Random(a.seed)
    payloads += [rng.choice(payloads[:n_data]) for _ in range(a.dups)]
    payloads += [(bytes(rng.getrandbits(8) for _ in range(200)), 6000) for _ in range(a.noise)]
    if a.shuffle:
        rng.shuffle(payloads)

    use_scapy = not a.no_scapy
    if use_scapy:
        try:
            frames = build_frames_scapy(payloads)
        except ImportError:
            print("Scapy not found, using the built-in packet builder")
            use_scapy = False
    if not use_scapy:
        frames = build_frames_manual(payloads)

    write_pcap(a.output, frames)
    write_frames_file(a.orig, imgs)
    if a.png:
        import os
        from PIL import Image
        os.makedirs("originals", exist_ok=True)
        for i, img in enumerate(imgs):
            Image.fromarray(img).save(f"originals/frame_{i:02d}.png")

    per_img = (FRAME_BYTES + CHUNK - 1) // CHUNK
    print(f"{a.num_images} images x {FRAME_BYTES:,} bytes, {per_img} packets per image "
          f"(last chunk {FRAME_BYTES - (per_img - 1) * CHUNK} bytes)")
    print(f"largest frame {max(map(len, frames))} bytes on the wire, IP packet "
          f"{max(map(len, frames)) - 14} bytes (MTU 1500)")
    print(f"wrote {a.output}: {len(frames)} packets ({n_data} data, {a.dups} duplicates, "
          f"{a.noise} other-port), {'shuffled' if a.shuffle else 'in order'}; originals in {a.orig}")


if __name__ == "__main__":
    main()
