"""Regression checks for malformed chunks and Makefile verification exit codes."""

import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

from lab8.packets import generate_pcap as gen

ROOT = Path(__file__).resolve().parents[1]


def invoke(args, cwd):
    return subprocess.run([str(arg) for arg in args], cwd=cwd, text=True,
                          capture_output=True, timeout=30)


def require(condition, result):
    if not condition:
        raise AssertionError(f"Unexpected exit {result.returncode}:\n{result.stdout}\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpdk", action="store_true", help="also exercise the real DPDK pcap port")
    args = parser.parse_args()
    build = invoke(["make", "pcap_replay_test", *(["dpdk_rx"] if args.dpdk else [])], ROOT)
    require(build.returncode == 0, build)
    with tempfile.TemporaryDirectory(prefix="ee542-packet-regression-") as directory:
        work = Path(directory)
        images = gen.make_images(16)
        original = b"".join(image.tobytes() for image in images)
        (work / "frames_original.bin").write_bytes(original)
        payloads = [(payload, gen.DPORT) for frame_id, image in enumerate(images)
                    for payload in gen.chunk_payloads(frame_id, image.tobytes())]

        def check(name, packets, expected_exit, expected_bad_headers=0):
            pcap = work / f"{name}.pcap"
            gen.write_pcap(pcap, gen.build_frames_manual(packets))
            for receiver in (["replay", "dpdk"] if args.dpdk else ["replay"]):
                output = work / f"{name}_{receiver}.bin"
                if receiver == "replay":
                    cmd = [ROOT / "build/pcap_replay_test", pcap, output]
                else:
                    cmd = [ROOT / "build/dpdk_rx", "-l", "0", "--no-huge", "-m", "512", "--no-pci",
                           "--vdev", f"net_pcap0,rx_pcap={pcap},tx_pcap=/dev/null", "--", output, "16"]
                result = invoke(cmd, work)
                require(result.returncode == expected_exit, result)
                require(f"bad header {expected_bad_headers}," in result.stdout, result)
                complete = 16 if expected_exit == 0 else 15
                require(f"complete images  : {complete}" in result.stdout, result)
                require((output.read_bytes() == original) == (expected_exit == 0), result)
                print(f"PASS {receiver}: {name}, exit={result.returncode}, "
                      f"complete={complete}, bad_headers={expected_bad_headers}", flush=True)

        check("valid", payloads, 0)
        check("missing_chunk", payloads[:1] + payloads[2:], 1)
        mutations = {
            "overlapping_offset": (1, 12, "!I", 0),
            "zero_length": (1, 16, "!H", 0),
            "wrong_segment_count": (1, 10, "!H", 107),
            "wrong_frame_size": (1, 18, "!I", gen.FRAME_BYTES - 1),
            "short_final_chunk": (107, 16, "!H", 727),
        }
        for name, (index, offset, fmt, value) in mutations.items():
            packets = list(payloads)
            changed = bytearray(packets[index][0])
            struct.pack_into(fmt, changed, offset, value)
            packets[index] = (bytes(changed), gen.DPORT)
            check(name, packets, 1, 1)
            if name in ("overlapping_offset", "zero_length"):
                check(name + "_then_valid_retry", packets + [payloads[index]], 0, 1)
        extra = list(payloads)
        extra[1] = (extra[1][0] + b"extra", gen.DPORT)
        check("extra_payload_bytes", extra, 1, 1)

        # A complete frame beyond the expected IDs must not terminate RX early.
        unexpected = [(payload, gen.DPORT) for payload in gen.chunk_payloads(16, images[15].tobytes())]
        check("unexpected_frame_before_last_expected", payloads[:15 * 108] + unexpected + payloads[15 * 108:], 0)

        # Use the actual recipes with valid PCAPs but a deliberately wrong
        # expected file. Both targets must propagate the verifier's failure.
        probe = work / "make_probe"
        probe.mkdir()
        names = ["Makefile", "build/pcap_replay_test", "tests/pcap_replay_test.c", "dpdk/rx_core.h",
                 "lab8/__init__.py", "lab8/packets/__init__.py", "lab8/packets/verify_frames.py"]
        if args.dpdk:
            names.extend(["build/dpdk_rx", "dpdk/dpdk_rx.c"])
        for name in names:
            (probe / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, probe / name)
        gen.write_pcap(probe / "data/pcaps/images_inorder.pcap", gen.build_frames_manual(payloads))
        for mode in ("shuffled", "messy"):
            shutil.copy2(probe / "data/pcaps/images_inorder.pcap", probe / f"data/pcaps/images_{mode}.pcap")
        wrong = bytearray(original)
        wrong[0] ^= 1
        (probe / "data/frames").mkdir(parents=True, exist_ok=True)
        (probe / "data/frames/frames_original.bin").write_bytes(wrong)
        for target in (["test_rx", "run"] if args.dpdk else ["test_rx"]):
            result = invoke(["make", "-o", "pcaps", target, "SUDO="], probe)
            require(result.returncode != 0, result)
            require("15/16 images identical byte for byte" in result.stdout, result)
            print(f"PASS make {target}: byte mismatch propagates failure, exit={result.returncode}", flush=True)
    print("All packet pipeline regression checks passed.")


if __name__ == "__main__":
    main()
