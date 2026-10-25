"""Compare frames_original.bin and frames_rx.bin byte for byte, frame by frame.

Both files hold raw RGB bytes (224 x 224 x 3) of frame 0, 1, ... back to back.

Usage:
    python -m lab8.packets.verify_frames data/frames/frames_original.bin data/frames/frames_rx.bin
"""
import sys

FRAME = 224 * 224 * 3


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = open(sys.argv[1], "rb").read(), open(sys.argv[2], "rb").read()
    if len(a) % FRAME or len(b) % FRAME:
        raise SystemExit(f"file sizes {len(a)} / {len(b)} are not a multiple of {FRAME}")
    n = len(a) // FRAME
    if len(b) != len(a):
        print(f"WARNING: {len(a) // FRAME} original frames but {len(b) // FRAME} received frames")
    ok = 0
    for i in range(n):
        fa, fb = a[i * FRAME:(i + 1) * FRAME], b[i * FRAME:(i + 1) * FRAME]
        if fa == fb:
            ok += 1
            print(f"frame {i:2d}: identical")
        elif not fb:
            print(f"frame {i:2d}: MISSING")
        else:
            diff = sum(x != y for x, y in zip(fa, fb))
            print(f"frame {i:2d}: MISMATCH ({diff} bytes differ)")
    print(f"{ok}/{n} images identical byte for byte")
    raise SystemExit(0 if ok == n and len(a) == len(b) else 1)


if __name__ == "__main__":
    main()
