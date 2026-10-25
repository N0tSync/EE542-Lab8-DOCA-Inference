"""Exercise dynamic batches, buffer reuse, validation, and stream handoff on GPU."""

import argparse
from pathlib import Path

import torch

from lab8.inference.trt_runner import TensorRTRunner


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    torch.manual_seed(542)
    for precision in ("fp32", "fp16"):
        runner = TensorRTRunner(args.model_dir / f"resnet18_{precision}.engine")
        caller = torch.cuda.Stream()
        for batch in (1, 8, 32, 1):
            with torch.cuda.stream(caller):
                x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=runner.input_dtype)
                out = torch.empty(batch, 1000, device="cuda", dtype=runner.output_dtype)
                assert runner(x, out=out) is out
                consumed = out.clone()  # Must wait for the runner's output automatically.
            caller.synchronize()
            with torch.cuda.stream(runner.stream):
                direct = runner(x)
            runner.synchronize()
            torch.testing.assert_close(consumed, direct, rtol=0, atol=0)
            assert torch.isfinite(consumed).all().item()
            print(f"PASS {precision}: batch {batch}, output reuse and stream handoff", flush=True)
        wrong_dtype = torch.float16 if runner.input_dtype == torch.float32 else torch.float32
        rejects(lambda: runner(x.cpu()))
        rejects(lambda: runner(x.to(dtype=wrong_dtype)))
        rejects(lambda: runner(x.transpose(2, 3)))
        rejects(lambda: runner(torch.empty(33, 3, 224, 224, device="cuda", dtype=runner.input_dtype)))
        rejects(lambda: runner(x, out=torch.empty(1, 999, device="cuda", dtype=runner.output_dtype)))
        runner.synchronize()
        print(f"PASS {precision}: CPU, dtype, layout, batch range, output shape rejection", flush=True)
    print("All TensorRT runner checks passed.")


if __name__ == "__main__":
    main()
