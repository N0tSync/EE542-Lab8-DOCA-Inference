"""Compare GPU-resident ResNet-18 inference for the nine Part 2 cases."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

import tensorrt as trt
import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18

from trt_runner import TensorRTRunner


def measure(call, stream, warmup, iterations):
    for _ in range(warmup):
        call()
    stream.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(iterations)]
    # Materialize event handles before timing; reuse fixed inputs and TRT outputs.
    for start, end in events:
        start.record(stream)
        end.record(stream)
    stream.synchronize()
    for start, end in events:
        start.record(stream)
        call()
        end.record(stream)
    stream.synchronize()
    samples = [start.elapsed_time(end) for start, end in events]
    return statistics.mean(samples), statistics.pstdev(samples)


@torch.inference_mode()
def benchmark(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.hub.set_dir(str(args.model_dir / "torch_cache"))
    stream = torch.cuda.Stream()
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().cuda()
    runners = {precision: TensorRTRunner(args.model_dir / f"resnet18_{precision}.engine", stream=stream)
               for precision in ("fp32", "fp16")}
    builds = {}
    for precision in runners:
        path = args.model_dir / f"resnet18_{precision}.engine"
        builds[precision] = json.loads(path.with_suffix(".engine.json").read_text())
        if builds[precision]["tf32"]:
            raise ValueError("Rebuild engines with TF32 disabled")
        if hashlib.sha256(path.read_bytes()).hexdigest() != builds[precision]["sha256"]:
            raise ValueError(f"Engine metadata does not match {path}")
    stream.wait_stream(torch.cuda.current_stream())
    rows = []
    with torch.cuda.stream(stream):
        for batch in (1, 8, 32):
            x = torch.randn(batch, 3, 224, 224, device="cuda", dtype=torch.float32)
            x_half = x.half()
            reference = model(x)
            if not torch.isfinite(reference).all().item():
                raise RuntimeError("Non-finite PyTorch logits")
            mean, std = measure(lambda: model(x), stream, args.warmup, args.iterations)
            baseline = mean
            rows.append(dict(implementation="pytorch_fp32", batch=batch, latency_ms=mean,
                             latency_std_ms=std, throughput_images_s=batch * 1000 / mean,
                             speedup_vs_pytorch=1.0, max_abs_logit_diff=0.0, top1_agreement=1.0))
            for precision, runner in runners.items():
                data = x if precision == "fp32" else x_half
                output = torch.empty((batch, 1000), device="cuda", dtype=runner.output_dtype)
                runner(data, out=output)
                if not torch.isfinite(output).all().item():
                    raise RuntimeError(f"Non-finite TensorRT {precision} logits")
                difference = (output.float() - reference).abs().max().item()
                agreement = (output.argmax(1) == reference.argmax(1)).float().mean().item()
                mean, std = measure(lambda: runner(data, out=output), stream, args.warmup, args.iterations)
                rows.append(dict(implementation=f"tensorrt_{precision}", batch=batch, latency_ms=mean,
                                 latency_std_ms=std, throughput_images_s=batch * 1000 / mean,
                                 speedup_vs_pytorch=baseline / mean, max_abs_logit_diff=difference,
                                 top1_agreement=agreement))
            for row in rows[-3:]:
                print(f"{row['implementation']:16} batch={batch:2d}  "
                      f"{row['latency_ms']:.4f} ms  {row['throughput_images_s']:10.1f} images/s  "
                      f"{row['speedup_vs_pytorch']:.2f}x  "
                      f"max_diff={row['max_abs_logit_diff']:.6g}  "
                      f"top1={row['top1_agreement']:.1%}", flush=True)
    torch.cuda.synchronize()
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
        "python_torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "tensorrt": trt.__version__,
        "weights": "ResNet18_Weights.IMAGENET1K_V1", "seed": args.seed,
        "warmup": args.warmup, "iterations": args.iterations,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "timing": "CUDA events per forward on one non-default stream; eager execution, no CUDA graphs",
        "excluded": ["model loading", "engine building", "H2D", "preprocessing", "FP16 input conversion"],
        "input": "Seeded Gaussian GPU input; FP16 is converted from the same FP32 input",
        "accuracy_note": "Top-1 agreement on random inputs is not ImageNet accuracy",
        "engines": builds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "part2_benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "part2_summary.json").write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=542)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("Warmup and iterations must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
