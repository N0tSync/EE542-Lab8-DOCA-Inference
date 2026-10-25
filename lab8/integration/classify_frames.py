"""Part 5: classify original/reassembled RGB frames and measure pinned H2D copies."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import statistics
from zoneinfo import ZoneInfo

import numpy as np
import tensorrt as trt
import torch
import torchvision
from torchvision.models import ResNet18_Weights

from lab8.inference.trt_runner import TensorRTRunner

FRAME_SHAPE = (224, 224, 3)
FRAME_BYTES = 224 * 224 * 3


def read_frames(path, count):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size != count * FRAME_BYTES:
        raise ValueError(f"{path}: expected {count * FRAME_BYTES} bytes, got {data.size}")
    return data.reshape(count, *FRAME_SHAPE)


def measure_h2d(source, destination, stream, warmup, iterations):
    if not source.is_pinned() or source.device.type != "cpu":
        raise ValueError("The H2D source must be pinned CPU memory")
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(iterations)]
    # Create event handles before measurement; all allocations and CPU fills
    # have also finished before entering the measured region.
    for start, end in events:
        start.record(stream)
        end.record(stream)
    for _ in range(warmup):
        destination.copy_(source, non_blocking=True)
    stream.synchronize()
    for start, end in events:
        start.record(stream)
        destination.copy_(source, non_blocking=True)
        end.record(stream)
    stream.synchronize()
    samples = [start.elapsed_time(end) for start, end in events]
    mean = statistics.mean(samples)
    size = source.numel() * source.element_size()
    return {"batch": len(source), "bytes": size, "pinned": True,
            "warmup": warmup, "iterations": iterations,
            "mean_ms": mean, "std_ms": statistics.pstdev(samples),
            "min_ms": min(samples), "max_ms": max(samples),
            "bandwidth_gb_s": size / (mean * 1e6)}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def classify(args):
    inputs = [args.original, *args.received]
    arrays = [read_frames(path, args.num_frames) for path in inputs]
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stream = torch.cuda.Stream()
    runner = TensorRTRunner(args.engine, stream=stream)
    if runner.input_dtype != torch.float16:
        raise ValueError("Part 5 requires the FP16 engine")
    weights = ResNet18_Weights.IMAGENET1K_V1
    transforms = weights.transforms()
    categories = weights.meta["categories"]
    pinned = torch.empty(arrays[0].shape, dtype=torch.uint8, pin_memory=True)
    raw_gpu = torch.empty(arrays[0].shape, dtype=torch.uint8, device=runner.device)
    stream.wait_stream(torch.cuda.current_stream(runner.device))
    h2d_rows, rows, datasets = [], [], []
    original_logits = None

    with torch.cuda.stream(stream):
        for index, (path, array) in enumerate(zip(inputs, arrays)):
            pinned.copy_(torch.from_numpy(array))
            timing = measure_h2d(pinned, raw_gpu, stream, args.warmup, args.iterations)
            h2d_rows.append({"dataset": path.stem, "input_path": str(path), **timing})
            # Apply the official preset entirely on GPU, including resize and
            # center crop. Casting to FP16 happens after FP32 normalization.
            model_input = transforms(raw_gpu.permute(0, 3, 1, 2))
            model_input = model_input.to(dtype=runner.input_dtype).contiguous()
            if (model_input.device != runner.device
                    or tuple(model_input.shape) != (args.num_frames, 3, 224, 224)):
                raise RuntimeError("Unexpected preprocessing output")
            if not torch.isfinite(model_input).all().item():
                raise RuntimeError("Non-finite preprocessed input")
            logits = runner(model_input).float().cpu().numpy()
            if not np.isfinite(logits).all():
                raise RuntimeError("Non-finite logits")
            classes = logits.argmax(axis=1)
            datasets.append({"path": str(path), "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                             "class_ids": classes.tolist(),
                             "class_names": [categories[int(value)] for value in classes]})
            if index == 0:
                original_logits = logits
            else:
                original_classes = original_logits.argmax(axis=1)
                for frame_id in range(args.num_frames):
                    original_class = int(original_classes[frame_id])
                    received_class = int(classes[frame_id])
                    rows.append({
                        "received_file": str(path), "frame_id": frame_id,
                        "bytes_identical": bool(np.array_equal(arrays[0][frame_id], array[frame_id])),
                        "original_class_id": original_class, "received_class_id": received_class,
                        "original_class": categories[original_class],
                        "received_class": categories[received_class],
                        "class_agreement": original_class == received_class,
                        "max_abs_logit_diff": float(np.max(np.abs(original_logits[frame_id] - logits[frame_id]))),
                    })
                group = rows[-args.num_frames:]
                byte_count = sum(row["bytes_identical"] for row in group)
                class_count = sum(row["class_agreement"] for row in group)
                print(f"{path}: bytes {byte_count}/{args.num_frames}, classes {class_count}/{args.num_frames}", flush=True)
            print(f"{path}: pinned H2D {timing['bytes']} bytes, "
                  f"{timing['mean_ms']:.6f} +/- {timing['std_ms']:.6f} ms, "
                  f"{timing['bandwidth_gb_s']:.3f} GB/s", flush=True)
    runner.synchronize()
    passed = all(row["bytes_identical"] and row["class_agreement"] for row in rows)
    metadata = {
        "timestamp": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(), "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda, "tensorrt": trt.__version__,
        "engine": str(args.engine), "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "weights": "ResNet18_Weights.IMAGENET1K_V1", "frames": args.num_frames,
        "preprocessing": {"device": "cuda", "input": "RGB uint8 NHWC",
                          "resize_short_side": transforms.resize_size,
                          "center_crop": transforms.crop_size, "mean": transforms.mean,
                          "std": transforms.std, "output": "contiguous FP16 NCHW",
                          "preset": str(transforms)},
        "h2d_timing": "CUDA events around pinned uint8 CPU -> GPU copy on one non-default stream",
        "h2d_excluded": ["file read", "allocation", "CPU buffer fill", "GPU preprocessing", "inference"],
        "accuracy_note": "Synthetic-image class agreement is a transport check, not ImageNet accuracy",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "part5_classification.csv", rows)
    write_csv(args.output_dir / "part5_h2d.csv", h2d_rows)
    summary = {"metadata": metadata, "passed": passed, "datasets": datasets,
               "classification": rows, "h2d": h2d_rows}
    (args.output_dir / "part5_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Part 5 validation: " + ("PASS" if passed else "FAIL"), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("data/frames/frames_original.bin"))
    parser.add_argument("--received", type=Path, nargs="+", default=[Path(f"data/frames/frames_rx_{mode}.bin")
                        for mode in ("inorder", "shuffled", "messy")])
    parser.add_argument("--engine", type=Path, default=Path("models/resnet18_fp16.engine"))
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=Path("results/part5"))
    args = parser.parse_args()
    if not 1 <= args.num_frames <= 32 or args.warmup < 1 or args.iterations < 1:
        parser.error("Expected 1-32 frames and positive warmup/iterations")
    try:
        passed = classify(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
