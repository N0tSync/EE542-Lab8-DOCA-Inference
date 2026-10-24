"""Build TensorRT 11 engines from typed FP32/FP16 ONNX models."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import tensorrt as trt
import torch


def build_engine(model_dir, precision, workspace_gib):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TensorRT 11 is always strongly typed.
    parser = trt.OnnxParser(network, logger)
    source = model_dir / f"resnet18_{precision}.onnx"
    if not parser.parse_from_file(str(source.resolve())):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Could not parse {source}:\n{errors}")
    expected = trt.float32 if precision == "fp32" else trt.float16
    assert network.num_inputs == network.num_outputs == 1
    assert network.get_input(0).name == "input"
    assert tuple(network.get_input(0).shape) == (-1, 3, 224, 224)
    assert network.get_output(0).name == "logits"
    assert tuple(network.get_output(0).shape) == (-1, 1000)
    assert network.get_input(0).dtype == network.get_output(0).dtype == expected

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gib * 2**30))
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224))
    if not profile:
        raise RuntimeError("Invalid optimization profile")
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Could not add the optimization profile")

    print(f"Building {precision}; TF32 disabled; workspace {workspace_gib:g} GiB", flush=True)
    start = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    elapsed = time.perf_counter() - start
    if plan is None:
        raise RuntimeError(f"Engine build failed: {source}")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Could not deserialize the newly built engine")
    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        assert engine.get_tensor_dtype(name) == expected
        io.append({"name": name, "dtype": str(engine.get_tensor_dtype(name)),
                   "shape": list(engine.get_tensor_shape(name)),
                   "format": str(engine.get_tensor_format(name))})
    shapes = [list(s) for s in engine.get_tensor_profile_shape("input", 0)]
    target = model_dir / f"resnet18_{precision}.engine"
    payload = bytes(plan)
    target.write_bytes(payload)
    metadata = {
        "engine": str(target), "sha256": hashlib.sha256(payload).hexdigest(),
        "tensorrt": trt.__version__, "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "precision": precision, "tf32": False, "workspace_gib": workspace_gib,
        "build_seconds": elapsed, "profile_min_opt_max": shapes, "io": io,
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
    }
    target.with_suffix(".engine.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--precision", choices=["fp32", "fp16", "both"], default="both")
    parser.add_argument("--workspace-gib", type=float, default=2)
    args = parser.parse_args()
    if int(trt.__version__.split(".")[0]) != 11:
        parser.error("This builder targets TensorRT 11")
    if args.workspace_gib <= 0:
        parser.error("--workspace-gib must be positive")
    torch.cuda.init()
    for precision in (["fp32", "fp16"] if args.precision == "both" else [args.precision]):
        build_engine(args.model_dir, precision, args.workspace_gib)


if __name__ == "__main__":
    main()
