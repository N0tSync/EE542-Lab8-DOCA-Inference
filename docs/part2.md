# Part 2 inference guide

Local Part 1 and Part 2 implementation for EE 542 Lab 8. The team is currently
working on Parts 1-6. This implementation uses an RTX 5080 under WSL2, with the
`ee542-lab8` Conda environment. Parts 3-5 integration is documented in the
[Part 5 guide](part5.md); Part 6 remains pending.

## Files

| File | Purpose |
| --- | --- |
| `lab8/check_env.py` | Record versions, check Python dependencies, GPU math, TensorRT Builder, and DPDK compilation/linking |
| `lab8/inference/export_onnx.py` | Export pretrained ResNet-18 as FP32 and FP16 ONNX, opset 18, dynamic batch |
| `lab8/inference/build_engine.py` | Build and inspect TensorRT 11 engines with batch profile min/opt/max = 1/8/32 |
| `lab8/inference/trt_runner.py` | Reusable CUDA tensor input/output runner with stream ordering |
| `tests/test_trt_runner.py` | GPU checks for dynamic shapes, buffer reuse, stream handoff, and invalid input rejection |
| `lab8/inference/benchmark.py` | Compare PyTorch FP32, TensorRT FP32, and TensorRT FP16 at batches 1, 8, 32 |
| `requirements-lock.txt` | Full Python package snapshot from the validated environment |
| `docs/environment.md` | Host configuration, installation, checks, and limitations |
| `results/` | Measured CSV/JSON results and actual execution logs |

The earlier two-convolution training experiment is archived on branch
`legacy-cnn-training` at commit `81fa203`. The `main` branch contains the Lab 8
inference work, using pretrained torchvision ResNet-18 without a training step.

## Setup

The existing environment already has the required inference packages. Activate it:

```bash
conda activate ee542-lab8
```

For a fresh environment on the same platform:

```bash
conda create -n ee542-lab8 python=3.12.13 pip -y
conda activate ee542-lab8
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-lock.txt
```

See [environment setup](environment.md) for system dependencies. In a
noninteractive shell, use `conda run --no-capture-output -n ee542-lab8 python ...`
or `/home/n0tsync/miniconda3/envs/ee542-lab8/bin/python` on the measured host.

## Run

Run from the repository root. The first export downloads the pinned pretrained
weights into `models/torch_cache/`. Keep both `.onnx` and `.onnx.data` files when
moving the models. Engines, weights, and caches are ignored by Git.

```bash
mkdir -p results/environment results/part2
set -o pipefail
python -u -m lab8.check_env 2>&1 | tee results/environment/environment.log
python -u -m lab8.inference.export_onnx 2>&1 | tee results/part2/export_onnx.log
python -u -m lab8.inference.build_engine 2>&1 | tee results/part2/build_engine.log
python -u -m tests.test_trt_runner 2>&1 | tee results/part2/runner_validation.log
python -u -m lab8.inference.benchmark --warmup 50 --iterations 200 --seed 542 2>&1 | tee results/part2/benchmark.log
```

Check each command's exit status before continuing. The output engines are
`models/resnet18_fp32.engine` and `models/resnet18_fp16.engine`. Each has an
`.engine.json` sidecar with its hash, build configuration, profile, and I/O types.
The benchmark verifies these sidecars and writes `results/part2/part2_benchmark.csv`
and `results/part2/part2_summary.json`. Rebuild engines on the deployment GPU with the
matching TensorRT version; the RTX 5080 plans are specific to this environment.

## Benchmark method

All three paths use `ResNet18_Weights.IMAGENET1K_V1` in evaluation mode. FP16
weights come from the same FP32 model; FP16 inputs are converted from each
batch's original FP32 tensor. Both PyTorch and TensorRT have TF32 disabled.
TensorRT 11 uses strongly typed networks, so FP16 comes from the FP16 ONNX
graph. The removed `BuilderFlag.FP16` API is not used.

Each case uses 50 warmup forwards and 200 timed forwards, with CUDA events on
the same non-default stream. Inputs and TensorRT output buffers are allocated
before timing. Loading, building, H2D copies, FP16 conversion, and preprocessing
are excluded. This measures eager inference as submitted from Python; GPU idle
gaps caused by host submission or WSL scheduling can affect event durations.
No CUDA graph capture is used. Means and population standard deviations are
reported, along with throughput and speedup relative to the same batch's
PyTorch result.

Numerical checks report maximum absolute logit error and top-1 agreement using
the same input values. Random-input agreement is not ImageNet classification
accuracy. See [measured results](../results/part2/part2_results.md) for this machine.

## Runner interface for Part 5

```python
import torch
from lab8.inference.trt_runner import TensorRTRunner

runner = TensorRTRunner("models/resnet18_fp16.engine")
# Example input; Part 5 supplies the GPU-preprocessed RGB tensor.
x = torch.randn(8, 3, 224, 224, device="cuda", dtype=runner.input_dtype)
logits = runner(x)  # CUDA tensor [8, 1000], no CPU copy
classes = logits.argmax(dim=1)
print(classes.cpu())  # CPU access synchronizes as needed.
runner.synchronize()  # Complete work before releasing the runner.
```

The input name is `input`, output name is `logits`; batch range is 1-32.
The FP16 engine requires contiguous FP16 NCHW input and returns FP16 logits;
the FP32 engine uses FP32. Input and output must be on the runner's GPU.
For repeated calls, allocate `out` once and use `runner(x, out=out)` to avoid
output allocation. The caller owns the buffers and must finish consuming an
output before reusing it.

One runner owns one context and one execution stream. It waits for the caller's
current stream before inference, then makes that stream wait for its output.
This enqueues GPU dependencies without a CPU synchronization for a repeated
shape; changing batch synchronizes the runner before updating context shapes.
Use one runner per host thread. Keep it alive until its GPU work completes.
If input was produced on a different stream, make the caller stream wait for
that producer first. If another stream consumes logits, make it wait for the
caller stream and use `logits.record_stream(consumer)` for allocator safety.
See the class docstring for the complete lifetime contract.

Part 5 preprocessing is implemented in `lab8/integration/classify_frames.py`. RGB uint8 HWC data
becomes NCHW on GPU, then the full `IMAGENET1K_V1.transforms()` preset resizes
the short side to 256, center crops to 224, scales by 1/255, and normalizes with
mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`. It then casts to
contiguous FP16. This also applies to the already 224x224 transport frames and
keeps the original and reassembled paths identical. See [the Part 5 guide](part5.md).
