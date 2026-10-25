# Part 2 measured results

Measured on 2026-10-24 (America/Los_Angeles) using NVIDIA GeForce RTX 5080
under Ubuntu 24.04 WSL2. PyTorch 2.10.0+cu130, torchvision 0.25.0+cu130,
TensorRT 11.0.0.114, CUDA runtime 13.0. ResNet18_Weights.IMAGENET1K_V1.

Seed 542; 50 warmup and 200 timed forwards per case. TF32 is disabled in
PyTorch and TensorRT. Timing uses CUDA events on one non-default stream.
Model loading, engine building, H2D, preprocessing, and FP16 conversion
are outside the timed region. These are eager Python inference measurements;
host submission gaps and WSL/display scheduling can affect the timings.

| Implementation | Batch | Mean (ms) | Std (ms) | Images/s | Speedup | Max logit error | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pytorch_fp32 | 1 | 1.6209 | 0.5896 | 616.9 | 1.00x | 0 | 100.000% |
| tensorrt_fp32 | 1 | 0.9889 | 0.1385 | 1011.2 | 1.64x | 7.62939e-06 | 100.000% |
| tensorrt_fp16 | 1 | 0.2834 | 0.0690 | 3528.2 | 5.72x | 0.026525 | 100.000% |
| pytorch_fp32 | 8 | 4.5111 | 0.3283 | 1773.4 | 1.00x | 0 | 100.000% |
| tensorrt_fp32 | 8 | 1.5688 | 0.1707 | 5099.5 | 2.88x | 1.04904e-05 | 100.000% |
| tensorrt_fp16 | 8 | 0.3724 | 0.0730 | 21482.8 | 12.11x | 0.0200236 | 100.000% |
| pytorch_fp32 | 32 | 8.4478 | 0.4735 | 3788.0 | 1.00x | 0 | 100.000% |
| tensorrt_fp32 | 32 | 4.8094 | 0.3418 | 6653.7 | 1.76x | 8.22544e-06 | 100.000% |
| tensorrt_fp16 | 32 | 1.1148 | 0.1333 | 28705.3 | 7.58x | 0.0252023 | 96.875% |

Speedup is the same-batch PyTorch FP32 mean divided by the TensorRT mean.
Throughput is batch * 1000 / mean_ms. Errors compare TensorRT logits, cast
to FP32, with PyTorch FP32 on the same inputs; FP16 inputs are converted
from the FP32 inputs. Each batch uses one random input tensor for all repeats.
Top-1 agreement is measured over that batch, not over 200 independent batches.
It is not ImageNet accuracy.

TensorRT FP32 agrees with PyTorch top-1 for all 41 sampled images. FP16
agrees for batch 1 and 8, and for 31/32 images at batch 32 (96.875%).
The FP16 top-1 difference is retained in the results; no 100% agreement
requirement or ImageNet accuracy threshold is assumed.

The batch-1 PyTorch standard deviation is 0.5896 ms (about 36% of its mean),
so the reported means describe this run rather than fixed GPU hardware limits.

Both engines built and deserialized successfully, with min/opt/max batch
1/8/32 and LINEAR GPU buffers. The FP32 build took 7.403 s and the FP16 build
6.652 s (excluded from inference timing). Both ONNX models passed the checker
with opset 18, symbolic batch, and the expected FP32/FP16 I/O types.

Runner validation passed for batch transitions 1 -> 8 -> 32 -> 1 in both
precisions, caller/runner stream handoff, output buffer reuse, and rejection
of CPU inputs, wrong dtype/layout, out-of-range batch, and wrong output shape.

Raw records: [CSV](part2_benchmark.csv), [JSON with build metadata](part2_summary.json),
[benchmark log](benchmark.log), [runner validation](runner_validation.log),
[build log](build_engine.log), and [ONNX export log](export_onnx.log).
See [environment checks](../environment/environment.json) and [usage](../../docs/part2.md).

Parts 3-6 integration and DOCA validation are outside these measurements.
Rebuild the engines on the destination GPU and TensorRT installation.
