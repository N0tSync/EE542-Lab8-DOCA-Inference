# Part 5: received-image classification and H2D measurement

`classify_frames.py` connects the Parts 3-4 raw RGB outputs to the Part 2 FP16
runner. It uses the existing pretrained ResNet-18 engine, without model training.
All steps below were validated on RTX 5080 under Ubuntu WSL2.

## Inputs and outputs

Each input holds 16 RGB uint8 HWC frames of shape 224x224x3, ordered by frame ID,
with no file header. Each file must contain exactly 2,408,448 bytes.
The defaults are `frames_original.bin`, `frames_rx_inorder.bin`,
`frames_rx_shuffled.bin`, and `frames_rx_messy.bin`. The existing `make run`
also copies the shuffled result to `frames_rx.bin` for a single-file handoff.

Outputs in `results/`:

| File | Contents |
| --- | --- |
| `part5_classification.csv` | Per-image byte equality, original/received class IDs and names, class agreement, and logit error |
| `part5_h2d.csv` | Pinned H2D copy statistics for the original and each reconstructed file |
| `part5_summary.json` | Results, file/engine hashes, preprocessing rules, environment, and overall pass/fail |
| `part5.log` | Actual program output |

The program returns nonzero if either byte equality or class agreement fails.
Matching classes alone cannot hide damaged transport data. Files with an
incorrect byte count are rejected before inference.

## Run from the repository root

Activate the existing Python environment and build the FP16 engine using the
[Part 2 guide](part2.md) if it is not already available. No extra Python packages
are required beyond the recorded dependencies.

```bash
conda activate ee542-lab8
mkdir -p results
set -o pipefail
python -u test_packet_pipeline.py --dpdk 2>&1 | tee results/part34_regression.log
make run SUDO= 2>&1 | tee results/part34_validation.log
python -u classify_frames.py --warmup 50 --iterations 200 2>&1 | tee results/part5.log
```

`SUDO=` was sufficient for the pcap virtual port on the measured WSL host. On
hosts requiring elevated DPDK access, use the default `make run` instead. Make
stops if reassembly or byte verification fails. A convenient combined command
after the engine is available is `make part5 SUDO=`. The `PYTHON` and `EAL`
Makefile variables can be overridden for other environments.

For a single received file:

```bash
python classify_frames.py --received frames_rx.bin
```

`--original`, `--received`, `--engine`, `--output-dir`, `--num-frames`,
`--warmup`, and `--iterations` are configurable. A single invocation supports
1-32 frames, matching the engine's batch profile; the lab run uses 16.

## GPU preprocessing and inference

The script transfers raw uint8 RGB bytes, then changes NHWC to NCHW on GPU.
It applies the full `ResNet18_Weights.IMAGENET1K_V1.transforms()` preset on GPU:
bilinear resize of the short side to 256 with antialiasing, center crop to 224,
scaling to [0,1], and FP32 normalization with mean [0.485,0.456,0.406] and std
[0.229,0.224,0.225]. Finally it casts to contiguous FP16 for `TensorRTRunner`.
The full preset is applied even though the transported frames are already
224x224. Original and received images always follow the same path.

One non-default CUDA stream orders copies, preprocessing, and inference.
CPU result access occurs after the necessary GPU work. The classification
table compares the same FP16 engine with itself on original and received data;
this is separate from the FP16-versus-PyTorch comparison in Part 2.
Synthetic-image agreement establishes transport consistency, not ImageNet
classification accuracy.

## H2D timing

The host source is a preallocated pinned uint8 buffer and the destination is a
preallocated CUDA uint8 buffer. File reading and CPU filling occur outside the
measurement. CUDA events bracket only `copy_(source, non_blocking=True)` on the
copy's stream. Event handles are initialized before measurement and results are
read only after synchronization. Each file uses 50 warmup and 200 timed copies.

The CSV reports bytes, mean, population standard deviation, min/max latency,
and effective decimal GB/s. Preprocessing, FP16 conversion, inference, file I/O,
and allocations are excluded. Small-copy event durations can include host
submission and WSL scheduling gaps; the numbers describe this run.

This is the measured host-to-GPU staging copy that a suitable GPUNetIO receive
path can avoid. It does not establish a complete GPUNetIO end-to-end speedup.

## Verification and scope

The packet regression script tests real DPDK and the shared replay core for
missing, overlapping, zero-length, inconsistent, and short chunks; valid retries;
expected-frame completion; and Makefile propagation of comparison failures.
The three normal DPDK inputs are verified independently before classification.
Additional GPU checks covered analytic black/white normalization, a one-byte
input mismatch with unchanged class labels, and truncated-file rejection.

See [measured results](../results/part5_results.md),
[packet regression output](../results/part34_regression.log),
[DPDK validation](../results/part34_validation.log), and
[Part 5 checks](../results/part5_validation.log).

The assignment permits WSL2 for Parts 1-5. Part 6 specifies a full GPU cloud VM.
GPU containers can also be developed locally using the WSL2 backend, as
documented in [Docker's GPU guide](https://docs.docker.com/desktop/features/gpu/).
Local container experiments would need their own capability/build/run logs and
must be identified as WSL results. Docker GPU access alone does not establish
the DOCA networking capabilities or fulfill the assignment's cloud-host test.
No Docker or Part 6 experiments have been performed by this implementation.
