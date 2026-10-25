# Part 1 environment

Measured on 2026-10-24, America/Los_Angeles. Raw checks are saved in
[`results/environment.json`](../results/environment.json) and its log.

| Component | Observed value |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS, WSL2 |
| Kernel | 6.6.87.2-microsoft-standard-WSL2 |
| GPU | NVIDIA GeForce RTX 5080, compute capability 12.0, 16303 MiB |
| NVIDIA-SMI version field | 610.57.01 |
| Windows KMD version | 610.88 |
| CUDA UMD version field | 13.3 |
| CUDA Toolkit compiler | nvcc 13.0.88, `/usr/local/cuda-13.0/bin/nvcc` |
| CUDA Toolkit package | cuda-toolkit-13-0 13.0.3-1 |
| Python | 3.12.13, Conda `ee542-lab8` |
| PyTorch / torchvision | 2.10.0+cu130 / 0.25.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| TensorRT | 11.0.0.114, `tensorrt-cu13` |
| ONNX / onnxscript | 1.23.2 / 0.7.2 |
| NumPy / Scapy | 2.5.2 / 2.8.0 |
| DPDK | 23.11.4, Ubuntu package 23.11.4-0ubuntu0.24.04.2 |
| libpcap | 1.10.4 |
| RDMA core packages | 50.0-2ubuntu0.2 |
| libibverbs pkg-config version | 1.14.50.0 |
| C++ compiler | GCC 13.3.0 |

The driver/UMD CUDA field, installed Toolkit, and PyTorch's bundled CUDA runtime
are different version records. `/usr/local` also contains CUDA 13.2, but this
shell resolves `nvcc` to CUDA 13.0. Parts 1-2 need no environment upgrade.

## Verified checks

- `nvcc --version` and `pkg-config --modversion libdpdk` succeed.
- `python -m pip check` reports no broken requirements.
- A 512x512 CUDA matrix multiplication completes with finite output.
- A TensorRT Builder is created successfully.
- A C program including `rte_version.h` compiles, links with the installed
  `libdpdk` pkg-config flags, and prints `DPDK 23.11.4`.
- Required development packages for DPDK, pcap, and RDMA are installed.

The DPDK check establishes headers and linker availability. Packet reception,
ordered/shuffled reconstruction, and DOCA capabilities are separate team
validations. Meson and Ninja are absent from this local PATH; neither is needed
for these Python inference scripts or the installed DPDK compile check. Install
them on the Part 6 build host before building DOCA/GPUNetIO examples.

## System setup reference

These commands document the setup for a fresh WSL Ubuntu environment. Existing
packages on the measured host were inspected, not reinstalled.

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config git wget ca-certificates python3-venv \
  pciutils dpdk dpdk-dev libdpdk-dev libpcap-dev rdma-core libibverbs-dev ibverbs-utils
```

The CUDA WSL package repository must be configured before installing the Toolkit:

```bash
wget -O /tmp/lab8-cuda-keyring.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/lab8-cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
export PATH=/usr/local/cuda-13.0/bin:$PATH
```

For persistent zsh use, add the PATH line once to `~/.zshrc`. The Windows host
provides the GPU driver for WSL. Use the Toolkit-only package, following
[NVIDIA's WSL guidance](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
The Python setup is in the [Part 2 guide](part2.md); `requirements-lock.txt` records
the actual package versions rather than an untested upgrade range.

## Scope and portability

The assignment recommends Ubuntu 22.04 with an L4/T4 cloud GPU and explicitly
allows Windows NVIDIA GPU users to complete Parts 1-5 under WSL2. This run uses
Ubuntu 24.04 WSL2 and RTX 5080 with CUDA 13 libraries. The implementation and
results were checked against Parts 1-2 of `lab8-f26.pdf`, supplied in the parent
workspace directory. Subsequent DPDK reassembly, GPU classification, and pinned
H2D measurements were also completed on this WSL host; see the [Part 5 guide](part5.md)
and its linked results. The original environment snapshot above predates those
tests. Parts 6-7 remain uncompleted. Part 6 specifies a full GPU cloud virtual
machine. The local shell currently has no Docker CLI or NVIDIA Container
Toolkit executable in PATH; no local container results are claimed.

The agent sandbox initially blocked NVML access. GPU checks, export, engine
building, runner validation, and benchmark must run with host GPU access. This
restriction is separate from the validated WSL GPU configuration.

TensorRT 11 removes precision-enabling flags and always uses strongly typed
networks; see the [official Python migration guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-python-api-patterns.html).
The implementation was checked against the installed 11.0 bindings as well.
Stream handling follows [PyTorch 2.10 CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html).
