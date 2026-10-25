# EE542 Lab 8: DOCA Inference

CNN inference on network packets with NVIDIA DOCA. The current team scope is
Parts 1-6: environment setup, ResNet-18 inference with PyTorch and TensorRT,
image packet generation and reconstruction with DPDK, GPU preprocessing and
classification, and DOCA/GPUNetIO exploration.
Inference uses pretrained ResNet-18 weights; no model training is required.

## Software environment

- Ubuntu Linux; WSL2 can be used for Parts 1-5.
- NVIDIA GPU with a compatible driver and CUDA Toolkit.
- Python 3.12, PyTorch, torchvision, TensorRT 11, ONNX, onnxscript, NumPy, and Scapy.
- C/C++ build tools, pkg-config, DPDK, libpcap, and RDMA development libraries.
- Docker and NVIDIA DOCA on a supported GPU host for Part 6.

Detailed setup, inference usage, and measurements are in
[the environment notes](docs/environment.md), [the Part 2 guide](docs/part2.md),
[the packet format](docs/packet_format.md), [the Part 5 guide](docs/part5.md),
and [the benchmark results](results/part2/part2_results.md).
See [repository layout](docs/layout.md) for module and generated-file locations.
