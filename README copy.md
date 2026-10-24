# Part B: PyTorch and GPU Experiment

Two-convolution CNN (output ReLU) trained with MSE on one-hot labels and Adam, on CIFAR-10 by default.

| File | Purpose |
|---|---|
| `model.py` | `TwoConvNet`: conv 3→32 → pool → conv 32→64 → pool → FC 4096→512→10 → ReLU |
| `train.py` | Training, CUDA-synchronized timing, test evaluation, saves checkpoint + `results.json` |
| `test_behavior.py` | Five behavior tests (no download needed) |
| `reload_check.py` | Reloads the checkpoint and checks it reproduces the saved test results |

## 1. Check the environment supports the RTX 5080

The RTX 5080 is Blackwell (compute capability `sm_120`), so PyTorch must be built with CUDA 12.8 or newer.

```bash
conda activate <your-env>
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())"
```

Expect `(12, 0)` and `sm_120` in the arch list. If not: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## 2. Run, in this order

```bash
python test_behavior.py                       # five behavior tests -> 5/5 passed
python train.py --subset 1000 --epochs 1 --ckpt checkpoints/smoke.pt --results smoke.json   # small-sample validation
python train.py --epochs 10                   # full experiment
python reload_check.py                        # checkpoint reproduces test results
```

On Windows, if DataLoader workers cause trouble, add `--workers 0`.
Other datasets: `--dataset fashionmnist` or `--dataset mnist`.

## Timing method

- CUDA launches kernels asynchronously, so every timer is bracketed by `torch.cuda.synchronize()`.
- Training time covers all epochs; the per-batch loss stays on the GPU so no hidden sync is added inside the loop.
- **Total test time**: wall clock for the whole test pass (data loading, host-to-device copy, forward, loss, accuracy).
- **Forward-only time**: just the `model(x)` calls, measured with CUDA events and summed over batches.

Record from the output: GPU name, PyTorch/CUDA versions, training time, test accuracy, total vs forward-only test time.
