"""Part B: train the two-convolution CNN with MSE + Adam on the GPU, with careful timing.

Usage:
    python train.py --subset 1000 --epochs 1      # small-sample validation
    python train.py --epochs 10                   # full experiment
"""
import argparse
import json
import os
import platform
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from model import DATASETS, build_model


def sync(device):
    """Wait for all queued GPU work. CUDA kernels run asynchronously, so a timer
    read without this can stop before the GPU has actually finished."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_loaders(dataset, batch_size, subset, data_dir, workers, pin):
    _, _, mean, std = DATASETS[dataset]
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    cls = {"cifar10": torchvision.datasets.CIFAR10,
           "fashionmnist": torchvision.datasets.FashionMNIST,
           "mnist": torchvision.datasets.MNIST}[dataset]
    train_set = cls(data_dir, train=True, download=True, transform=tf)
    test_set = cls(data_dir, train=False, download=True, transform=tf)
    if subset:
        g = torch.Generator().manual_seed(0)
        train_set = Subset(train_set, torch.randperm(len(train_set), generator=g)[:subset].tolist())
        test_set = Subset(test_set, torch.randperm(len(test_set), generator=g)[:max(subset // 5, 100)].tolist())
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin,
              persistent_workers=workers > 0)
    return (DataLoader(train_set, shuffle=True, drop_last=False, **kw),
            DataLoader(test_set, shuffle=False, **kw))


def mse_loss(outputs, labels, num_classes=10):
    target = F.one_hot(labels, num_classes).float()
    return F.mse_loss(outputs, target)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy, mean loss, total test time, and forward-only time (seconds).

    total_time  : wall clock for the whole test pass (data loading, host-to-device
                  copies, forward, loss, accuracy), bracketed by synchronize().
    forward_time: only the model(x) calls, measured with CUDA events on the GPU
                  (perf_counter + synchronize on CPU).
    """
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    forward_time = 0.0
    use_events = device.type == "cuda"

    sync(device)
    t0 = time.perf_counter()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_events:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(x)
            end.record()
            end.synchronize()
            forward_time += start.elapsed_time(end) / 1000.0
        else:
            f0 = time.perf_counter()
            out = model(x)
            forward_time += time.perf_counter() - f0
        loss_sum += mse_loss(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    sync(device)
    total_time = time.perf_counter() - t0

    return {"accuracy": correct / total, "loss": loss_sum / total, "samples": total,
            "total_test_time_s": total_time, "forward_only_time_s": forward_time}


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    pin = device.type == "cuda"
    train_loader, test_loader = get_loaders(args.dataset, args.batch_size, args.subset,
                                            args.data_dir, args.workers, pin)

    model = build_model(args.dataset).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device),
           "python": platform.python_version()}
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
        env["capability"] = "sm_%d%d" % torch.cuda.get_device_capability(device)
        env["arch_list"] = torch.cuda.get_arch_list()
    print("Environment:", json.dumps(env, indent=2))
    print(f"Model: flatten dim = {model.flat_dim}, parameters = "
          f"{sum(p.numel() for p in model.parameters()):,}")

    history = []
    sync(device)
    t_train = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        sync(device)
        t_ep = time.perf_counter()
        loss_sum, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = mse_loss(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * y.size(0)   # stays on GPU: no sync per batch
            n += y.size(0)
        sync(device)
        ep_time = time.perf_counter() - t_ep
        ep_loss = (loss_sum / n).item()
        history.append({"epoch": epoch, "train_loss": ep_loss, "time_s": ep_time})
        print(f"epoch {epoch:2d}  train MSE {ep_loss:.5f}  {ep_time:.2f} s")
    sync(device)
    train_time = time.perf_counter() - t_train

    test = evaluate(model, test_loader, device)
    print(f"\nTraining time: {train_time:.2f} s for {args.epochs} epochs")
    print(f"Test accuracy: {test['accuracy'] * 100:.2f}%  (MSE {test['loss']:.5f}, "
          f"{test['samples']} images)")
    print(f"Total test time: {test['total_test_time_s'] * 1000:.1f} ms   "
          f"forward-only: {test['forward_only_time_s'] * 1000:.1f} ms")

    results = {"args": vars(args), "env": env, "history": history,
               "train_time_s": train_time, "test": test}
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "dataset": args.dataset, "results": results}, args.ckpt)
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved checkpoint to {args.ckpt} and results to {args.results}")
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10", choices=list(DATASETS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--subset", type=int, default=0, help="use only N training images (0 = all)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--ckpt", default="checkpoints/twoconv.pt")
    p.add_argument("--results", default="results.json")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":   # guard needed for DataLoader workers on Windows
    train(parse_args())
