"""Reload the saved checkpoint, re-run the test set, and compare with the saved results.

Usage:
    python reload_check.py --ckpt checkpoints/twoconv.pt
"""
import argparse

import torch

from model import build_model
from train import evaluate, get_loaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/twoconv.pt")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    saved = ckpt["results"]
    args = saved["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(ckpt["dataset"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    _, test_loader = get_loaders(ckpt["dataset"], args["batch_size"], args["subset"],
                                 a.data_dir, a.workers, device.type == "cuda")
    new = evaluate(model, test_loader, device)

    old = saved["test"]
    print(f"Saved   : accuracy {old['accuracy'] * 100:.2f}%  MSE {old['loss']:.6f}")
    print(f"Reloaded: accuracy {new['accuracy'] * 100:.2f}%  MSE {new['loss']:.6f}")
    same_acc = new["accuracy"] == old["accuracy"]
    close_loss = abs(new["loss"] - old["loss"]) < 1e-5
    if same_acc and close_loss:
        print("PASS: reloaded checkpoint reproduces the test-set results")
    else:
        raise SystemExit("FAIL: reloaded results differ from the saved results")


if __name__ == "__main__":
    main()
