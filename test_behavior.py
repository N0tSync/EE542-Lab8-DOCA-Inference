"""Five behavior tests for the Part B model. No dataset download needed.

Run with:  python test_behavior.py      (or: pytest -q test_behavior.py)
"""
import io

import torch
import torch.nn as nn

from model import build_model
from train import mse_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make(seed=0):
    torch.manual_seed(seed)
    return build_model("cifar10").to(DEVICE)


def test_1_output_shape_and_relu():
    """Output is (batch, 10) and never negative because of the output ReLU."""
    m = make().eval()
    out = m(torch.randn(8, 3, 32, 32, device=DEVICE))
    assert out.shape == (8, 10)
    assert (out >= 0).all()
    assert isinstance(m.classifier[-1], nn.ReLU)


def test_2_layer_dimensions():
    """Two conv layers and FC dimensions 4096 -> 512 -> 10."""
    m = make()
    convs = [l for l in m.modules() if isinstance(l, nn.Conv2d)]
    fcs = [l for l in m.modules() if isinstance(l, nn.Linear)]
    assert len(convs) == 2
    assert [(f.in_features, f.out_features) for f in fcs] == [(64 * 8 * 8, 512), (512, 10)]
    feat = m.features(torch.zeros(1, 3, 32, 32, device=DEVICE))
    assert feat.shape == (1, 64, 8, 8)


def test_3_mse_on_one_hot():
    """Loss equals the MSE against one-hot targets, and is zero for a perfect output."""
    y = torch.tensor([0, 3, 9], device=DEVICE)
    perfect = torch.nn.functional.one_hot(y, 10).float()
    assert mse_loss(perfect, y).item() == 0.0
    out = torch.zeros(3, 10, device=DEVICE)
    assert abs(mse_loss(out, y).item() - 1 / 10) < 1e-7   # one 1 in each row of 10


def test_4_adam_steps_reduce_loss():
    """A few Adam steps on one fixed batch lower the loss, and every parameter gets a gradient."""
    m = make().train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(32, 3, 32, 32, generator=g).to(DEVICE)
    y = torch.randint(0, 10, (32,), generator=g).to(DEVICE)
    first = None
    for step in range(20):
        opt.zero_grad()
        loss = mse_loss(m(x), y)
        loss.backward()
        if step == 0:
            first = loss.item()
            assert all(p.grad is not None for p in m.parameters())
        opt.step()
    assert mse_loss(m(x), y).item() < first


def test_5_checkpoint_round_trip():
    """Saving and loading the state dict gives identical outputs in eval mode."""
    m = make(seed=3).eval()
    x = torch.randn(4, 3, 32, 32, device=DEVICE)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = make(seed=99)
    m2.load_state_dict(torch.load(buf, map_location=DEVICE))
    m2.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), rtol=0, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"Running {len(tests)} behavior tests on {DEVICE}")
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e!r}")
    print(f"{passed}/{len(tests)} passed")
    raise SystemExit(0 if passed == len(tests) else 1)
