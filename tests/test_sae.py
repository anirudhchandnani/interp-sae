# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy>=1.26"]
# ///
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "sae", Path(__file__).resolve().parent.parent / "interp_sae" / "sae.py"
)
_sae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sae)
SparseAutoencoder = _sae.SparseAutoencoder


def test_feature_shape_and_nonneg():
    torch.manual_seed(0)
    sae = SparseAutoencoder(input_dim=8, hidden_dim=16)
    x = torch.randn(5, 8)
    recon, features = sae(x)
    assert features.shape == (5, 16)
    assert torch.all(features >= 0)


def test_recon_shape():
    torch.manual_seed(1)
    sae = SparseAutoencoder(input_dim=8, hidden_dim=16)
    x = torch.randn(5, 8)
    recon, features = sae(x)
    assert recon.shape == (5, 8)


def test_loss_values_finite():
    torch.manual_seed(2)
    sae = SparseAutoencoder(input_dim=8, hidden_dim=16)
    x = torch.randn(4, 8)
    losses = sae.loss(x, l1_coef=0.01)
    assert set(losses.keys()) == {"recon", "l1", "total"}
    for value in losses.values():
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_learns():
    torch.manual_seed(3)
    sae = SparseAutoencoder(input_dim=6, hidden_dim=24)
    x = torch.randn(10, 6)
    opt = torch.optim.Adam(sae.parameters(), lr=0.01)
    start = sae.loss(x, l1_coef=0.0)["recon"].item()
    for _ in range(100):
        opt.zero_grad()
        loss = sae.loss(x, l1_coef=0.0)["recon"]
        loss.backward()
        opt.step()
    end = sae.loss(x, l1_coef=0.0)["recon"].item()
    assert end < start


def main():
    tests = [
        test_feature_shape_and_nonneg,
        test_recon_shape,
        test_loss_values_finite,
        test_learns,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
