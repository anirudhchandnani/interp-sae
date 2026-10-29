# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "numpy>=1.26"]
# ///
import importlib.util
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    "probe", Path(__file__).resolve().parent.parent / "interp_sae" / "probe.py"
)
_probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_probe)
LinearProbe = _probe.LinearProbe


def make_clusters(dim=6, n=200, gap=4.0, seed=0):
    torch.manual_seed(seed)
    true_dir = torch.zeros(dim)
    true_dir[0] = 1.0
    half = n // 2
    pos = torch.randn(half, dim) + true_dir * gap
    neg = torch.randn(half, dim) - true_dir * gap
    x = torch.cat([pos, neg], dim=0)
    y = torch.cat([torch.ones(half), torch.zeros(half)], dim=0)
    return x, y, true_dir


def test_fit_accuracy_high():
    x, y, _ = make_clusters()
    probe = LinearProbe(x.shape[1])
    probe.fit(x, y)
    assert probe.accuracy(x, y) >= 0.9


def test_direction_aligns():
    x, y, true_dir = make_clusters()
    probe = LinearProbe(x.shape[1])
    probe.fit(x, y)
    cos = torch.dot(probe.direction, true_dir / true_dir.norm()).item()
    assert cos > 0.9


def test_predict_shapes_and_proba_range():
    x, y, _ = make_clusters()
    probe = LinearProbe(x.shape[1])
    probe.fit(x, y, steps=50)
    proba = probe.predict_proba(x)
    preds = probe.predict(x)
    assert proba.shape == (x.shape[0],)
    assert preds.shape == (x.shape[0],)
    assert torch.all(proba >= 0.0)
    assert torch.all(proba <= 1.0)


def test_direction_normalized():
    x, y, _ = make_clusters()
    probe = LinearProbe(x.shape[1])
    probe.fit(x, y, steps=50)
    assert abs(probe.direction.norm().item() - 1.0) < 1e-5


def test_fit_improves_accuracy():
    x, y, _ = make_clusters()
    probe = LinearProbe(x.shape[1])
    start = probe.accuracy(x, y)
    probe.fit(x, y)
    end = probe.accuracy(x, y)
    assert end > start


def main():
    tests = [
        test_fit_accuracy_high,
        test_direction_aligns,
        test_predict_shapes_and_proba_range,
        test_direction_normalized,
        test_fit_improves_accuracy,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
