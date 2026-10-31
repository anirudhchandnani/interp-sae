# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "transformers>=4.44", "numpy>=1.26"]
# ///
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp_sae import capture_activations, steer, ramp, strength, coherence, calibrate


def tiny_model():
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_embd=16,
        n_layer=3,
        n_head=2,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model, config


class StubTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = [int(t) for t in text.split()]
        return {"input_ids": torch.tensor([ids])}

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def resid_at(model, inputs, layer, direction, coef):
    with torch.no_grad():
        with steer(model, layer, direction, coef):
            return capture_activations(model, inputs, points=("resid",))["resid"][layer]


def test_ramp_schedule_applied_per_step():
    torch.manual_seed(0)
    model, config = tiny_model()
    inputs = {"input_ids": torch.randint(0, config.vocab_size, (2, 4))}
    direction = torch.randn(config.n_embd)
    layer = 1
    with torch.no_grad():
        base = capture_activations(model, inputs, points=("resid",))["resid"][layer]
        with steer(model, layer, direction, ramp(2.0, 8.0, 4)):
            first = capture_activations(model, inputs, points=("resid",))["resid"][layer]
            later = capture_activations(model, inputs, points=("resid",))["resid"][layer]
    assert torch.allclose(first - base, (2.0 * direction).expand_as(base), atol=1e-4)
    assert torch.allclose(later - base, (4.0 * direction).expand_as(base), atol=1e-4)


def test_constant_coef_backward_compatible():
    torch.manual_seed(0)
    model, config = tiny_model()
    inputs = {"input_ids": torch.randint(0, config.vocab_size, (2, 4))}
    direction = torch.randn(config.n_embd)
    layer = 1
    with torch.no_grad():
        base = capture_activations(model, inputs, points=("resid",))["resid"][layer]
        with steer(model, layer, direction, 3.0):
            first = capture_activations(model, inputs, points=("resid",))["resid"][layer]
            later = capture_activations(model, inputs, points=("resid",))["resid"][layer]
    assert torch.allclose(first - base, (3.0 * direction).expand_as(base), atol=1e-4)
    assert torch.allclose(later - base, (3.0 * direction).expand_as(base), atol=1e-4)


def test_ramp_counter_scoped_and_hooks_removed():
    torch.manual_seed(0)
    model, config = tiny_model()
    inputs = {"input_ids": torch.randint(0, config.vocab_size, (2, 4))}
    direction = torch.randn(config.n_embd)
    layer = 1
    with torch.no_grad():
        base = capture_activations(model, inputs, points=("resid",))["resid"][layer]
        with steer(model, layer, direction, ramp(2.0, 8.0, 4)):
            capture_activations(model, inputs, points=("resid",))
            capture_activations(model, inputs, points=("resid",))
    for block in model.transformer.h:
        assert len(block._forward_hooks) == 0
    with torch.no_grad():
        with steer(model, layer, direction, ramp(2.0, 8.0, 4)):
            again = capture_activations(model, inputs, points=("resid",))["resid"][layer]
    assert torch.allclose(again - base, (2.0 * direction).expand_as(base), atol=1e-4)


def test_strength_increases_with_coef():
    torch.manual_seed(0)
    model, config = tiny_model()
    tokenizer = StubTokenizer()
    direction = torch.randn(config.n_embd)
    low = strength(model, tokenizer, "1 2 3", 1, direction, 0.0)
    high = strength(model, tokenizer, "1 2 3", 1, direction, 6.0)
    assert high > low


def test_coherence_penalizes_repetition():
    torch.manual_seed(0)
    model, config = tiny_model()
    tokenizer = StubTokenizer()
    repetitive = torch.tensor([[7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7]])
    varied = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    rep = coherence(model, tokenizer, repetitive)
    var = coherence(model, tokenizer, varied)
    assert torch.isfinite(torch.tensor(rep))
    assert torch.isfinite(torch.tensor(var))
    assert rep < var


def test_calibrate_table_and_selection():
    torch.manual_seed(0)
    model, config = tiny_model()
    tokenizer = StubTokenizer()
    direction = torch.randn(config.n_embd)
    coefs = [0.0, 1.0, 3.0, 10.0, 30.0]
    rows, selected = calibrate(model, tokenizer, "1 2 3 4 5", 1, direction, coefs, tokens=8)
    assert len(rows) == len(coefs)
    assert selected in coefs
    baseline = [r["coherence"] for r in rows if r["coef"] == 0.0][0]
    worst = min(r["coherence"] for r in rows)
    threshold = (baseline + worst) / 2
    _, strict = calibrate(model, tokenizer, "1 2 3 4 5", 1, direction, coefs, min_coherence=threshold, tokens=8)
    assert strict in coefs
    assert strict != max(coefs)


def main():
    tests = [
        test_ramp_schedule_applied_per_step,
        test_constant_coef_backward_compatible,
        test_ramp_counter_scoped_and_hooks_removed,
        test_strength_increases_with_coef,
        test_coherence_penalizes_repetition,
        test_calibrate_table_and_selection,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
