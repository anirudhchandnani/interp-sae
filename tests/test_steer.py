# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "transformers>=4.44", "numpy>=1.26"]
# ///
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp_sae import capture_activations, steer, project


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


def make_inputs(batch, seq, vocab):
    return {"input_ids": torch.randint(0, vocab, (batch, seq))}


def test_nonzero_coef_changes_logits():
    model, config = tiny_model()
    inputs = make_inputs(2, 5, config.vocab_size)
    direction = torch.randn(config.n_embd)
    with torch.no_grad():
        base = model(**inputs).logits
        with steer(model, 1, direction, 3.0):
            steered = model(**inputs).logits
    assert not torch.allclose(base, steered)


def test_zero_coef_unchanged():
    model, config = tiny_model()
    inputs = make_inputs(2, 5, config.vocab_size)
    direction = torch.randn(config.n_embd)
    with torch.no_grad():
        base = model(**inputs).logits
        with steer(model, 1, direction, 0.0):
            steered = model(**inputs).logits
    assert torch.allclose(base, steered, atol=1e-6)


def test_hooks_removed():
    model, config = tiny_model()
    inputs = make_inputs(1, 3, config.vocab_size)
    direction = torch.randn(config.n_embd)
    with torch.no_grad():
        with steer(model, 2, direction, 1.0):
            model(**inputs)
    for block in model.transformer.h:
        assert len(block._forward_hooks) == 0


def test_residual_shifted_by_coef_direction():
    model, config = tiny_model()
    inputs = make_inputs(2, 4, config.vocab_size)
    direction = torch.randn(config.n_embd)
    coef = 2.5
    layer = 1
    with torch.no_grad():
        base = capture_activations(model, inputs, points=("resid",))["resid"][layer]
        with steer(model, layer, direction, coef):
            shifted = capture_activations(model, inputs, points=("resid",))["resid"][layer]
    delta = shifted - base
    expected = coef * direction
    assert torch.allclose(delta, expected.expand_as(delta), atol=1e-5)


def test_project_shape_and_norm():
    acts = torch.randn(2, 5, 16)
    direction = torch.randn(16)
    scores = project(acts, direction)
    assert scores.shape == (2, 5)
    vector = torch.randn(16)
    recovered = project(vector, vector)
    assert abs(recovered.item() - vector.norm().item()) < 1e-5


def main():
    tests = [
        test_nonzero_coef_changes_logits,
        test_zero_coef_unchanged,
        test_hooks_removed,
        test_residual_shifted_by_coef_direction,
        test_project_shape_and_norm,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
