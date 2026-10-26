# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "transformers>=4.44", "numpy>=1.26"]
# ///
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp_sae import capture_activations


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


def test_resid_shapes():
    model, config = tiny_model()
    batch, seq = 2, 5
    inputs = make_inputs(batch, seq, config.vocab_size)
    acts = capture_activations(model, inputs, points=("resid",))
    resid = acts["resid"]
    assert resid.shape == (config.n_layer, batch, seq, config.n_embd)


def test_mlp_shapes():
    model, config = tiny_model()
    batch, seq = 3, 4
    inputs = make_inputs(batch, seq, config.vocab_size)
    acts = capture_activations(model, inputs, points=("mlp",))
    mlp = acts["mlp"]
    assert mlp.shape == (config.n_layer, batch, seq, config.n_embd)


def test_both_points_and_layer_count():
    model, config = tiny_model()
    inputs = make_inputs(1, 6, config.vocab_size)
    acts = capture_activations(model, inputs, points=("resid", "mlp"))
    assert set(acts.keys()) == {"resid", "mlp"}
    assert acts["resid"].shape[0] == config.n_layer
    assert acts["mlp"].shape[0] == config.n_layer


def test_hooks_removed():
    model, config = tiny_model()
    inputs = make_inputs(1, 3, config.vocab_size)
    capture_activations(model, inputs)
    for block in model.transformer.h:
        assert len(block._forward_hooks) == 0
        assert len(block.mlp._forward_hooks) == 0


def main():
    tests = [
        test_resid_shapes,
        test_mlp_shapes,
        test_both_points_and_layer_count,
        test_hooks_removed,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
