# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "transformers>=4.44", "numpy>=1.26"]
# ///
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp_sae import SparseAutoencoder, flatten_activations, train_sae, train_on_model


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
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, prompts, return_tensors=None, padding=None):
        length = max(len(p.split()) for p in prompts)
        rows = []
        for p in prompts:
            ids = [(abs(hash(w)) % self.vocab) for w in p.split()]
            ids = ids + [0] * (length - len(ids))
            rows.append(ids)
        return {"input_ids": torch.tensor(rows)}


def low_rank_data(n, hidden, rank):
    torch.manual_seed(0)
    basis = torch.randn(rank, hidden)
    coeffs = torch.randn(n, rank)
    return coeffs @ basis


def test_flatten_shapes_and_values():
    n_layer, batch, seq, hidden = 3, 2, 4, 5
    resid = torch.randn(n_layer, batch, seq, hidden)
    acts = {"resid": resid}
    flat = flatten_activations(acts, "resid", 1)
    assert flat.shape == (batch * seq, hidden)
    assert torch.equal(flat[0], resid[1, 0, 0])
    assert torch.equal(flat[seq], resid[1, 1, 0])
    assert torch.equal(flat[seq + 1], resid[1, 1, 1])


def test_train_reduces_loss():
    torch.manual_seed(1)
    data = low_rank_data(256, 12, 3)
    sae = SparseAutoencoder(input_dim=12, hidden_dim=48)
    steps = 150
    history = train_sae(sae, data, l1_coef=1e-3, lr=1e-2, steps=steps, batch_size=32)
    assert len(history) == steps
    for record in history:
        for value in record.values():
            assert torch.isfinite(torch.tensor(value))
    assert history[-1]["total"] < history[0]["total"]


def test_l1_drives_sparsity():
    torch.manual_seed(2)
    data = low_rank_data(256, 12, 3)
    sparse_sae = SparseAutoencoder(input_dim=12, hidden_dim=48)
    train_sae(sparse_sae, data, l1_coef=0.1, lr=1e-2, steps=200, batch_size=32)
    dense_sae = SparseAutoencoder(input_dim=12, hidden_dim=48)
    train_sae(dense_sae, data, l1_coef=0.0, lr=1e-2, steps=200, batch_size=32)
    sparse_frac = (sparse_sae.encode(data).abs() < 1e-6).float().mean().item()
    dense_frac = (dense_sae.encode(data).abs() < 1e-6).float().mean().item()
    assert sparse_frac > dense_frac


def test_train_on_model_end_to_end():
    model, config = tiny_model()
    tokenizer = StubTokenizer(config.vocab_size)
    prompts = ["the cat sat", "hello there world"]
    sae, history = train_on_model(
        model,
        tokenizer,
        prompts,
        point="resid",
        layer=1,
        hidden_ratio=4,
        l1_coef=1e-3,
        steps=5,
        batch_size=8,
    )
    assert isinstance(sae, SparseAutoencoder)
    assert sae.input_dim == config.n_embd
    assert len(history) == 5


def main():
    tests = [
        test_flatten_shapes_and_values,
        test_train_reduces_loss,
        test_l1_drives_sparsity,
        test_train_on_model_end_to_end,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
