# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.2", "transformers>=4.44", "numpy>=1.26"]
# ///
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interp_sae import (
    SparseAutoencoder,
    capture_activations,
    top_activating_tokens,
    feature_stats,
    steering_examples,
    build_report,
    save_report,
    load_report,
)


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
        self.forward = {}
        self.inverse = {}

    def _id(self, word):
        if word not in self.forward:
            token = abs(hash(word)) % self.vocab
            self.forward[word] = token
            self.inverse[token] = word
        return self.forward[word]

    def __call__(self, prompts, return_tensors=None, padding=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        length = max(len(p.split()) for p in prompts)
        rows = []
        for p in prompts:
            ids = [self._id(w) for w in p.split()]
            ids = ids + [0] * (length - len(ids))
            rows.append(ids)
        return {"input_ids": torch.tensor(rows)}

    def decode(self, ids):
        return " ".join(self.inverse.get(int(i), str(int(i))) for i in ids)


def identity_sae(dim):
    sae = SparseAutoencoder(dim, dim)
    with torch.no_grad():
        sae.W_enc.copy_(torch.eye(dim))
        sae.W_dec.copy_(torch.eye(dim))
        sae.b_enc.zero_()
        sae.b_dec.zero_()
    return sae


def test_top_activating_tokens_schema_and_order():
    torch.manual_seed(0)
    model, config = tiny_model()
    tokenizer = StubTokenizer(config.vocab_size)
    sae = SparseAutoencoder(config.n_embd, config.n_embd * 4)
    prompts = ["one two three", "four five six seven"]
    feature_ids = [0, 1]
    k = 3
    result = top_activating_tokens(model, tokenizer, sae, prompts, "resid", 1, feature_ids, k=k)
    assert set(result.keys()) == set(feature_ids)
    for fid in feature_ids:
        records = result[fid]
        assert len(records) <= k
        values = [r["activation"] for r in records]
        assert values == sorted(values, reverse=True)
        for r in records:
            assert set(r.keys()) == {"feature", "activation", "prompt", "position", "token"}
            assert isinstance(r["feature"], int) and r["feature"] == fid
            assert isinstance(r["activation"], float)
            assert isinstance(r["prompt"], int)
            assert isinstance(r["position"], int)
            assert isinstance(r["token"], str)
            json.dumps(r)


def test_top_activating_tokens_points_at_true_max():
    torch.manual_seed(1)
    model, config = tiny_model()
    tokenizer = StubTokenizer(config.vocab_size)
    sae = SparseAutoencoder(config.n_embd, config.n_embd * 4)
    prompts = ["one two three", "four five six seven"]
    fid = 2
    result = top_activating_tokens(model, tokenizer, sae, prompts, "resid", 1, [fid], k=4)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    acts = capture_activations(model, inputs, points=("resid",))["resid"][1]
    batch, seq, width = acts.shape
    with torch.no_grad():
        feats = sae.encode(acts.reshape(batch * seq, width)).reshape(batch, seq, -1)
    column = feats[:, :, fid].reshape(-1)
    best = int(column.argmax().item())
    expected_b, expected_s = divmod(best, seq)
    top = result[fid][0]
    assert top["prompt"] == expected_b
    assert top["position"] == expected_s
    assert abs(top["activation"] - column[best].item()) < 1e-6


def test_feature_stats_known_frequencies():
    sae = identity_sae(3)
    data = torch.tensor([
        [1.0, 1.0, -1.0],
        [2.0, -1.0, -2.0],
        [3.0, 1.0, -3.0],
        [4.0, -1.0, -4.0],
    ])
    stats = feature_stats(sae, data)
    assert len(stats) == 3
    frequencies = [row["frequency"] for row in stats]
    assert frequencies == sorted(frequencies, reverse=True)
    for row in stats:
        assert 0.0 <= row["frequency"] <= 1.0
        assert torch.isfinite(torch.tensor(row["mean"]))
    by_feature = {row["feature"]: row for row in stats}
    assert by_feature[0]["frequency"] == 1.0
    assert by_feature[1]["frequency"] == 0.5
    assert by_feature[2]["frequency"] == 0.0
    assert abs(by_feature[0]["mean"] - 2.5) < 1e-6
    assert abs(by_feature[2]["mean"] - 0.0) < 1e-6


def test_steering_examples_one_per_coef():
    torch.manual_seed(0)
    model, config = tiny_model()
    tokenizer = StubTokenizer(config.vocab_size)
    direction = torch.randn(config.n_embd)
    coefs = [0.0, 5.0, 10.0]
    examples = steering_examples(model, tokenizer, "one two three", 1, direction, coefs, max_new_tokens=3)
    assert len(examples) == len(coefs)
    for example, coef in zip(examples, coefs):
        assert example["coef"] == coef
        assert isinstance(example["text"], str)


def test_build_and_round_trip():
    torch.manual_seed(0)
    model, config = tiny_model()
    tokenizer = StubTokenizer(config.vocab_size)
    sae = SparseAutoencoder(config.n_embd, config.n_embd * 2)
    prompts = ["one two three", "four five six seven"]
    direction = torch.randn(config.n_embd)
    steering = {
        "prompt": "one two three",
        "layer": 1,
        "direction": direction,
        "coefs": [0.0, 4.0],
        "max_new_tokens": 2,
    }
    report = build_report(
        model, tokenizer, sae, prompts, "resid", 1,
        top_features=3, k=2, steering=steering,
    )
    assert report["point"] == "resid"
    assert report["layer"] == 1
    assert len(report["stats"]) == config.n_embd * 2
    assert len(report["top_tokens"]) == 3
    assert len(report["steering"]) == 2

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        save_report(report, path)
        loaded = load_report(path)
        assert loaded == report
        with open(path) as f:
            plain = json.load(f)
        assert plain == report
    finally:
        os.remove(path)


def main():
    tests = [
        test_top_activating_tokens_schema_and_order,
        test_top_activating_tokens_points_at_true_max,
        test_feature_stats_known_frequencies,
        test_steering_examples_one_per_coef,
        test_build_and_round_trip,
    ]
    for test in tests:
        test()
        print(f"{test.__name__} ok")
    print("all passed")


if __name__ == "__main__":
    main()
