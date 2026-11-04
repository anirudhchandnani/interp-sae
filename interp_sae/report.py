import json

import torch

from .capture import capture_activations, capture_from_text
from .steer import steer
from .train import flatten_activations


def top_activating_tokens(model, tokenizer, sae, prompts, point, layer, feature_ids, k=5):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    acts = capture_activations(model, inputs, points=(point,))
    hidden = acts[point][layer]
    batch, seq, width = hidden.shape
    ids = inputs["input_ids"]
    with torch.no_grad():
        features = sae.encode(hidden.reshape(batch * seq, width)).reshape(batch, seq, -1)
    result = {}
    for fid in feature_ids:
        records = []
        for b in range(batch):
            for s in range(seq):
                records.append({
                    "feature": int(fid),
                    "activation": float(features[b, s, fid].item()),
                    "prompt": int(b),
                    "position": int(s),
                    "token": tokenizer.decode([int(ids[b, s].item())]),
                })
        records.sort(key=lambda r: r["activation"], reverse=True)
        result[fid] = records[:k]
    return result


def feature_stats(sae, data):
    with torch.no_grad():
        features = sae.encode(data)
    frequency = (features > 0).float().mean(dim=0)
    mean = features.mean(dim=0)
    stats = []
    for i in range(features.shape[1]):
        stats.append({
            "feature": int(i),
            "frequency": float(frequency[i].item()),
            "mean": float(mean[i].item()),
        })
    stats.sort(key=lambda r: r["frequency"], reverse=True)
    return stats


def steering_examples(model, tokenizer, prompt, layer, direction, coefs, max_new_tokens=8):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    examples = []
    for coef in coefs:
        generated = ids
        with steer(model, layer, direction, coef):
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    logits = model(input_ids=generated).logits
                    nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, nxt], dim=1)
        examples.append({
            "coef": float(coef),
            "text": tokenizer.decode(generated[0].tolist()),
        })
    return examples


def build_report(model, tokenizer, sae, prompts, point, layer, data=None, top_features=5, k=5, steering=None):
    if data is None:
        acts = capture_from_text(model, tokenizer, prompts, points=(point,))
        data = flatten_activations(acts, point, layer)
    stats = feature_stats(sae, data)
    feature_ids = [row["feature"] for row in stats[:top_features]]
    tokens = top_activating_tokens(model, tokenizer, sae, prompts, point, layer, feature_ids, k=k)
    report = {
        "point": point,
        "layer": int(layer),
        "stats": stats,
        "top_tokens": {str(fid): tokens[fid] for fid in feature_ids},
    }
    if steering is not None:
        report["steering"] = steering_examples(model, tokenizer, **steering)
    return report


def save_report(report, path):
    with open(path, "w") as f:
        json.dump(report, f)


def load_report(path):
    with open(path) as f:
        return json.load(f)
