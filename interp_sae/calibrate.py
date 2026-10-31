import torch

from .capture import capture_activations
from .steer import steer, project


def _generate(model, ids, layer, direction, coef, tokens):
    generated = ids
    with steer(model, layer, direction, coef):
        with torch.no_grad():
            for _ in range(tokens):
                logits = model(input_ids=generated).logits
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, nxt], dim=1)
    return generated


def _projection(model, ids, layer, direction, coef):
    with steer(model, layer, direction, coef):
        acts = capture_activations(model, {"input_ids": ids}, points=("resid",))["resid"][layer]
    return project(acts, direction).mean().item()


def _trigram_repetition(text):
    words = text.split()
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(trigrams)) / len(trigrams)


def strength(model, tokenizer, prompt, layer, direction, coef, tokens=4):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    full = _generate(model, ids, layer, direction, coef, tokens)
    return _projection(model, full, layer, direction, coef)


def coherence(model, tokenizer, ids):
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = ids[:, 1:]
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mean_lp = token_lp.mean().item()
    text = tokenizer.decode(ids[0].tolist())
    return mean_lp - _trigram_repetition(text)


def calibrate(model, tokenizer, prompt, layer, direction, coefs, min_coherence=None, tokens=4):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    rows = []
    for coef in coefs:
        full = _generate(model, ids, layer, direction, coef, tokens)
        rows.append({
            "coef": coef,
            "strength": _projection(model, full, layer, direction, coef),
            "coherence": coherence(model, tokenizer, full),
        })
    baseline = [r["coherence"] for r in rows if r["coef"] == 0]
    anchor = baseline[0] if baseline else max(r["coherence"] for r in rows)
    threshold = min_coherence if min_coherence is not None else anchor - 0.1 * abs(anchor)
    eligible = [r["coef"] for r in rows if r["coherence"] >= threshold]
    selected = max(eligible) if eligible else min(r["coef"] for r in rows)
    return rows, selected
