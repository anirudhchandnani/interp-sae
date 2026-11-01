import torch

from .capture import capture_from_text
from .sae import SparseAutoencoder


def flatten_activations(acts, point, layer):
    selected = acts[point][layer]
    batch, seq, hidden = selected.shape
    return selected.reshape(batch * seq, hidden)


def train_sae(sae, data, l1_coef, lr=1e-3, steps=200, batch_size=64):
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = data.shape[0]
    history = []
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),))
        batch = data[idx]
        losses = sae.loss(batch, l1_coef)
        opt.zero_grad()
        losses["total"].backward()
        opt.step()
        history.append({k: v.detach().item() for k, v in losses.items()})
    return history


def train_on_model(model, tokenizer, prompts, point, layer, hidden_ratio=4, l1_coef=1e-3, **kwargs):
    acts = capture_from_text(model, tokenizer, prompts, points=(point,))
    data = flatten_activations(acts, point, layer)
    hidden = data.shape[1]
    sae = SparseAutoencoder(input_dim=hidden, hidden_dim=hidden * hidden_ratio)
    history = train_sae(sae, data, l1_coef, **kwargs)
    return sae, history
