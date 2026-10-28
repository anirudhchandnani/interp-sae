import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_enc = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.W_dec = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

    def encode(self, x):
        centered = x - self.b_dec
        return torch.relu(centered @ self.W_enc.t() + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec.t() + self.b_dec

    def forward(self, x):
        features = self.encode(x)
        recon = self.decode(features)
        return recon, features

    def loss(self, x, l1_coef):
        recon, features = self.forward(x)
        recon_loss = ((recon - x) ** 2).mean()
        l1 = features.abs().sum(dim=-1).mean()
        total = recon_loss + l1_coef * l1
        return {"recon": recon_loss, "l1": l1, "total": total}
