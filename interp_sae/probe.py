import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)

    def fit(self, x, y, steps=300, lr=0.05):
        y = y.float()
        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = loss_fn(self.forward(x), y)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).long()

    def accuracy(self, x, y):
        preds = self.predict(x)
        return (preds == y.long()).float().mean().item()

    @property
    def direction(self):
        w = self.linear.weight.detach().squeeze(0)
        return w / w.norm()
