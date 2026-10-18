"""Actor descentralizado (obs local) y crítico centralizado (estado global + ID)."""
import numpy as np
import torch
from torch import nn


def mlp(inp, out, hidden=128, out_gain=1.0):
    layers = [nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out)]
    for i, l in enumerate(m for m in layers if isinstance(m, nn.Linear)):
        nn.init.orthogonal_(l.weight, out_gain if i == 2 else np.sqrt(2))
        nn.init.zeros_(l.bias)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = mlp(obs_dim, n_actions, hidden, out_gain=0.01)

    def dist(self, obs, mask=None):
        """mask: bool [..., n_actions]; las acciones inválidas quedan con prob. 0."""
        logits = self.net(obs)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden, out_gain=1.0)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class ValueNorm:
    """Media/varianza móvil de los retornos; el crítico predice valores normalizados."""

    def __init__(self, eps=1e-5):
        self.mean, self.var, self.count, self.eps = 0.0, 1.0, eps, eps

    def update(self, x):
        x = np.asarray(x, np.float64).ravel()
        bm, bv, bc = x.mean(), x.var(), x.size
        d, tot = bm - self.mean, self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + self.eps)

    def denormalize(self, x):
        return x * np.sqrt(self.var + self.eps) + self.mean

    def state_dict(self):
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]
