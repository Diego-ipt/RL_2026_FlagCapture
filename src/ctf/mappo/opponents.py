"""Políticas para jugar/evaluar. Todas aceptan `env` para leer la máscara de acciones."""
import numpy as np
import torch

from ctf.env import N_ACTIONS


def _mask(env, a, masks=None):
    if masks is not None and a in masks:
        return masks[a]
    return env.action_mask(a) if env is not None else np.ones(N_ACTIONS, bool)


class RandomPolicy:
    """Acción uniforme entre las válidas."""

    def __init__(self, seed=None, n_actions=N_ACTIONS):
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, obs: dict, env=None, masks=None) -> dict:
        return {a: int(self.rng.choice(np.flatnonzero(_mask(env, a, masks)))) for a in obs}


class ActorPolicy:
    """Envuelve un Actor entrenado."""

    def __init__(self, actor, stochastic=False):
        self.actor, self.stochastic = actor, stochastic

    @torch.no_grad()
    def act(self, obs: dict, env=None, masks=None) -> dict:
        keys = list(obs)
        if not keys:
            return {}
        x = torch.as_tensor(np.stack([obs[k] for k in keys]))
        m = torch.as_tensor(np.stack([_mask(env, k, masks) for k in keys]))
        d = self.actor.dist(x, m)
        acts = d.sample() if self.stochastic else d.probs.argmax(-1)
        return dict(zip(keys, acts.tolist()))


class TeamPolicy:
    """Un actor por equipo (dos MAPPO distintos): usa el del color de cada agente."""

    def __init__(self, actors: dict, stochastic=False):
        self.pols = {tm: ActorPolicy(a, stochastic) for tm, a in actors.items()}

    def act(self, obs: dict, env=None, masks=None) -> dict:
        out = {}
        for tm, pol in self.pols.items():
            out.update(pol.act({k: o for k, o in obs.items() if k.startswith(tm)}, env, masks))
        return out
