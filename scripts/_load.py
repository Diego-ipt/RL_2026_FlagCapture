"""Carga runs guardados y construye políticas desde un texto."""
import json
from pathlib import Path

from ctf.checkpoint import RunStore
from ctf.mappo.opponents import RandomPolicy, TeamPolicy
from ctf.mappo.trainer import build_actor


def load_actors(run, epoch="latest", root="runs"):
    """Devuelve ({equipo: actor}, cfg, época). Un run self-play tiene un actor por
    equipo; un run vs random solo tiene `actor_blue` (se usa para ambos colores)."""
    cfg = json.loads((Path(root) / run / "config.json").read_text())
    blob = RunStore(run, root).load(epoch)
    actors = {}
    for t in ("blue", "red"):
        key = f"actor_{t}" if f"actor_{t}" in blob["models"] else "actor_blue"
        a = build_actor(cfg)
        a.load_state_dict(blob["models"][key])
        a.eval()
        actors[t] = a
    return actors, cfg, blob["epoch"]


def make_policy(spec, seed=0, stochastic=False):
    """spec: 'random' o 'RUN' o 'RUN:EPOCH'. Devuelve (política, nombre, cfg o None)."""
    if spec in (None, "random"):
        return RandomPolicy(seed), "random", None
    run, _, epoch = spec.partition(":")
    actors, cfg, ep = load_actors(run, epoch or "latest")
    return TeamPolicy(actors, stochastic), f"{run}@{ep}", cfg
