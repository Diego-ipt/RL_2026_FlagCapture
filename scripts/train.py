"""Entrena MAPPO.  docker compose run --rm ctf python scripts/train.py --config configs/mappo_v2_rules.yaml"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import yaml  # noqa: E402

from ctf.mappo.trainer import MAPPOTrainer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/mappo_v2_rules.yaml")
ap.add_argument("--resume", action="store_true", help="continúa desde runs/<run>/ckpt/latest.pt")
ap.add_argument("--run-name", help="sobrescribe run_name de la config")
ap.add_argument("--iterations", type=int)
ap.add_argument("--set", nargs="*", default=[], help="overrides clave=valor, p.ej. beta0=0 env.n_per_team=2")
a = ap.parse_args()

cfg = yaml.safe_load(open(a.config))
if a.run_name:
    cfg["run_name"] = a.run_name
for kv in a.set:
    k, v = kv.split("=", 1)
    d = cfg
    *path, last = k.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = yaml.safe_load(v)
tr = MAPPOTrainer(cfg)
if a.resume:
    tr.resume()
    print(f"reanudando desde epoch {tr.epoch}")
tr.train(a.iterations)
