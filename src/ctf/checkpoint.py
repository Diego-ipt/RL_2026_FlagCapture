"""Almacenamiento de checkpoints y métricas por run.

runs/<run>/
  config.json            hiperparámetros del run
  metrics.csv            una fila por época (epoch, steps, winrate, ...)
  ckpt/epoch_000010.pt   {"epoch","steps","models":{nombre:state_dict},"optim":{...},"meta":{...}}
  ckpt/latest.pt         copia del último
  index.json             lista de checkpoints (para elegir instancias/oponentes del pool)
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import torch


class RunStore:
    def __init__(self, run_name, root="runs"):
        self.dir = Path(root) / run_name
        self.ckpt_dir = self.dir / "ckpt"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.dir / "index.json"
        self.metrics_path = self.dir / "metrics.csv"

    def save_config(self, cfg: dict):
        (self.dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    def _fields(self, row):
        """Columnas fijadas por la primera fila (las claves nuevas se ignoran)."""
        if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
            with open(self.metrics_path, newline="") as f:
                return next(csv.reader(f))
        return list(row)

    def log(self, **row):
        fields = self._fields(row)
        new = not self.metrics_path.exists() or self.metrics_path.stat().st_size == 0
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)

    def save(self, epoch, steps, models: dict, optim: dict | None = None, meta: dict | None = None):
        """models/optim: {nombre: objeto con state_dict()}. Devuelve la ruta."""
        blob = {"epoch": epoch, "steps": steps, "meta": meta or {},
                "models": {k: m.state_dict() for k, m in models.items()},
                "optim": {k: o.state_dict() for k, o in (optim or {}).items()}}
        path = self.ckpt_dir / f"epoch_{epoch:06d}.pt"
        torch.save(blob, path)
        shutil.copyfile(path, self.ckpt_dir / "latest.pt")
        idx = self.list()
        idx.append({"epoch": epoch, "steps": steps, "path": str(path.name), "meta": meta or {}})
        self.index_path.write_text(json.dumps(idx, indent=2))
        return path

    def list(self):
        return json.loads(self.index_path.read_text()) if self.index_path.exists() else []

    def load(self, which="latest", map_location="cpu"):
        """which: 'latest' o número de época."""
        name = "latest.pt" if which == "latest" else f"epoch_{int(which):06d}.pt"
        return torch.load(self.ckpt_dir / name, map_location=map_location)

    def restore(self, models: dict, optim: dict | None = None, which="latest"):
        blob = self.load(which)
        for k, m in models.items():
            m.load_state_dict(blob["models"][k])
        for k, o in (optim or {}).items():
            o.load_state_dict(blob["optim"][k])
        return blob["epoch"], blob["steps"]
