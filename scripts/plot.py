"""Grafica curvas de varios runs.  python scripts/plot.py --runs v2_1v1 v2_1v1_nobeta"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--out", default="runs/plots/compare.png")
ap.add_argument("--smooth", type=int, default=5)
ap.add_argument("--keys", nargs="+", default=["win_rate", "ep_len", "entropy", "critic_loss"],
                help="columnas de metrics.csv (p.ej. strips_learner passes_learner draw_rate)")
a = ap.parse_args()

keys = a.keys
cols = 2 if len(keys) <= 4 else 3
rows = -(-len(keys) // cols)
fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.5 * rows), squeeze=False)
for run in a.runs:
    df = pd.read_csv(Path("runs") / run / "metrics.csv")
    for ax, k in zip(axs.flat, keys):
        if k not in df:
            continue
        ax.plot(df["env_steps"], df[k].rolling(a.smooth, min_periods=1).mean(), label=run)
for ax, k in zip(axs.flat, keys):
    ax.set_title(k)
    ax.set_xlabel("env_steps")
    ax.grid(alpha=.3)
axs[0, 0].legend()
fig.tight_layout()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=120)
print("guardado", a.out)
