"""Grafica el marcador de suma cero durante partidas entre dos políticas.
docker compose run --rm ctf python scripts/marker_trace.py --blue sp_2v2 --red sp_2v2:50
Arriba: M_azul(t) y M_rojo(t) = -M_azul(t). Abajo: retorno acumulado de cada equipo
(cambios del marcador + victoria); al final del episodio vale exactamente ±r_win o 0."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _load import make_policy  # noqa: E402
from ctf.env import CTFParallelEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--blue", default="random")
ap.add_argument("--red", default="random")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--beta", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="runs/plots/marker_trace.png")
a = ap.parse_args()

blue, nb, cb = make_policy(a.blue, a.seed)
red, nr, cr = make_policy(a.red, a.seed + 1)
env_kw = (cb or cr or {"env": {"size": 10, "n_per_team": 2}})["env"]
env = CTFParallelEnv(**{**env_kw, "shaping_scale": a.beta})
fig, axs = plt.subplots(2, a.episodes, figsize=(4.5 * a.episodes, 6), squeeze=False, sharey="row")
for ep in range(a.episodes):
    obs, _ = env.reset(seed=a.seed + ep)
    ts, M, rb, rr, events = [0], [env.marker()], [0.0], [0.0], []
    while env.agents:
        prev = dict(env.stats)
        obs, r, _, _, info = env.step({**blue.act({k: o for k, o in obs.items() if k.startswith("blue")}, env),
                                       **red.act({k: o for k, o in obs.items() if k.startswith("red")}, env)})
        ts.append(env.t)
        M.append(env.marker())
        rb.append(env.ret["blue"])
        rr.append(env.ret["red"])
        for k, v in env.stats.items():
            if v > prev[k]:
                events.append((env.t, k))
    ax = axs[0, ep]
    ax.plot(ts, M, color="tab:blue", label="M_azul")
    ax.plot(ts, [-m for m in M], color="tab:red", label="M_rojo = -M_azul")
    for t, k in events:
        ax.axvline(t, color="gray", lw=.6, ls=":")
        ax.text(t, max(M + [-m for m in M]) * 1.02 + 0.01, k.replace("_", " "), fontsize=6,
                rotation=90, va="bottom")
    ax.set_title(f"ep {ep}: ganador={info['blue_0']['winner']}")
    ax.axhline(0, color="k", lw=.5)
    ax = axs[1, ep]
    ax.step(ts, rb, where="post", color="tab:blue", label="retorno azul")
    ax.step(ts, rr, where="post", color="tab:red", label="retorno rojo")
    ax.axhline(0, color="k", lw=.5)
    ax.set_xlabel("paso")
axs[0, 0].legend(fontsize=7)
axs[1, 0].legend(fontsize=7)
axs[0, 0].set_ylabel("marcador")
axs[1, 0].set_ylabel("retorno acumulado")
fig.suptitle(f"azul={nb}  vs  rojo={nr}   (beta={a.beta})")
fig.tight_layout()
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=120)
print("guardado", a.out)
