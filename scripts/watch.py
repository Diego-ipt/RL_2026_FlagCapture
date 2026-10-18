"""Visualiza partidas rápidamente (con marcador de suma cero en la parte superior).

Docker (sin pantalla): genera GIF   ->  runs/viz/<nombre>.gif
  docker compose run --rm ctf python scripts/watch.py --out runs/viz/random.gif
  docker compose run --rm ctf python scripts/watch.py --blue sp_2v2 --red sp_2v2:100 --out runs/viz/sp.gif
En tu PC (ventana): pip install pygame-ce gymnasium pettingzoo numpy torch; luego
  set PYTHONPATH=src & python scripts/watch.py --human --blue sp_2v2 --red sp_2v2
Política: 'random', 'RUN' (último checkpoint) o 'RUN:EPOCH'.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ap = argparse.ArgumentParser()
ap.add_argument("--human", action="store_true", help="ventana pygame (en el host)")
ap.add_argument("--out", default="runs/viz/random.gif")
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--size", type=int, default=10)
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--boxes", type=int, default=2)
ap.add_argument("--walls", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fps", type=int, default=8)
ap.add_argument("--blue", "--run", dest="blue", default="random", help="política del azul")
ap.add_argument("--red", default="random", help="política del rojo")
ap.add_argument("--beta", type=float, default=0.5, help="escala del marcador mostrado")
ap.add_argument("--stochastic", action="store_true", help="muestrear acciones en vez de argmax")
a = ap.parse_args()

if not a.human:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from _load import make_policy  # noqa: E402
from ctf.env import CTFParallelEnv  # noqa: E402

blue, nb, cfg_b = make_policy(a.blue, a.seed, a.stochastic)
red, nr, cfg_r = make_policy(a.red, a.seed + 1, a.stochastic)
cfg = cfg_b or cfg_r
env_kw = cfg["env"] if cfg else dict(size=a.size, n_per_team=a.n, n_boxes=a.boxes, n_walls=a.walls)
print(f"azul = {nb}   rojo = {nr}")
env = CTFParallelEnv(**{**env_kw, "shaping_scale": a.beta}, render_mode="human" if a.human else "rgb_array")
frames = []
for ep in range(a.episodes):
    obs, _ = env.reset(seed=a.seed + ep)
    frames.append(env.render())
    while env.agents:
        acts = {**blue.act({k: o for k, o in obs.items() if k.startswith("blue")}, env),
                **red.act({k: o for k, o in obs.items() if k.startswith("red")}, env)}
        obs, r, _, _, info = env.step(acts)
        if a.human:
            env.render()
            time.sleep(1 / a.fps)
        else:
            frames.append(env.render())
    frames += [frames[-1]] * a.fps  # pausa al final de cada ronda
    i = info["blue_0"]
    print(f"episodio {ep}: {env.t} pasos, ganador={i['winner']}, retorno azul={env.ret['blue']:+.3f} "
          f"rojo={env.ret['red']:+.3f}, quitadas a/r={i['strips_blue']}/{i['strips_red']}, "
          f"pases a/r={i['passes_blue']}/{i['passes_red']}, devoluciones a/r={i['returns_blue']}/{i['returns_red']}, "
          f"rechazados={i['rejected_moves']}, forzados={i['forced_moves']}")
if not a.human:
    import imageio
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, frames, fps=a.fps, loop=0)
    print("guardado", a.out, f"({len(frames)} frames)")
env.close()
