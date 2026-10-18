"""Enfrenta dos políticas (mitad de las partidas de cada lado).
docker compose run --rm ctf python scripts/evaluate.py --a sp_2v2 --b random
docker compose run --rm ctf python scripts/evaluate.py --a sp_2v2:400 --b sp_2v2:50
Política: 'random', 'RUN' (último checkpoint) o 'RUN:EPOCH'."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from _load import make_policy  # noqa: E402
from ctf.env import CTFParallelEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--a", "--run", dest="a", required=True, help="política evaluada")
ap.add_argument("--b", "--opponent", dest="b", default="random", help="rival")
ap.add_argument("--episodes", type=int, default=200)
ap.add_argument("--stochastic", action="store_true")
ap.add_argument("--seed", type=int, default=12345)
a = ap.parse_args()

pa, na, cfg = make_policy(a.a, a.seed, a.stochastic)
pb, nb, cfg_b = make_policy(a.b, a.seed + 1, a.stochastic)
env = CTFParallelEnv(**(cfg or cfg_b)["env"])
res, lens, rets, zs, sides = [], [], [], [], []
for k in range(a.episodes):
    side = "blue" if k % 2 == 0 else "red"  # A juega la mitad de azul y la mitad de rojo
    obs, _ = env.reset(seed=a.seed + k)
    ret = {"blue": 0.0, "red": 0.0}
    while env.agents:
        acts = {**pa.act({x: o for x, o in obs.items() if x.startswith(side)}, env),
                **pb.act({x: o for x, o in obs.items() if not x.startswith(side)}, env)}
        obs, r, _, _, info = env.step(acts)
        ret["blue"] += r["blue_0"]
        ret["red"] += r["red_0"]
    w = info["blue_0"]["winner"]
    res.append("win" if w == side else "draw" if w is None else "loss")
    sides.append(side)
    lens.append(env.t)
    rets.append(ret[side])
    zs.append(abs(ret["blue"] + ret["red"]))
n = len(res)
print(f"A={na} vs B={nb} ({n} partidas, mitad por lado, {'estocástico' if a.stochastic else 'greedy'})")
print(f"A: win={res.count('win') / n:.2%} draw={res.count('draw') / n:.2%} loss={res.count('loss') / n:.2%} "
      f"len={np.mean(lens):.1f} return_A={np.mean(rets):.3f} |ret_A+ret_B|={np.mean(zs):.1e}")
for sd in ("blue", "red"):
    rr = [x for x, s2 in zip(res, sides) if s2 == sd]
    print(f"   A como {sd:4s}: win={rr.count('win') / len(rr):.0%} draw={rr.count('draw') / len(rr):.0%} "
          f"loss={rr.count('loss') / len(rr):.0%}")
