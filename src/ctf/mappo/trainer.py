"""MAPPO por equipo: actor compartido dentro del equipo + crítico centralizado.

Modos (`opponent` en la config):
- "random": solo aprende el azul (un MAPPO) contra un rojo aleatorio.
- "selfplay": DOS MAPPO distintos, uno azul y uno rojo (pesos, crítico y
  optimizador propios). En cada entorno juegan ambos aprendices actuales y los
  dos aprenden; con prob. `p_past` un aprendiz enfrenta en cambio a una versión
  congelada del otro equipo, elegida por PFSP de la liga de ese equipo.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from ctf.checkpoint import RunStore
from ctf.env import TEAMS, CTFParallelEnv
from ctf.mappo.buffer import RolloutBuffer, compute_gae
from ctf.mappo.networks import Actor, Critic, ValueNorm
from ctf.mappo.opponents import ActorPolicy, RandomPolicy

DEFAULTS = dict(
    run_name="mappo", seed=0, iterations=300, save_every=25,
    n_envs=16, T=128, gamma=0.99, lam=0.95, lr=5e-4, ppo_epochs=5, minibatches=4,
    clip=0.2, vf_coef=1.0, ent_coef=0.01, max_grad_norm=10.0, hidden=128,
    # marcador: beta0 inicial; decay "training" lo baja a 0 en anneal_epochs,
    # "episode" lo baja dentro de cada ronda, "none" lo deja fijo
    beta0=0.5, anneal_epochs=200, decay="training",
    # self-play con dos redes y ligas por equipo
    opponent="random", p_past=0.3, snapshot_every=10, pool_size=20, pfsp_power=2.0,
    env=dict(size=8, n_per_team=1, view_radius=3, max_steps=100, n_boxes=0, n_walls=0,
             fixed_map=False, r_win=1.0, w_def=1.0, w_atk=1.0, drop_cooldown=2),
)


def full_config(cfg):
    return {**DEFAULTS, **cfg, "env": {**DEFAULTS["env"], **cfg.get("env", {})}}


def build_actor(cfg):
    cfg = full_config(cfg)
    env = CTFParallelEnv(**cfg["env"])
    return Actor(env.observation_space("blue_0").shape[0], env.action_space("blue_0").n, cfg["hidden"])


class League:
    """Versiones congeladas de un equipo, con el win rate (EMA) del rival contra cada una."""

    def __init__(self, rng, size, power):
        self.rng, self.size, self.power = rng, size, power
        self.members = []  # {"name", "policy", "wr"}

    def add(self, name, policy):
        self.members.append({"name": name, "policy": policy, "wr": 0.5})
        while len(self.members) > self.size:  # se conserva siempre el primero (random)
            self.members.pop(1)

    def sample(self):
        w = np.array([(1 - m["wr"]) ** self.power + 0.05 for m in self.members])
        return self.members[self.rng.choice(len(self.members), p=w / w.sum())]

    def record(self, name, score):
        for m in self.members:
            if m["name"] == name:
                m["wr"] = 0.95 * m["wr"] + 0.05 * score


class TeamLearner:
    """Un MAPPO completo para un equipo."""

    def __init__(self, team, obs_dim, state_dim, n_actions, cfg):
        self.team, self.cfg = team, cfg
        self.actor = Actor(obs_dim, n_actions, cfg["hidden"])
        self.critic = Critic(state_dim, cfg["hidden"])
        self.params = [*self.actor.parameters(), *self.critic.parameters()]
        self.opt = torch.optim.Adam(self.params, lr=cfg["lr"], eps=1e-5)
        self.vnorm = ValueNorm()

    def models(self):
        t = self.team
        return {f"actor_{t}": self.actor, f"critic_{t}": self.critic, f"vnorm_{t}": self.vnorm}

    def optims(self):
        return {f"opt_{self.team}": self.opt}

    @torch.no_grad()
    def value(self, state):
        return self.vnorm.denormalize(self.critic(torch.as_tensor(state)).numpy())

    def update(self, buf, adv, ret):
        c = self.cfg
        valid = buf.valid.reshape(-1)
        if valid.sum() < 2:
            return {}
        sel = np.flatnonzero(valid)

        def flat(x):
            return torch.as_tensor(x.reshape(-1, *x.shape[3:])[sel])

        obs, state, amask = flat(buf.obs), flat(buf.state), flat(buf.amask)
        act, old_logp = flat(buf.actions), flat(buf.logp)
        old_v = torch.as_tensor(self.vnorm.normalize(buf.values.reshape(-1)[sel]), dtype=torch.float32)
        ret = ret.reshape(-1)[sel]
        self.vnorm.update(ret)
        ret_n = torch.as_tensor(self.vnorm.normalize(ret), dtype=torch.float32)
        adv = torch.as_tensor(adv.reshape(-1)[sel], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.shape[0]
        mb = max(1, N // c["minibatches"])
        stats = {k: [] for k in ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac")}
        for _ in range(c["ppo_epochs"]):
            perm = torch.randperm(N)
            for s in range(0, N, mb):
                idx = perm[s:s + mb]
                d = self.actor.dist(obs[idx], amask[idx])
                logp = d.log_prob(act[idx])
                ratio = torch.exp(logp - old_logp[idx])
                a = adv[idx]
                pl = -torch.min(ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a).mean()
                v = self.critic(state[idx])
                v_clip = old_v[idx] + (v - old_v[idx]).clamp(-c["clip"], c["clip"])
                vl = torch.max((v - ret_n[idx]) ** 2, (v_clip - ret_n[idx]) ** 2).mean()
                ent = d.entropy().mean()
                loss = pl + c["vf_coef"] * vl - c["ent_coef"] * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, c["max_grad_norm"])
                self.opt.step()
                with torch.no_grad():
                    stats["actor_loss"].append(pl.item())
                    stats["critic_loss"].append(vl.item())
                    stats["entropy"].append(ent.item())
                    stats["approx_kl"].append((old_logp[idx] - logp).mean().item())
                    stats["clip_frac"].append(((ratio - 1).abs() > c["clip"]).float().mean().item())
        return {f"{k}_{self.team}": float(np.mean(v)) for k, v in stats.items()}


class MAPPOTrainer:
    def __init__(self, cfg: dict, root="runs"):
        self.cfg = cfg = full_config(cfg)
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        self.rng = np.random.default_rng(cfg["seed"])
        env_kw = {**cfg["env"], "shaping_scale": cfg["beta0"],
                  "shaping_decay": "episode" if cfg["decay"] == "episode" else "none"}
        self.envs = [CTFParallelEnv(**env_kw) for _ in range(cfg["n_envs"])]
        e = self.envs[0]
        self.n = cfg["env"]["n_per_team"]
        self.obs_dim = e.observation_space("blue_0").shape[0]
        e.reset(seed=0)
        self.state_dim = e.state().shape[0] + self.n  # estado global + ID del agente
        self.n_actions = e.action_space("blue_0").n
        self.selfplay = cfg["opponent"] == "selfplay"
        teams = TEAMS if self.selfplay else ("blue",)
        self.learners = {t: TeamLearner(t, self.obs_dim, self.state_dim, self.n_actions, cfg) for t in teams}
        self.random_opp = RandomPolicy(cfg["seed"] + 1, self.n_actions)
        # liga[t] = versiones congeladas del equipo t (rivales del otro aprendiz)
        self.league = {t: League(self.rng, cfg["pool_size"], cfg["pfsp_power"]) for t in TEAMS}
        for t in TEAMS:
            self.league[t].add("random", self.random_opp)
        self.store = RunStore(cfg["run_name"], root)
        self.store.save_config(cfg)
        self.epoch, self.steps = 0, 0
        self._ids = np.eye(self.n, dtype=np.float32)
        E = cfg["n_envs"]
        self._ep_len = np.zeros(E, int)
        self.ctrl = [None] * E  # por entorno: {equipo: ("learner"|nombre, política o None)}
        self._obs, self._mask = [], []
        for i, env in enumerate(self.envs):
            o, inf = env.reset(seed=cfg["seed"] * 1000 + i)
            self._obs.append(o)
            self._mask.append({a: inf[a]["action_mask"] for a in o})
            self._assign(i)

    # ---- utilidades ----
    def _models(self):
        out = {}
        for L in self.learners.values():
            out.update(L.models())
        return out

    def _optims(self):
        out = {}
        for L in self.learners.values():
            out.update(L.optims())
        return out

    def resume(self, which="latest"):
        self.epoch, self.steps = self.store.restore(self._models(), self._optims(), which)
        if self.selfplay:  # reconstruye las ligas con los checkpoints guardados
            for ck in self.store.list():
                if ck["epoch"] > self.epoch:
                    continue
                blob = self.store.load(ck["epoch"])
                for t in TEAMS:
                    a = build_actor(self.cfg)
                    a.load_state_dict(blob["models"][f"actor_{t}"])
                    self.league[t].add(f"ep{ck['epoch']}", ActorPolicy(a, stochastic=True))

    def _agents(self, team):
        return [f"{team}_{k}" for k in range(self.n)]

    def _assign(self, i):
        """Elige quién controla cada equipo en el entorno i para el próximo episodio."""
        if not self.selfplay:
            self.ctrl[i] = {"blue": ("learner", None), "red": ("random", self.random_opp)}
        elif self.rng.random() < self.cfg["p_past"]:
            t = TEAMS[int(self.rng.integers(2))]  # equipo que aprende en este episodio
            other = "red" if t == "blue" else "blue"
            m = self.league[other].sample()
            self.ctrl[i] = {t: ("learner", None), other: (m["name"], m["policy"])}
        else:
            self.ctrl[i] = {t: ("learner", None) for t in TEAMS}

    def _snapshot(self):
        for t, L in self.learners.items():
            a = copy.deepcopy(L.actor)
            a.eval()
            self.league[t].add(f"ep{self.epoch}", ActorPolicy(a, stochastic=True))

    def _state(self, env, team):
        s = env.state(team)
        return np.stack([np.concatenate([s, self._ids[k]]) for k in range(self.n)])

    # ---- rollout ----
    @torch.no_grad()
    def collect(self):
        c = self.cfg
        E = len(self.envs)
        bufs = {t: RolloutBuffer(c["T"], E, self.n, self.obs_dim, self.state_dim, self.n_actions)
                for t in self.learners}
        episodes = []
        for _ in range(c["T"]):
            step_data = {}
            actions = [dict() for _ in range(E)]
            for t, L in self.learners.items():
                ag = self._agents(t)
                obs = np.stack([np.stack([self._obs[i][a] for a in ag]) for i in range(E)])
                amask = np.stack([np.stack([self._mask[i][a] for a in ag]) for i in range(E)])
                state = np.stack([self._state(env, t) for env in self.envs])
                d = L.actor.dist(torch.as_tensor(obs), torch.as_tensor(amask))
                act = d.sample()
                logp = d.log_prob(act).numpy()
                act = act.numpy()
                valid = np.array([[self.ctrl[i][t][0] == "learner"] * self.n for i in range(E)])
                step_data[t] = [obs, state, amask, act, logp, L.value(state), valid]
                for i in range(E):
                    if valid[i, 0]:
                        actions[i].update(zip(ag, act[i].tolist()))
            rew = {t: np.zeros((E, self.n), np.float32) for t in self.learners}
            done = np.zeros((E, self.n), np.float32)
            for i, env in enumerate(self.envs):
                for t, (who, pol) in self.ctrl[i].items():
                    if who != "learner":
                        ag = self._agents(t)
                        actions[i].update(pol.act({a: self._obs[i][a] for a in ag}, env,
                                                  {a: self._mask[i][a] for a in ag}))
                nobs, r, term, trunc, info = env.step(actions[i])
                for t in self.learners:
                    rew[t][i] = [r[a] for a in self._agents(t)]
                self._ep_len[i] += 1
                if term["blue_0"] or trunc["blue_0"]:
                    # el límite de tiempo es parte del juego (la ronda acaba y el
                    # tiempo está en la obs): se trata como terminal, sin bootstrap
                    done[i] = 1.0
                    w = info["blue_0"]["winner"]
                    ctrl = {t: v[0] for t, v in self.ctrl[i].items()}
                    for t, who in ctrl.items():  # liga: resultado del aprendiz vs el congelado
                        other = "red" if t == "blue" else "blue"
                        if who != "learner" and ctrl[other] == "learner":
                            s = 1.0 if w == other else 0.0 if w is not None else 0.5
                            self.league[t].record(who, s)
                    episodes.append(dict(ctrl=ctrl, winner=w, len=self._ep_len[i],
                                         ret_blue=env.ret["blue"], ret_red=env.ret["red"],
                                         info=info["blue_0"]))
                    self._ep_len[i] = 0
                    nobs, info = env.reset()
                    self._assign(i)
                self._obs[i] = nobs
                self._mask[i] = {a: info[a]["action_mask"] for a in nobs}
            for t in self.learners:
                obs, state, amask, act, logp, val, valid = step_data[t]
                bufs[t].add(obs, state, amask, act, logp, val, rew[t], done, valid)
        out = {}
        for t, L in self.learners.items():
            last_v = L.value(np.stack([self._state(env, t) for env in self.envs]))
            b = bufs[t]
            adv, ret = compute_gae(b.rewards, b.values, b.dones, last_v, c["gamma"], c["lam"])
            out[t] = (b, adv, ret)
        self.steps += c["T"] * E
        return out, episodes

    def beta(self):
        c = self.cfg
        if c["decay"] == "training":
            return c["beta0"] * max(0.0, 1 - self.epoch / max(1, c["anneal_epochs"]))
        return c["beta0"]

    @staticmethod
    def _summary(eps):
        def mean(xs):
            xs = list(xs)
            return float(np.mean(xs)) if xs else float("nan")

        live = [e for e in eps if all(v == "learner" for v in e["ctrl"].values())]

        def score(e, t):
            return 1.0 if e["winner"] == t else 0.0 if e["winner"] is not None else 0.5

        row = dict(
            win_blue=mean(e["winner"] == "blue" for e in eps),
            win_red=mean(e["winner"] == "red" for e in eps),
            draw_rate=mean(e["winner"] is None for e in eps),
            # partidas entre los dos aprendices actuales
            live_win_blue=mean(e["winner"] == "blue" for e in live),
            live_win_red=mean(e["winner"] == "red" for e in live),
            # aprendiz vs versión pasada / random del otro equipo
            blue_vs_past=mean(score(e, "blue") for e in eps
                              if e["ctrl"].get("blue") == "learner" and e["ctrl"].get("red") != "learner"),
            red_vs_past=mean(score(e, "red") for e in eps
                             if e["ctrl"].get("red") == "learner" and e["ctrl"].get("blue") != "learner"),
            ep_len=mean(e["len"] for e in eps),
            ret_blue=mean(e["ret_blue"] for e in eps),
            # suma cero: lo que gana un equipo lo pierde el otro (debe ser 0)
            zero_sum_err=mean(abs(e["ret_blue"] + e["ret_red"]) for e in eps),
        )
        for k in ("strips", "passes", "returns", "pickups"):
            for t in TEAMS:
                row[f"{k}_{t}"] = mean(e["info"][f"{k}_{t}"] for e in eps)
        for k in ("rejected_moves", "forced_moves", "bfs_calls"):
            row[k] = mean(e["info"][k] for e in eps)
        return row

    def train(self, iterations=None):
        c = self.cfg
        end = self.epoch + (iterations or c["iterations"])
        while self.epoch < end:
            t0 = time.time()
            beta = self.beta()
            for env in self.envs:
                env.set_shaping_scale(beta)
            data, eps = self.collect()
            t_col = time.time() - t0
            st = {}
            for t, L in self.learners.items():
                st.update(L.update(*data[t]))
            self.epoch += 1
            if self.selfplay and self.epoch % c["snapshot_every"] == 0:
                self._snapshot()
            row = dict(epoch=self.epoch, env_steps=self.steps, episodes=len(eps), beta=beta,
                       **self._summary(eps), **st,
                       sec=round(time.time() - t0, 2), sec_collect=round(t_col, 2))
            self.store.log(**row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
            if self.epoch % c["save_every"] == 0 or self.epoch == end:
                self.store.save(self.epoch, self.steps, self._models(), self._optims(),
                                {"win_blue": row["win_blue"], "win_red": row["win_red"],
                                 "league": {t: {m["name"]: round(m["wr"], 3) for m in L.members}
                                            for t, L in self.league.items()}})
