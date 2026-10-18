"""Buffer de rollout [T, n_envs, n_agents] y GAE."""
import numpy as np


def compute_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.95):
    """rewards/values/dones: [T, ...]; last_values: [...] = V(s_T).
    dones[t]=1 si el episodio terminó en el paso t (no se propaga valor del siguiente)."""
    T = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_values)
    for t in reversed(range(T)):
        next_v = last_values if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


class RolloutBuffer:
    """amask: acciones válidas de cada agente; valid: si el equipo era aprendiz en
    ese entorno y paso (si jugaba una versión congelada, la transición no se usa)."""

    def __init__(self, T, n_envs, n_agents, obs_dim, state_dim, n_actions):
        sh = (T, n_envs, n_agents)
        self.obs = np.zeros(sh + (obs_dim,), np.float32)
        self.state = np.zeros(sh + (state_dim,), np.float32)
        self.amask = np.ones(sh + (n_actions,), bool)
        self.actions = np.zeros(sh, np.int64)
        self.logp = np.zeros(sh, np.float32)
        self.values = np.zeros(sh, np.float32)   # valores desnormalizados
        self.rewards = np.zeros(sh, np.float32)
        self.dones = np.zeros(sh, np.float32)
        self.valid = np.ones(sh, bool)
        self.t = 0

    def add(self, obs, state, amask, actions, logp, values, rewards, dones, valid):
        i = self.t
        self.obs[i], self.state[i], self.amask[i], self.actions[i] = obs, state, amask, actions
        self.logp[i], self.values[i], self.rewards[i], self.dones[i] = logp, values, rewards, dones
        self.valid[i] = valid
        self.t += 1
