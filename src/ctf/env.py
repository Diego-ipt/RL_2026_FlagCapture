"""Capture the Flag en grilla 2D como PettingZoo ParallelEnv.

Reglas (v3):
- Dos equipos (blue/red), cada uno con base y bandera en su lado. Mapa con cajas
  y murallas simétricas (ver ctf.maps), nuevo en cada reset.
- Acciones: 0 quieto, 1 norte, 2 sur, 3 este, 4 oeste, 5 quitar, 6 pasar.
- Los agentes ocupan celdas: nadie entra a una celda ocupada, salvo las bases y
  las celdas con una bandera en el suelo, que se pueden compartir.
- Camino garantizado: para cada equipo siempre existe un camino de movimientos
  válidos desde la bandera rival hasta su base. Se guarda en caché y solo se
  recalcula (BFS) si un agente pisa ese camino; el movimiento es válido si queda
  otro camino, si no se rechaza. Si tras un evento de bandera no hay camino, los
  agentes que lo tapan quedan obligados a moverse (máscara de acciones).
- Orden por paso: mover -> anotar -> quitar -> pasar -> contactos con banderas.
- Anotar: llevar la bandera rival a tu base (no hace falta tener la propia en casa).
- Quitar: solo a un portador rival adyacente que esté en TU mitad del mapa; la
  bandera cae al suelo y el portador no puede recoger durante `drop_cooldown` pasos.
- Bandera en el suelo: si la toca su equipo vuelve a su base; si la toca un rival, la recoge.
- Pasar: el portador entrega la bandera a un aliado adyacente.
- Recompensa: cambio de un marcador de posición de banderas (distancias de camino
  reales, antisimétrico, se anula al terminar) + ±r_win al anotar. Retorno de la
  ronda = ±r_win o 0.
- Vista egocéntrica: el rojo ve el mapa espejado y sus acciones E/O se intercambian.
- infos[a]["action_mask"]: acciones válidas (en la vista egocéntrica del agente).
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv

from ctf.maps import generate_map
from ctf.paths import bfs_dist, bfs_path

MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, 1], [0, -1]])  # (fila, col)
STAY, STRIP, PASS = 0, 5, 6
N_ACTIONS = 7
TEAMS = ("blue", "red")
EGO_ACTION = {3: 4, 4: 3}  # el rojo ve el mapa espejado: este <-> oeste


class CTFParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "ctf_v3"}

    def __init__(self, size=10, n_per_team=2, view_radius=3, max_steps=200,
                 n_boxes=0, n_walls=0, fixed_map=False, r_win=1.0, w_def=1.0, w_atk=1.0,
                 shaping_scale=0.5, shaping_decay="none", drop_cooldown=2,
                 render_mode=None):
        self.size, self.n, self.r = size, n_per_team, view_radius
        self.max_steps = max_steps
        self.n_boxes, self.n_walls, self.fixed_map = n_boxes, n_walls, fixed_map
        self.r_win, self.w_def, self.w_atk = r_win, w_def, w_atk
        self.shaping_scale, self.shaping_decay = shaping_scale, shaping_decay
        self.drop_cooldown = drop_cooldown
        self.render_mode = render_mode
        self.possible_agents = [f"{t}_{i}" for t in TEAMS for i in range(n_per_team)]
        self.agents = []
        self._screen = None
        self._map = None
        mid = size // 2
        # base/bandera de azul a la izquierda, rojo a la derecha
        self.base = {"blue": np.array([mid, 0]), "red": np.array([mid, size - 1])}

    def set_shaping_scale(self, beta):
        """β del marcador; el trainer lo baja a lo largo del entrenamiento."""
        self.shaping_scale = float(beta)

    # ---- espacios ----
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        w = 2 * self.r + 1
        return Box(0.0, 1.0, (7 * w * w + self.n + 16,), np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(N_ACTIONS)

    @staticmethod
    def team(agent):
        return agent.split("_")[0]

    @staticmethod
    def enemy(team):
        return "red" if team == "blue" else "blue"

    # ---- estado de banderas ----
    # holder[tm]: agente rival que lleva la bandera de tm (o None)
    # flag[tm]: posición de la bandera de tm (en base, en el suelo o con el portador)
    def flag_status(self, tm):
        if self.holder[tm] is not None:
            return 1  # llevada
        return 0 if (self.flag[tm] == self.base[tm]).all() else 2  # base / suelo

    def carrying(self, a):
        return self.holder[self.enemy(self.team(a))] == a

    def in_half(self, tm, cell):
        """True si la celda está en la mitad del equipo tm."""
        return cell[1] < self.size / 2 if tm == "blue" else cell[1] > (self.size - 1) / 2

    # ---- ocupación y caminos ----
    def _shareable(self, cell):
        """Bases y celdas con bandera en el suelo admiten varios agentes y no bloquean."""
        cell = tuple(cell)
        if any(cell == tuple(self.base[tm]) for tm in TEAMS):
            return True
        return any(self.flag_status(tm) == 2 and cell == tuple(self.flag[tm]) for tm in TEAMS)

    def _blocking(self, exclude=None, extra=None):
        """Celdas bloqueadas por agentes (sin contar celdas compartibles)."""
        cells = {tuple(self.pos[b]) for b in self.agents if b != exclude}
        if extra is not None:
            cells.add(tuple(extra))
        return frozenset(c for c in cells if not self._shareable(c))

    def _bfs(self, start, goal, blocked):
        self.stats["bfs_calls"] += 1
        return bfs_path(self.walls, start, goal, blocked)

    def _endpoints(self, e):
        """Camino del equipo e: desde la bandera rival hasta la base de e."""
        return tuple(self.flag[self.enemy(e)]), tuple(self.base[e])

    def _refresh_paths(self):
        """Recalcula los caminos (tras eventos de bandera o si falta alguno) y marca
        a los agentes obligados a moverse si un camino está tapado."""
        self.forced = set()
        self.virtual = {}
        blocked = self._blocking()
        for e in TEAMS:
            start, goal = self._endpoints(e)
            if self.path[e] is not None and self.path[e][0] == start and self.path[e][-1] == goal \
                    and not (set(self.path[e][1:-1]) & blocked):
                continue  # el camino guardado sigue libre
            self.path[e] = self._bfs(start, goal, blocked)
            if self.path[e] is None:
                v = bfs_path(self.walls, start, goal)  # solo paredes: existe por conectividad
                self.virtual[e] = set(v[1:-1])
                self.forced |= {b for b in self.agents if tuple(self.pos[b]) in self.virtual[e]}

    def _check_move(self, a, q):
        """¿Puede `a` moverse a q? Devuelve (válido, caminos actualizados)."""
        q = tuple(int(x) for x in q)
        if not (0 <= q[0] < self.size and 0 <= q[1] < self.size) or self.walls[q]:
            return False, None
        if not self._shareable(q) and any(tuple(self.pos[b]) == q for b in self.agents if b != a):
            return False, None
        if a not in self.forced and any(q in v for e, v in self.virtual.items() if self.path[e] is None):
            return False, None  # nadie más puede entrar a un camino que está siendo despejado
        blocked = None
        updates = {}
        for e in TEAMS:
            path = self.path[e]
            if path is None:
                continue
            if self.carrying(a) and self.team(a) == e:
                # el portador mueve el inicio de su camino a una celda adyacente
                updates[e] = path[path.index(q):] if q in path else [q] + path
            elif q in path[1:-1] and not self._shareable(q):
                if blocked is None:
                    blocked = self._blocking(exclude=a, extra=q)
                start, goal = self._endpoints(e)
                alt = self._bfs(start, goal, blocked)
                if alt is None:
                    return False, None  # cerraría el último camino de e
                updates[e] = alt
        return True, updates

    def action_mask(self, a):
        """Acciones válidas de `a` en su vista egocéntrica (bool[7])."""
        forced = a in self.forced
        tm = self.team(a)
        m = np.zeros(N_ACTIONS, bool)
        m[STAY] = not forced
        for k in range(1, 5):
            m[k] = self._check_move(a, self.pos[a] + MOVES[k])[0]
        c = self.holder[tm]  # rival que lleva mi bandera
        m[STRIP] = (not forced and c is not None and np.abs(self.pos[a] - self.pos[c]).sum() <= 1
                    and self.in_half(tm, self.pos[c]))
        m[PASS] = (not forced and self.carrying(a) and any(
            b != a and self.team(b) == tm and np.abs(self.pos[a] - self.pos[b]).sum() <= 1 for b in self.agents))
        if not m.any():
            m[STAY] = True  # bloqueado sin salida: espera
        if tm == "red":
            m[3], m[4] = m[4], m[3]
        return m

    # ---- dinámica ----
    def reset(self, seed=None, options=None):
        if seed is not None or not hasattr(self, "np_random"):
            from gymnasium.utils import seeding
            self.np_random, _ = seeding.np_random(seed)
        self.agents = self.possible_agents[:]
        self.t = 0
        if self._map is None or not self.fixed_map:
            self._map = generate_map(self.size, self.np_random, self.n_boxes, self.n_walls,
                                     bases=list(self.base.values()))
        self.walls = self._map.copy()
        self._walls_pad = np.pad(self.walls.astype(np.float32), self.r, constant_values=1.0)
        self.dist = {tm: bfs_dist(self.walls, tuple(self.base[tm])) for tm in TEAMS}
        self.D = max(1, int(max(d.max() for d in self.dist.values())))
        self.stats = {f"{k}_{tm}": 0 for k in ("strips", "passes", "returns", "pickups") for tm in TEAMS}
        self.stats.update(bfs_calls=0, rejected_moves=0, forced_moves=0)
        self.cooldown = {a: 0 for a in self.agents}
        self.flag = {tm: self.base[tm].copy() for tm in TEAMS}
        self.holder = {tm: None for tm in TEAMS}
        self.pos = {}
        for tm in TEAMS:  # spawn en la columna de la base, filas distintas
            col = 0 if tm == "blue" else self.size - 1
            rows = self.np_random.permutation(self.size)[:self.n]
            for i, r in enumerate(rows):
                self.pos[f"{tm}_{i}"] = np.array([int(r), col])
        self.path = {tm: None for tm in TEAMS}
        self.forced, self.virtual = set(), {}
        self._refresh_paths()
        self._M = self._marker()
        self.ret = {tm: 0.0 for tm in TEAMS}  # retorno acumulado de la ronda (HUD/análisis)
        return self._obs_all(), {a: {"action_mask": self.action_mask(a)} for a in self.agents}

    def step(self, actions):
        self.t += 1
        acts = {}
        for a in self.agents:
            act = int(actions.get(a, STAY))
            acts[a] = EGO_ACTION.get(act, act) if self.team(a) == "red" else act
        for a in self.agents:
            self.cooldown[a] = max(0, self.cooldown[a] - 1)
        # obligados a moverse: si no eligieron un movimiento válido, se elige uno al azar
        for a in self.forced:
            valid = [k for k in range(1, 5) if self._check_move(a, self.pos[a] + MOVES[k])[0]]
            if acts[a] not in valid and valid:
                acts[a] = int(self.np_random.choice(valid))
                self.stats["forced_moves"] += 1
        self._move(acts)

        # anotar se resuelve antes que quitar
        scorers = [tm for tm in TEAMS if self.holder[self.enemy(tm)] is not None
                   and (self.pos[self.holder[self.enemy(tm)]] == self.base[tm]).all()]
        winner = scorers[0] if len(scorers) == 1 else None
        done = len(scorers) > 0  # si ambos anotan en el mismo paso: empate
        if not done:
            before = (dict(self.holder), {tm: tuple(self.flag[tm]) for tm in TEAMS})
            dropped = self._strip(acts)
            self._pass(acts)
            self._flag_contacts(skip=dropped)
            after = (dict(self.holder), {tm: tuple(self.flag[tm]) for tm in TEAMS})
            if before != after or self.forced or any(p is None for p in self.path.values()):
                self._refresh_paths()
        trunc = self.t >= self.max_steps and not done

        # marcador: se paga su cambio; al terminar la ronda vuelve a 0
        M_new = 0.0 if (done or trunc) else self._marker()
        r_blue = M_new - self._M
        self._M = M_new
        if winner:
            r_blue += self.r_win if winner == "blue" else -self.r_win
        r_team = {"blue": r_blue, "red": -r_blue}
        for tm in TEAMS:
            self.ret[tm] += r_team[tm]

        obs = self._obs_all()
        rew = {a: r_team[self.team(a)] for a in self.agents}
        term = {a: done for a in self.agents}
        tr = {a: trunc for a in self.agents}
        infos = {a: {"winner": winner, "action_mask": self.action_mask(a), **self.stats} for a in self.agents}
        if done or trunc:
            self.agents = []
        return obs, rew, term, tr, infos

    def _move(self, acts):
        order = list(self.agents)
        self.np_random.shuffle(order)  # orden aleatorio por paso (evita sesgo)
        for a in order:
            act = acts[a]
            if not 1 <= act <= 4:
                continue
            ok, updates = self._check_move(a, self.pos[a] + MOVES[act])
            if not ok:
                self.stats["rejected_moves"] += 1
                continue
            self.pos[a] = self.pos[a] + MOVES[act]
            self.path.update(updates)
            for tm in TEAMS:  # el portador arrastra la bandera
                if self.holder[tm] == a:
                    self.flag[tm] = self.pos[a].copy()
        # los obligados que ya salieron del camino tapado dejan de estarlo
        self.forced = {b for b in self.forced if any(tuple(self.pos[b]) in v for v in self.virtual.values())}

    def _strip(self, acts):
        """Quitar: tira al suelo la bandera de mi equipo si su portador está adyacente
        y dentro de MI mitad del mapa."""
        dropped = set()
        for a in self.agents:
            if acts[a] != STRIP:
                continue
            tm = self.team(a)
            c = self.holder[tm]  # rival que lleva MI bandera
            if c is not None and np.abs(self.pos[a] - self.pos[c]).sum() <= 1 and self.in_half(tm, self.pos[c]):
                self.holder[tm] = None
                self.flag[tm] = self.pos[c].copy()
                self.cooldown[c] = self.drop_cooldown
                self.stats[f"strips_{tm}"] += 1
                dropped.add(tm)
        return dropped

    def _pass(self, acts):
        """Pasar: el portador entrega la bandera al aliado adyacente más cercano a la base propia."""
        passed = set()
        for a in sorted(self.agents):
            tm, en = self.team(a), self.enemy(self.team(a))
            if acts[a] != PASS or self.holder[en] != a or en in passed:
                continue
            mates = [b for b in sorted(self.agents) if b != a and self.team(b) == tm
                     and np.abs(self.pos[a] - self.pos[b]).sum() <= 1]
            if not mates:
                continue
            b = min(mates, key=lambda m: self.dist[tm][tuple(self.pos[m])])
            self.holder[en] = b
            self.flag[en] = self.pos[b].copy()
            self.stats[f"passes_{tm}"] += 1
            passed.add(en)

    def _flag_contacts(self, skip=()):
        """Banderas no llevadas: el dueño la devuelve a la base, un rival la recoge."""
        for tm in TEAMS:
            if self.holder[tm] is not None or tm in skip:
                continue
            here = [a for a in sorted(self.agents) if (self.pos[a] == self.flag[tm]).all()]
            owners = [a for a in here if self.team(a) == tm]
            rivals = [a for a in here if self.team(a) != tm and self.cooldown[a] == 0
                      and not self.carrying(a)]
            on_ground = not (self.flag[tm] == self.base[tm]).all()
            if on_ground and owners:
                self.flag[tm] = self.base[tm].copy()  # vuelve a la base (niega el avance)
                self.stats[f"returns_{tm}"] += 1
            elif rivals:
                self.holder[tm] = rivals[0]
                self.flag[tm] = self.pos[rivals[0]].copy()
                self.stats[f"pickups_{self.enemy(tm)}"] += 1

    # ---- recompensa ----
    def _d(self, cell, tm):
        """Distancia de camino real (con paredes) de una celda a la base de tm."""
        d = self.dist[tm][tuple(cell)]
        return self.D if d < 0 else d

    def _score(self, tm):
        """S_e: mi bandera cerca de mi base + bandera enemiga cerca de mi base."""
        en = self.enemy(tm)
        own = 1 - self._d(self.flag[tm], tm) / self.D
        atk = 1 - self._d(self.flag[en], tm) / self.D
        return self.w_def * own + self.w_atk * atk

    def _beta(self):
        if self.shaping_decay == "episode":
            return self.shaping_scale * max(0.0, 1 - self.t / self.max_steps)
        return self.shaping_scale

    def _marker(self):
        """Marcador M_azul = β·(S_azul − S_rojo); M_rojo = −M_azul."""
        return float(self._beta() * (self._score("blue") - self._score("red")))

    def marker(self):
        """Marcador actual (azul); el del rojo es el negativo."""
        return self._M

    # ---- observaciones ----
    def _obs_all(self):
        return {a: self._obs(a) for a in self.agents}

    def _ego(self, viewer, p):
        """Posición vista por el equipo `viewer` (el rojo ve las columnas espejadas)."""
        p = np.asarray(p, np.float32)
        return p if viewer == "blue" else np.array([p[0], self.size - 1 - p[1]], np.float32)

    def _flag_vec(self, tm, viewer="blue"):
        st = np.zeros(3, np.float32)
        st[self.flag_status(tm)] = 1.0
        return st, self._ego(viewer, self.flag[tm]) / (self.size - 1)

    def _obs(self, a):
        w, r = 2 * self.r + 1, self.r
        tm, en = self.team(a), self.enemy(self.team(a))
        # canales: paredes, aliados, rivales, bandera propia, bandera rival,
        #          aliado portador, rival portador
        grid = np.zeros((7, w, w), np.float32)
        p = self.pos[a]

        def put(ch, q):
            d = q - p + r
            if 0 <= d[0] < w and 0 <= d[1] < w:
                grid[ch, d[0], d[1]] = 1.0

        grid[0] = self._walls_pad[p[0]:p[0] + w, p[1]:p[1] + w]  # fuera del mapa = pared
        for b in self.agents:
            if b != a:
                ally = self.team(b) == tm
                put(1 if ally else 2, self.pos[b])
                if self.carrying(b):
                    put(5 if ally else 6, self.pos[b])
        put(3, self.flag[tm])
        put(4, self.flag[en])
        if tm == "red":
            grid = grid[:, :, ::-1]  # vista espejada: la base propia queda a la izquierda
        idv = np.zeros(self.n, np.float32)
        idv[int(a.split("_")[1])] = 1.0
        own_st, own_pos = self._flag_vec(tm, tm)
        en_st, en_pos = self._flag_vec(en, tm)
        ep = self._ego(tm, p) / (self.size - 1)
        extra = np.array([self.carrying(a), self.t / self.max_steps, ep[0], ep[1],
                          self.cooldown[a] / max(1, self.drop_cooldown), a in self.forced], np.float32)
        # 6 + 3+2 + 3+2 = 16 (+ n del ID)
        return np.concatenate([grid.ravel(), idv, extra, own_st, own_pos, en_st, en_pos]).astype(np.float32)

    def state(self, team="blue"):
        """Estado global para el crítico centralizado, desde la perspectiva de `team`.
        Canales: mi equipo, rival, mi bandera, bandera rival, portadores (+ paredes)."""
        en = self.enemy(team)
        s = np.zeros((5, self.size, self.size), np.float32)
        for a in self.possible_agents:
            if a in self.pos:
                s[0 if self.team(a) == team else 1, self.pos[a][0], self.pos[a][1]] = 1
                if self.carrying(a):
                    s[4, self.pos[a][0], self.pos[a][1]] = 1
        s[2, self.flag[team][0], self.flag[team][1]] = 1
        s[3, self.flag[en][0], self.flag[en][1]] = 1
        walls = self.walls.astype(np.float32)
        if team == "red":
            s, walls = s[:, :, ::-1], walls[:, ::-1]
        status = np.concatenate([self._flag_vec(team)[0], self._flag_vec(en)[0]])
        return np.concatenate([s.ravel(), walls.ravel(), status,
                               [self.t / self.max_steps]]).astype(np.float32)

    # ---- render ----
    def render(self):
        if self.render_mode is None:
            return None
        import pygame
        cs, hud = 40, 56
        W, H = self.size * cs, self.size * cs + hud
        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((W, H)) if self.render_mode == "human" else pygame.Surface((W, H))
            self._font = pygame.font.Font(None, 20)
        screen = self._screen
        screen.fill((255, 255, 255))
        col = {"blue": (40, 90, 220), "red": (220, 60, 60)}
        # HUD: marcador M_azul = -M_rojo (barra centrada) y retornos acumulados de la ronda
        M, lim = self._M, max(1e-6, self.shaping_scale * (self.w_def + self.w_atk))
        pygame.draw.rect(screen, (225, 225, 225), (10, 30, W - 20, 14))
        x0, half = W // 2, (W - 20) // 2
        wbar = int(half * min(1.0, abs(M) / lim))
        if M > 0:
            pygame.draw.rect(screen, col["blue"], (x0, 30, wbar, 14))
        elif M < 0:
            pygame.draw.rect(screen, col["red"], (x0 - wbar, 30, wbar, 14))
        pygame.draw.line(screen, (0, 0, 0), (x0, 26), (x0, 47), 2)
        txt = (f"t={self.t}  M az {M:+.2f} / ro {-M:+.2f}   "
               f"ret az {self.ret['blue']:+.2f} / ro {self.ret['red']:+.2f}")
        screen.blit(self._font.render(txt, True, (0, 0, 0)), (10, 8))
        s = screen.subsurface((0, hud, W, H - hud))
        s.fill((240, 240, 240))
        # mitades (tono suave) y línea central
        pygame.draw.rect(s, (232, 236, 248), (0, 0, W // 2, H - hud))
        pygame.draw.rect(s, (248, 234, 234), (W // 2, 0, W - W // 2, H - hud))
        for tm in TEAMS:
            b = self.base[tm]
            pygame.draw.rect(s, tuple(int(c * .4 + 150) for c in col[tm]), (b[1] * cs, b[0] * cs, cs, cs))
        for i, j in zip(*np.nonzero(self.walls)):
            pygame.draw.rect(s, (60, 60, 60), (j * cs, i * cs, cs, cs))
        for tm in TEAMS:
            if self.holder[tm] is not None:
                continue  # la bandera llevada se dibuja como anillo en el portador
            f = self.flag[tm]
            tri = [(f[1] * cs + 8, f[0] * cs + 6), (f[1] * cs + 8, f[0] * cs + 30), (f[1] * cs + 30, f[0] * cs + 18)]
            pygame.draw.polygon(s, col[tm], tri)
            if self.flag_status(tm) == 2:  # en el suelo: borde negro
                pygame.draw.polygon(s, (0, 0, 0), tri, 2)
        for a in self.agents:
            p = self.pos[a]
            c = (p[1] * cs + cs // 2, p[0] * cs + cs // 2)
            pygame.draw.circle(s, col[self.team(a)], c, cs // 3)
            if self.carrying(a):
                pygame.draw.circle(s, col[self.enemy(self.team(a))], c, cs // 3 + 4, 4)
            if self.cooldown[a] > 0:
                pygame.draw.circle(s, (255, 255, 255), c, cs // 8)
            if a in self.forced:  # obligado a moverse: borde amarillo
                pygame.draw.circle(s, (250, 200, 0), c, cs // 3, 3)
        if self.render_mode == "human":
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
