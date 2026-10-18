import numpy as np
from pettingzoo.test import parallel_api_test

from ctf.env import PASS, STAY, STRIP, TEAMS, CTFParallelEnv
from ctf.maps import generate_map, protected_cells
from ctf.paths import bfs_dist, bfs_path, connected

N, S, E, W = 1, 2, 3, 4  # mover (en la vista del azul)
RED_W = 3   # el rojo ve el mapa espejado: su "este" es el oeste del mundo


def test_api():
    parallel_api_test(CTFParallelEnv(), num_cycles=300)


def test_api_walls_episode_decay():
    parallel_api_test(CTFParallelEnv(n_boxes=2, n_walls=2, shaping_decay="episode"), num_cycles=300)


def make(n=1, size=8, **kw):
    env = CTFParallelEnv(size=size, n_per_team=n, **kw)
    env.reset(seed=0)
    return env


def place(env, walls=None, **pos):
    """Coloca agentes a mano y recalcula caminos/obligaciones."""
    if walls is not None:
        env.walls = walls
        env._walls_pad = np.pad(walls.astype(np.float32), env.r, constant_values=1.0)
        env.dist = {tm: bfs_dist(env.walls, tuple(env.base[tm])) for tm in TEAMS}
    for a, p in pos.items():
        env.pos[a] = np.array(p)
    for tm in TEAMS:
        if env.holder[tm] is not None:
            env.flag[tm] = env.pos[env.holder[tm]].copy()
    env.path = {tm: None for tm in TEAMS}
    env.forced, env.virtual = set(), {}
    env._refresh_paths()


def give(env, carrier, flag_team):
    env.holder[flag_team] = carrier
    env.flag[flag_team] = env.pos[carrier].copy()
    place(env)


def stay(env, **acts):
    return {a: acts.get(a, STAY) for a in env.agents}


def path_exists(env, e):
    start, goal = env._endpoints(e)
    return bfs_path(env.walls, start, goal, env._blocking()) is not None


# ---------- reglas básicas ----------
def test_pickup_at_enemy_base_and_score_without_own_flag_home():
    env = make()
    place(env, walls=np.zeros((8, 8), bool), blue_0=(4, 6), red_0=(0, 2))
    give(env, "red_0", "blue")
    env.step(stay(env, blue_0=E))  # pisa la base roja (4,7) -> recoge
    assert env.holder["red"] == "blue_0"
    place(env, blue_0=(4, 1))
    _, r, term, _, info = env.step(stay(env, blue_0=W))  # llega a su base (4,0)
    assert info["blue_0"]["winner"] == "blue" and term["blue_0"]
    assert r["blue_0"] > 0 and abs(r["blue_0"] + r["red_0"]) < 1e-9


def test_score_resolves_before_strip():
    env = make()
    # el rojo espera junto a la base azul; el azul llega con la bandera
    place(env, walls=np.zeros((8, 8), bool), blue_0=(4, 1), red_0=(3, 0))
    give(env, "blue_0", "red")
    _, _, term, _, info = env.step(stay(env, blue_0=W, red_0=STRIP))
    assert info["blue_0"]["winner"] == "blue" and term["blue_0"]


def test_strip_only_in_own_half():
    env = make()
    # portador azul en la mitad AZUL (col 2): el rojo no puede quitarle la bandera
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 2), red_0=(2, 3))
    give(env, "blue_0", "red")
    assert not env.action_mask("red_0")[STRIP]
    env.step(stay(env, red_0=STRIP))
    assert env.holder["red"] == "blue_0"
    # portador azul en la mitad ROJA (col 5): sí se puede
    place(env, blue_0=(2, 5), red_0=(2, 6))
    assert env.action_mask("red_0")[STRIP]
    env.step(stay(env, red_0=STRIP))
    assert env.holder["red"] is None and env.flag_status("red") == 2
    assert env.cooldown["blue_0"] == 2


def test_strip_needs_adjacency():
    env = make()
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 4), red_0=(2, 7))
    give(env, "blue_0", "red")
    env.step(stay(env, red_0=STRIP))
    assert env.holder["red"] == "blue_0"


def test_owner_returns_dropped_flag_sharing_the_cell():
    env = make()
    # la bandera roja cae donde está el portador azul (en cooldown); el rojo la pisa
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 5), red_0=(2, 6))
    give(env, "blue_0", "red")
    env.step(stay(env, red_0=STRIP))
    env.step(stay(env, red_0=RED_W))  # entra a la celda compartida con la bandera en el suelo
    assert (env.flag["red"] == env.base["red"]).all() and env.holder["red"] is None
    assert env.stats["returns_red"] == 1


def test_rival_touching_ground_flag_picks_it_up():
    env = make()
    env.flag["red"] = np.array([2, 3])
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 2), red_0=(6, 6))
    env.step(stay(env, blue_0=E))
    assert env.holder["red"] == "blue_0"


def test_pass_to_adjacent_ally():
    env = make(n=2)
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 4), blue_1=(2, 3), red_0=(7, 7), red_1=(0, 7))
    give(env, "blue_0", "red")
    assert env.action_mask("blue_0")[PASS]
    env.step(stay(env, blue_0=PASS))
    assert env.holder["red"] == "blue_1" and (env.flag["red"] == (2, 3)).all()


def test_pass_without_ally_is_masked_and_noop():
    env = make(n=2)
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 4), blue_1=(6, 1), red_0=(7, 7), red_1=(0, 7))
    give(env, "blue_0", "red")
    assert not env.action_mask("blue_0")[PASS]
    env.step(stay(env, blue_0=PASS))
    assert env.holder["red"] == "blue_0"


# ---------- ocupación y camino garantizado ----------
def test_agents_block_but_bases_are_shared():
    env = make(n=2)
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 2), blue_1=(2, 3), red_0=(7, 7), red_1=(0, 7))
    env.step(stay(env, blue_0=E))  # celda ocupada por un aliado
    assert (env.pos["blue_0"] == (2, 2)).all()
    place(env, blue_0=(4, 6), red_0=(4, 7))  # el rojo está sobre su base
    env.step(stay(env, blue_0=E))  # el azul igual entra a la base roja (compartida)
    assert (env.pos["blue_0"] == (4, 7)).all()


def corridor_walls(size=8):
    """Muralla vertical en la columna 3 con un único hueco en la fila 2."""
    w = np.zeros((size, size), bool)
    w[:, 3] = True
    w[2, 3] = False
    return w


def test_cannot_close_last_corridor():
    env = make(n=2)
    place(env, walls=corridor_walls(), blue_0=(6, 1), blue_1=(7, 1), red_0=(1, 4), red_1=(0, 7))
    assert not env.forced
    # (2,4) es el único acceso al hueco desde el este: pararse ahí cerraría el camino
    assert not env.action_mask("red_0")[S]
    env.step(stay(env, red_0=S))
    assert (env.pos["red_0"] == (1, 4)).all() and env.stats["rejected_moves"] >= 1
    assert path_exists(env, "blue") and path_exists(env, "red")


def test_stepping_on_path_with_alternative_is_valid():
    env = make()
    place(env, walls=np.zeros((8, 8), bool), blue_0=(7, 1), red_0=(0, 7))
    path = env.path["blue"]
    cell = path[len(path) // 2]
    nb = (cell[0] - 1, cell[1]) if cell[0] > 0 else (cell[0] + 1, cell[1])
    env.pos["red_0"] = np.array(nb)
    calls = env.stats["bfs_calls"]
    env.step(stay(env, red_0=S if cell[0] > nb[0] else N))
    assert tuple(env.pos["red_0"]) == cell  # hay alternativa en mapa abierto: válido
    assert env.stats["bfs_calls"] > calls  # se recalculó solo porque pisó el camino
    assert cell not in env.path["blue"][1:-1] and path_exists(env, "blue")


def test_cannot_surround_base():
    env = make(n=3, size=8)
    # base azul (4,0): vecinos (3,0) (5,0) (4,1). Dos rojos ya ocupan dos de ellos.
    place(env, walls=np.zeros((8, 8), bool), blue_0=(0, 4), blue_1=(1, 4), blue_2=(7, 4),
          red_0=(3, 0), red_1=(5, 0), red_2=(4, 2))
    give(env, "blue_0", "red")
    assert not env.action_mask("red_2")[RED_W]  # cerrar el cerco está prohibido
    env.step(stay(env, red_2=RED_W))
    assert (env.pos["red_2"] == (4, 2)).all()
    assert path_exists(env, "blue")


def test_forced_blocker_must_move_after_event():
    env = make(n=2)
    place(env, walls=corridor_walls(), blue_0=(2, 5), blue_1=(7, 6), red_0=(0, 7), red_1=(6, 6))
    give(env, "blue_0", "red")  # el azul lleva la bandera roja en territorio rojo
    # un aliado azul aparece en el único hueco (2,3): tapa el camino del portador
    env.pos["blue_1"] = np.array([2, 3])
    env.path = {tm: None for tm in TEAMS}
    env._refresh_paths()
    assert "blue_1" in env.forced
    m = env.action_mask("blue_1")
    assert not m[STAY] and not m[STRIP] and not m[PASS] and m[1:5].any()
    env.step(stay(env))  # aunque pida quedarse, se mueve a una celda válida
    assert tuple(env.pos["blue_1"]) != (2, 3)
    for _ in range(20):  # sigue obligado hasta salir del camino tapado
        if not env.forced:
            break
        env.step(stay(env))
    assert path_exists(env, "blue") and not env.forced


def test_path_invariant_fuzz():
    """En episodios aleatorios con paredes, después de cada paso existe camino
    bandera -> base para ambos equipos (o hay obligados a moverse), y ningún
    agente entra a una celda no compartible ya ocupada."""
    rng = np.random.default_rng(0)
    for seed in range(60):
        env = CTFParallelEnv(size=10, n_per_team=2, max_steps=80, n_boxes=3, n_walls=3)
        env.reset(seed=seed)
        while env.agents:
            masks = {a: env.action_mask(a) for a in env.agents}
            acts = {a: int(rng.choice(np.flatnonzero(m))) for a, m in masks.items()}
            before = {a: tuple(env.pos[a]) for a in env.agents}
            shareable = {(r, c) for r in range(10) for c in range(10) if env._shareable((r, c))}
            env.step(acts)
            if not env.agents:
                break
            for e in TEAMS:
                assert path_exists(env, e) or env.forced, (seed, env.t, e)
            for a in before:  # quien se movió a una celda no compartible la tiene para sí
                q = tuple(env.pos[a])
                if q != before[a] and q not in shareable:  # regla vigente al moverse
                    others = [b for b in before if b != a and tuple(env.pos[b]) == q]
                    assert not others, (seed, env.t, a, others)
                assert not env.walls[q]


# ---------- mapas ----------
def test_maps_connected_symmetric_and_protected():
    rng = np.random.default_rng(1)
    size = 10
    bases = [np.array([5, 0]), np.array([5, 9])]
    prot = protected_cells(size, bases)
    n_walled = 0
    for _ in range(200):
        w = generate_map(size, rng, 3, 3, bases)
        n_walled += w.any()
        assert connected(w)
        assert (w == w[:, ::-1]).all()  # espejo horizontal
        assert not any(w[p] for p in prot)
    assert n_walled > 150  # casi siempre se genera un mapa con obstáculos


def test_bfs_distance_respects_walls():
    d = bfs_dist(corridor_walls(), (7, 2))
    assert d[7, 4] == 2 * 5 + 2  # debe subir hasta el hueco de la fila 2 y bajar


# ---------- recompensa ----------
def test_zero_sum_and_marker_cancels_at_end():
    for seed in range(5):
        env = CTFParallelEnv(size=8, n_per_team=2, max_steps=60, shaping_scale=0.5, n_boxes=2, n_walls=2)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total = 0.0
        while env.agents:
            _, r, _, _, info = env.step({a: int(rng.integers(7)) for a in env.agents})
            assert abs(r["blue_0"] + r["red_0"]) < 1e-9
            total += r["blue_0"]
        expected = {"blue": 1.0, "red": -1.0, None: 0.0}[info["blue_0"]["winner"]]
        assert abs(total - expected) < 1e-9, (seed, total)


def test_marker_moves_with_flags():
    env = make(shaping_scale=1.0)
    place(env, walls=np.zeros((8, 8), bool), blue_0=(4, 3), red_0=(0, 7))
    give(env, "blue_0", "red")
    env._M = env._marker()
    _, r, *_ = env.step(stay(env, blue_0=W))  # la bandera roja se acerca a la base azul
    assert r["blue_0"] > 0


# ---------- vista egocéntrica ----------
def test_red_view_mirrors_blue_view():
    env = make()
    place(env, walls=np.zeros((8, 8), bool), blue_0=(2, 1), red_0=(2, 6))
    assert np.allclose(env._obs("blue_0"), env._obs("red_0"))
    assert np.allclose(env.state("blue"), env.state("red"))


def test_red_actions_are_mirrored():
    env = make()
    place(env, walls=np.zeros((8, 8), bool), blue_0=(0, 0), red_0=(5, 5))
    env.step(stay(env, red_0=E))  # "este" del rojo = oeste del mundo
    assert (env.pos["red_0"] == (5, 4)).all()
