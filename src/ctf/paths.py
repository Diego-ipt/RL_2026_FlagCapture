"""BFS en la grilla (4-vecinos) para distancias, conectividad y caminos."""
from __future__ import annotations

from collections import deque

import numpy as np

NEIGH = ((-1, 0), (1, 0), (0, 1), (0, -1))


def bfs_dist(walls: np.ndarray, src) -> np.ndarray:
    """Distancia de camino desde `src` a cada celda (-1 si es pared o inalcanzable)."""
    h, w = walls.shape
    dist = np.full((h, w), -1, int)
    dist[src] = 0
    q = deque([tuple(src)])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGH:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not walls[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def connected(walls: np.ndarray) -> bool:
    """True si todas las celdas libres forman una sola componente."""
    free = np.argwhere(~walls)
    if len(free) == 0:
        return False
    return int((bfs_dist(walls, tuple(free[0])) >= 0).sum()) == len(free)


def bfs_path(walls: np.ndarray, start, goal, blocked=frozenset()):
    """Camino más corto start→goal (lista de celdas, incluye ambos extremos) que no pasa
    por paredes ni por celdas de `blocked`. start y goal siempre son transitables.
    Devuelve None si no existe."""
    start, goal = tuple(start), tuple(goal)
    if start == goal:
        return [start]
    h, w = walls.shape
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        for dr, dc in NEIGH:
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in prev or not (0 <= nxt[0] < h and 0 <= nxt[1] < w) or walls[nxt]:
                continue
            if nxt != goal and nxt in blocked:
                continue
            prev[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            q.append(nxt)
    return None
