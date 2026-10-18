"""Generador de mapas con cajas y murallas, simétrico en espejo horizontal.

Garantías (validadas, si no se reintenta):
- todas las celdas libres forman una sola componente conexa (siempre hay camino
  entre cualquier par de celdas libres, en particular bandera -> base);
- las bases y sus 4 vecinos están libres;
- las columnas de spawn (0 y size-1) están libres.
"""
from __future__ import annotations

import numpy as np

from ctf.paths import connected


def protected_cells(size, bases):
    cells = set()
    for b in bases:
        r, c = int(b[0]), int(b[1])
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)):
            if 0 <= r + dr < size and 0 <= c + dc < size:
                cells.add((r + dr, c + dc))
    for r in range(size):
        cells.add((r, 0))
        cells.add((r, size - 1))
    return cells


def _shapes(rng, n_boxes, n_walls):
    """Obstáculos como listas de offsets (fila, col)."""
    out = []
    for _ in range(n_boxes):
        k = int(rng.integers(1, 3))  # caja 1x1 o 2x2
        out.append([(i, j) for i in range(k) for j in range(k)])
    for _ in range(n_walls):
        L = int(rng.integers(2, 5))  # muralla de largo 2-4
        out.append([(i, 0) for i in range(L)] if rng.random() < 0.5 else [(0, j) for j in range(L)])
    return out


def generate_map(size, rng, n_boxes=2, n_walls=2, bases=None, max_tries=50):
    """Devuelve una matriz bool de paredes. Los obstáculos se colocan en la mitad
    izquierda (columnas 1..size//2-1) y se reflejan en la derecha."""
    walls = np.zeros((size, size), bool)
    if n_boxes == 0 and n_walls == 0:
        return walls
    prot = protected_cells(size, bases or [])
    half = size // 2
    for _ in range(max_tries):
        walls[:] = False
        for shape in _shapes(rng, n_boxes, n_walls):
            r0 = int(rng.integers(0, size))
            c0 = int(rng.integers(1, max(2, half)))
            for dr, dc in shape:
                r, c = r0 + dr, c0 + dc
                mc = size - 1 - c  # columna espejo
                if 0 <= r < size and 1 <= c < half and (r, c) not in prot and (r, mc) not in prot:
                    walls[r, c] = True
                    walls[r, mc] = True
        if walls.any() and connected(walls):
            return walls
    walls[:] = False  # sin mapa válido: vacío
    return walls
