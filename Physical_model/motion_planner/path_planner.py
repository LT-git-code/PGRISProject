"""Grid A* path planner (8-connected).

中文职责：在占用栅格上用 A* 求从起点到终点的折线路径（世界坐标 x–y）。
"""

from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np

from Physical_model.motion_planner.grid_map import (
    GridSpec,
    cell_to_world,
    in_bounds,
    is_free,
    world_to_cell,
)

# 8-connected moves: (dix, diy, cost)
_NEIGHBORS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, float(np.sqrt(2.0))),
    (-1, 1, float(np.sqrt(2.0))),
    (1, -1, float(np.sqrt(2.0))),
    (1, 1, float(np.sqrt(2.0))),
)


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _nearest_free(
    ix: int,
    iy: int,
    grid: np.ndarray,
    spec: GridSpec,
    *,
    max_radius: int = 20,
) -> tuple[int, int]:
    """If start/goal lands on occupied cell, search nearby free cell."""
    if is_free(ix, iy, grid, spec):
        return ix, iy
    for r in range(1, max_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                jx, jy = ix + dx, iy + dy
                if is_free(jx, jy, grid, spec):
                    return jx, jy
    raise RuntimeError(f"no free cell near ({ix}, {iy})")


def plan_path_astar(
    grid: np.ndarray,
    start_xy: np.ndarray | tuple[float, float],
    goal_xy: np.ndarray | tuple[float, float],
    spec: GridSpec,
) -> list[np.ndarray]:
    """Plan a path; return list of world (x, y) waypoints including start/goal.

    中文简释：A* 搜索；输出世界坐标折线（至少含起终点）。
    """
    sx, sy = world_to_cell(start_xy, spec)
    gx, gy = world_to_cell(goal_xy, spec)
    if not in_bounds(sx, sy, spec) or not in_bounds(gx, gy, spec):
        raise ValueError("start or goal outside grid bounds")
    start = _nearest_free(sx, sy, grid, spec)
    goal = _nearest_free(gx, gy, grid, spec)

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _f, _c, current = heapq.heappop(open_heap)
        if current == goal:
            break
        cx, cy = current
        for dix, diy, step in _NEIGHBORS:
            nx, ny = cx + dix, cy + diy
            if not is_free(nx, ny, grid, spec):
                continue
            # Discourage cutting corners through occupied diagonals
            if dix != 0 and diy != 0:
                if (not is_free(cx + dix, cy, grid, spec)) or (
                    not is_free(cx, cy + diy, grid, spec)
                ):
                    continue
            tentative = g_score[current] + step
            neigh = (nx, ny)
            if tentative < g_score.get(neigh, float("inf")):
                came_from[neigh] = current
                g_score[neigh] = tentative
                f = tentative + _heuristic(neigh, goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, neigh))
    else:
        raise RuntimeError("A* failed: no path from start to goal")

    # Reconstruct
    cells: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        cells.append(node)
        node = came_from[node]
    cells.reverse()

    waypoints = [cell_to_world(ix, iy, spec) for ix, iy in cells]
    # Snap first/last to exact requested start/goal (still free-checked upstream)
    waypoints[0] = np.asarray(start_xy, dtype=float).reshape(2)
    waypoints[-1] = np.asarray(goal_xy, dtype=float).reshape(2)
    return waypoints


def simplify_waypoints(
    waypoints: Iterable[np.ndarray],
    *,
    min_step_m: float = 0.25,
) -> list[np.ndarray]:
    """Drop near-duplicate consecutive points."""
    pts = [np.asarray(p, dtype=float).reshape(2) for p in waypoints]
    if not pts:
        return []
    out = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) >= min_step_m:
            out.append(p)
    if len(out) == 1 and len(pts) > 1:
        out.append(pts[-1])
    elif len(out) >= 1 and not np.allclose(out[-1], pts[-1]):
        out.append(pts[-1])
    return out
