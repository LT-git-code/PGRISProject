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

# 8-connected moves: (dix, diy, cost) 8邻域空间集定义
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

# 欧式距离
def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

# 函数作用是：如果用户传入的起点或终点不小心踩在了障碍物格里，代码不会直接报错崩溃，而是会自动向外螺线搜索最近的自由网格作为真正的起终点。
# 函数通过螺旋式外扩搜索（曼哈顿距离层层向外扩展，最大半径 20 格），寻找离原始坐标最近的安全自由网格，保证算法能够顺利起步和收尾
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

#A*实现路径规划
def plan_path_astar(
    grid: np.ndarray,
    start_xy: np.ndarray | tuple[float, float],
    goal_xy: np.ndarray | tuple[float, float],
    spec: GridSpec,
) -> list[np.ndarray]:
    """Plan a path; return list of world (x, y) waypoints including start/goal.

    中文简释：A* 搜索；输出世界坐标折线（至少含起终点）。
    """
    #将世界坐标（米）转换成栅格坐标（像素/铬）
    sx, sy = world_to_cell(start_xy, spec)
    gx, gy = world_to_cell(goal_xy, spec)
    if not in_bounds(sx, sy, spec) or not in_bounds(gx, gy, spec):
        raise ValueError("start or goal outside grid bounds")
    start = _nearest_free(sx, sy, grid, spec)
    goal = _nearest_free(gx, gy, grid, spec)

    # 核心数据结构
    open_heap: list[tuple[float, int, tuple[int, int]]] = [] #优先队列
    counter = 0 #计数器
    heapq.heappush(open_heap, (0.0, counter, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None} #父节点指针字典,记录每一个节点是哪一个来的
    g_score: dict[tuple[int, int], float] = {start: 0.0} #实际代价字典，从起点开始到当前仍意节点的实际开销

    while open_heap:
        _f, _c, current = heapq.heappop(open_heap)
        #弹出最优节点，如果是终点，直接跳出循环
        if current == goal:
            break
        cx, cy = current
        #遍历8邻域与防切角
        for dix, diy, step in _NEIGHBORS:
            nx, ny = cx + dix, cy + diy
            if not is_free(nx, ny, grid, spec):
                continue
            # Discourage cutting corners through occupied diagonals（当机器人斜向移动的时候）
            if dix != 0 and diy != 0:
                # 如果斜向不是自由空间，那么不会进入这里找路径节点
                if (not is_free(cx + dix, cy, grid, spec)) or (
                    not is_free(cx, cy + diy, grid, spec)
                ):
                    continue
            # 当前节点到达邻居的新代价
            tentative = g_score[current] + step
            neigh = (nx, ny)
            # 如果这个新代价比以往记录的代价更小（或者邻居从未被访问过），说明找到了更好的一条路：
            if tentative < g_score.get(neigh, float("inf")):
                # 更新其父节点 came_from
                came_from[neigh] = current
                # 更新其 g_score
                g_score[neigh] = tentative
                #计算综合代价 f = g + h（其中 h 由欧几里得距离 _heuristic 提供）
                f = tentative + _heuristic(neigh, goal)
                counter += 1
                # 压入优先队列
                heapq.heappush(open_heap, (f, counter, neigh))
    else:
        raise RuntimeError("A* failed: no path from start to goal")

    # Reconstruct 路径回溯
    cells: list[tuple[int, int]] = []
    node: tuple[int, int] | None = goal
    while node is not None:
        cells.append(node)
        node = came_from[node]
    cells.reverse()

    # 将栅格化路径再返还成世界模型（x,y）
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
