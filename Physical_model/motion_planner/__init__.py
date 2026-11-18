"""Lightweight 2D grid motion planner → (T, 3) robot trajectories.

中文职责：障碍物 → 占用栅格 → A* → 等弧长采样，输出 p_plan 形状 (T,3)。
"""

from __future__ import annotations

import numpy as np

from Physical_model import config
from Physical_model.motion_planner.grid_map import (
    build_occupancy_grid,
    grid_spec_from_config,
)
from Physical_model.motion_planner.obstacles import default_scene_obstacles
from Physical_model.motion_planner.path_planner import plan_path_astar, simplify_waypoints
from Physical_model.motion_planner.traj_sample import resample_polyline


def plan_robot_trajectory(T: int | None = None) -> np.ndarray:
    """Plan a collision-free path and resample to (T, 3).

    中文简释：主入口——读 config 场景，A* 规划后采样为固定长度轨迹。
    """
    if T is None:
        T = config.NUM_TIME_STEPS
    if T < 1:
        raise ValueError("T must be >= 1")
    #生成障碍物
    obstacles = default_scene_obstacles()
    #单位栅格像素大小
    spec = grid_spec_from_config(config)
    #栅格化膨胀后写入数组，grid数组是bool数组，记录实际坐标系下该点是否在栅格像素内，可以理解为像素坐标
    grid = build_occupancy_grid(
        obstacles,
        spec,
        robot_radius_m=float(config.PLANNER_ROBOT_RADIUS_M),
    )
    #获取路径起点和终点
    start = np.asarray(config.PLAN_START_XY_M, dtype=float)
    goal = np.asarray(config.PLAN_GOAL_XY_M, dtype=float)

    # Start/goal must not lie inside inflated obstacles
    from Physical_model.motion_planner.obstacles import point_in_obstacles

    inflate = float(config.PLANNER_ROBOT_RADIUS_M) #膨胀大小

    #判断起点终点是否包括了障碍物（带膨胀大小）
    if point_in_obstacles(start, obstacles, inflate=inflate):
        raise ValueError(f"PLAN_START_XY_M is inside an obstacle: {start}")
    if point_in_obstacles(goal, obstacles, inflate=inflate):
        raise ValueError(f"PLAN_GOAL_XY_M is inside an obstacle: {goal}")


    waypoints = plan_path_astar(grid, start, goal, spec)
    waypoints = simplify_waypoints(waypoints, min_step_m=0.5 * spec.resolution)
    return resample_polyline(waypoints, T, z=float(config.ROBOT_Z_M))


__all__ = ["plan_robot_trajectory"]
