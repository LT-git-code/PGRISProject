"""Occupancy grid from axis-aligned obstacles.

中文职责：将障碍物光栅化为占用栅格，并提供世界坐标 ↔ 格子索引转换。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Physical_model.motion_planner.obstacles import BoxObstacle


@dataclass(frozen=True)
class GridSpec:
    """Top-down x–y grid specification (meters)."""
    """俯视x-y坐标时怎么栅格化分割，给A*使用，并非障碍物本身，栅格像素的大小，在配置中读取"""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    @property
    def n_x(self) -> int:
        return int(np.ceil((self.x_max - self.x_min) / self.resolution))

    @property
    def n_y(self) -> int:
        return int(np.ceil((self.y_max - self.y_min) / self.resolution))


def grid_spec_from_config(config_module) -> GridSpec:
    """中文简释：从 Physical_model.config 读取栅格范围与分辨率。"""
    return GridSpec(
        x_min=float(config_module.GRID_X_MIN_M),
        x_max=float(config_module.GRID_X_MAX_M),
        y_min=float(config_module.GRID_Y_MIN_M),
        y_max=float(config_module.GRID_Y_MAX_M),
        resolution=float(config_module.GRID_RESOLUTION_M),
    )


def world_to_cell(xy: np.ndarray | tuple[float, float], spec: GridSpec) -> tuple[int, int]:
    """Map world (x, y) → (ix, iy). iy grows with +y."""
    x, y = float(xy[0]), float(xy[1])
    ix = int(np.floor((x - spec.x_min) / spec.resolution))
    iy = int(np.floor((y - spec.y_min) / spec.resolution))
    return ix, iy


def cell_to_world(ix: int, iy: int, spec: GridSpec) -> np.ndarray:
    """Map cell center → world (x, y)."""
    x = spec.x_min + (ix + 0.5) * spec.resolution
    y = spec.y_min + (iy + 0.5) * spec.resolution
    return np.array([x, y], dtype=float)


def in_bounds(ix: int, iy: int, spec: GridSpec) -> bool:
    return 0 <= ix < spec.n_x and 0 <= iy < spec.n_y


def build_occupancy_grid(
    obstacles: list[BoxObstacle],
    spec: GridSpec,
    *,
    robot_radius_m: float = 0.0,
) -> np.ndarray:
    """Return bool grid shape (n_x, n_y); True = occupied.

    中文简释：障碍膨胀 robot_radius 后写入占用栅格。
    """
    grid = np.zeros((spec.n_x, spec.n_y), dtype=bool)
    inflate = float(robot_radius_m)
    for ix in range(spec.n_x):
        for iy in range(spec.n_y):
            xy = cell_to_world(ix, iy, spec)
            for ob in obstacles:
                if ob.contains_xy(float(xy[0]), float(xy[1]), inflate=inflate):
                    grid[ix, iy] = True
                    break
    return grid


def is_free(ix: int, iy: int, grid: np.ndarray, spec: GridSpec) -> bool:
    """中文简释：格子在界内且非占用。"""
    return in_bounds(ix, iy, spec) and (not bool(grid[ix, iy]))
