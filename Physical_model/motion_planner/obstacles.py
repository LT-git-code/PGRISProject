"""Axis-aligned obstacle boxes for planning (and later spatial NLoS).

中文职责：场景障碍物（AABB）；规划避障与后续直连可见性共用同一描述。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Physical_model import config


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned box in the x–y plane (z ignored for planning)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains_xy(self, x: float, y: float, *, inflate: float = 0.0) -> bool:
        """中文简释：点是否落在膨胀后的盒子内。"""
        return (
            (self.xmin - inflate) <= x <= (self.xmax + inflate)
            and (self.ymin - inflate) <= y <= (self.ymax + inflate)
        )


def default_scene_obstacles(
    boxes: tuple[tuple[float, float, float, float], ...] | None = None,
) -> list[BoxObstacle]:
    """Build obstacles from config.OBSTACLE_BOXES_M (or an override).

    中文简释：从 config 读取 AABB 列表，供栅格与可见性共用。
    """
    raw = config.OBSTACLE_BOXES_M if boxes is None else boxes
    out: list[BoxObstacle] = []
    for xmin, xmax, ymin, ymax in raw:
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"invalid obstacle box: {(xmin, xmax, ymin, ymax)}")
        out.append(BoxObstacle(float(xmin), float(xmax), float(ymin), float(ymax)))
    return out


def point_in_obstacles(
    xy: np.ndarray | tuple[float, float],
    obstacles: list[BoxObstacle],
    *,
    inflate: float = 0.0,
) -> bool:
    """中文简释：平面点是否落入任一障碍（可带膨胀）。"""
    x, y = float(xy[0]), float(xy[1])
    return any(ob.contains_xy(x, y, inflate=inflate) for ob in obstacles)


def segment_hits_obstacle(
    p0: np.ndarray | tuple[float, float],
    p1: np.ndarray | tuple[float, float],
    obstacles: list[BoxObstacle],
    *,
    inflate: float = 0.0,
    n_samples: int = 32,
) -> bool:
    """Approximate segment–box intersection via dense sampling (2D).

    中文简释：用线段采样粗判是否穿过障碍（后续 NLoS 直连也可用）。
    """
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    for t in np.linspace(0.0, 1.0, int(n_samples)):
        x = x0 + t * (x1 - x0)
        y = y0 + t * (y1 - y0)
        if point_in_obstacles((x, y), obstacles, inflate=inflate):
            return True
    return False
