"""Resample polyline waypoints to a fixed-length (T, 3) trajectory.

中文职责：将 2D 折线等弧长采样为 (T,3) 位姿序列（z 固定）。
"""

from __future__ import annotations

import numpy as np


def polyline_arclength(waypoints: list[np.ndarray] | np.ndarray) -> float:
    """Total 2D path length in meters."""
    pts = np.asarray(waypoints, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)))


def resample_polyline(
    waypoints: list[np.ndarray] | np.ndarray,
    T: int,
    *,
    z: float,
) -> np.ndarray:
    """Equal-arclength resampling to shape (T, 3).

    中文简释：沿折线等弧长取 T 个点，z 填常数。
    """
    if T < 1:
        raise ValueError("T must be >= 1")
    pts = np.asarray(waypoints, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("waypoints must be (N, 2+) array-like")
    if pts.shape[0] == 0:
        raise ValueError("waypoints must be non-empty")
    if pts.shape[0] == 1 or polyline_arclength(pts) < 1e-9:
        xy = np.repeat(pts[0, :2].reshape(1, 2), T, axis=0)
        zcol = np.full((T, 1), float(z), dtype=float)
        return np.hstack([xy, zcol])

    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    targets = np.linspace(0.0, total, T)
    xy = np.empty((T, 2), dtype=float)
    for i, s in enumerate(targets):
        j = int(np.searchsorted(cum, s, side="right") - 1)
        j = min(max(j, 0), len(seg) - 1)
        span = seg[j]
        if span < 1e-12:
            xy[i] = pts[j, :2]
        else:
            alpha = (s - cum[j]) / span
            xy[i] = (1.0 - alpha) * pts[j, :2] + alpha * pts[j + 1, :2]
    zcol = np.full((T, 1), float(z), dtype=float)
    return np.hstack([xy, zcol])
