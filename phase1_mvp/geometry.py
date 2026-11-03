"""Geometry helpers: AP / RIS positions (static scene).

中文职责：静态场景几何——AP 位置与 RIS 阵元世界坐标。
输入：读取 config 中的 AP/RIS 中心、阵元数与间距；
输出：形状 (3,) 的 AP/中心，或 (N, 3) 的 RIS 阵元坐标（米）。
在流程中的位置：坐标准备的第一步；供 channel.cascaded_channel 使用。
注意：机器人位姿不在此模块（由轨迹 / 上层传入 p_plan 或 p_true）。
"""

from __future__ import annotations

import numpy as np

from phase1_mvp import config


def get_ap_position() -> np.ndarray:
    """Return AP position as shape (3,) array in meters.

    中文简释：返回 AP 世界坐标 (3,)，供构造 h_ar / 直连信道。
    """
    return np.asarray(config.POS_AP_M, dtype=float)


def get_ris_center() -> np.ndarray:
    """Return RIS center position as shape (3,) array in meters.

    中文简释：返回 RIS 面板中心 (3,)；阵元网格绕此中心展开。
    """
    return np.asarray(config.POS_RIS_CENTER_M, dtype=float)


def get_ris_element_positions() -> np.ndarray:
    """Return RIS element world positions on an X–Z URA.

    中文简释：生成 X–Z 平面均匀矩形阵列（URA）各阵元世界坐标。
    要点：N = RIS_NUM_X × RIS_NUM_Z；y 偏移为 0（面板法向沿 +Y 场景约定）。
    注意：返回的是固定 RIS 几何，与机器人 p_plan / p_true 无关。

    Returns
    -------
    np.ndarray
        Shape (N, 3), N = RIS_NUM_X * RIS_NUM_Z, coordinates in meters.
    """
    n_x, n_z = config.RIS_NUM_X, config.RIS_NUM_Z
    spacing = config.RIS_ELEMENT_SPACING_M
    center = get_ris_center()

    # 步骤1：按 (ix, iz) 相对中心偏移；步骤2：堆叠为 (N, 3)
    positions: list[np.ndarray] = []
    for ix in range(n_x):
        for iz in range(n_z):
            x_off = (ix - (n_x - 1) / 2.0) * spacing
            z_off = (iz - (n_z - 1) / 2.0) * spacing
            positions.append(center + np.array([x_off, 0.0, z_off], dtype=float))
    return np.asarray(positions, dtype=float)
