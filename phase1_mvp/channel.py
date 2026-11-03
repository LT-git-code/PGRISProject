"""Wireless LoS channel models (far-field free-space) for phase-1.

中文职责：Phase-1 物理层信道建模（远场自由空间 LoS）。
输入：AP / RIS 阵元 / 机器人三维坐标与波长 λ；
输出：直连增益、级联信道 h_ar、h_ru，以及 h_eff = h_ar ⊙ h_ru。
在流程中的位置：坐标 →【本模块】→ 相位配置（beamforming）→ SNR。
注意：本模块只构造信道，不负责配相；design / evaluation 用哪组位姿由上层决定。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from phase1_mvp import config


class CascadedChannel(TypedDict):

    """AP–RIS–Robot cascaded LoS channels.

    中文简释：级联信道三元组；后续配相与 SNR 主要使用 h_eff。
    """
    h_ar: np.ndarray  # (N,) AP → RIS
    h_ru: np.ndarray  # (N,) RIS → Robot
    h_eff: np.ndarray  # (N,) element-wise cascade h_ar * h_ru  （即 h_ar ⊙ h_ru）

    """Far-field free-space complex path gain.

    Amplitude uses Friis factor ``λ / (4 π d)``; phase is ``exp(-j 2π d / λ)``
    with geometric distance as the phase reference.

    中文简释：自由空间复路径增益；幅度 Friis，相位按几何距离。
    要点：d 必须 > 0；返回值与 d 同形。

    Parameters
    ----------
    d :
        Link distance(s) in meters (scalar or array). Must be > 0.
    lambda_ :
        Carrier wavelength in meters.

    Returns
    -------
    complex or np.ndarray
        Complex path gain(s), same shape as ``d``.
    """
def path_loss_fs(d: np.ndarray | float, lambda_: float) -> np.ndarray | complex:

    d_arr = np.asarray(d, dtype=float)
    if np.any(d_arr <= 0.0):
        raise ValueError("distance d must be strictly positive")

    # 步骤1：Friis 幅度 λ/(4πd)
    amp = lambda_ / (4.0 * np.pi * d_arr)
    # 步骤2：几何相位 exp(-j 2π d / λ)
    phase = np.exp(-1j * (2.0 * np.pi / lambda_) * d_arr)
    gain = amp * phase

    if np.ndim(d) == 0:
        return complex(gain)
    return gain

    """LoS channel vector from a single TX to an RX array.

    Pure LoS, far-field phase model: each entry is the free-space path gain
    from ``tx_pos`` to the corresponding RX element (no scattering / polarization).

    中文简释：单发到接收阵列的 LoS 信道向量（每元一个复增益）。
    要点：无散射/极化；形状 (N,)。

    Parameters
    ----------
    tx_pos :
        Transmitter position, shape (3,).
    rx_positions :
        Receiver element positions, shape (N, 3).
    lambda_ :
        Carrier wavelength in meters.

    Returns
    -------
    np.ndarray
        Complex vector of shape (N,).
    """
def los_steering_vector(
    tx_pos: np.ndarray,
    rx_positions: np.ndarray,
    lambda_: float,
) -> np.ndarray:

    tx = np.asarray(tx_pos, dtype=float).reshape(3)
    rx = np.asarray(rx_positions, dtype=float)
    if rx.ndim != 2 or rx.shape[1] != 3:
        raise ValueError("rx_positions must have shape (N, 3)")

    # 步骤1：逐元几何距离；步骤2：套用自由空间路径增益
    distances = np.linalg.norm(rx - tx, axis=1)
    return path_loss_fs(distances, lambda_)


    """AP→robot scalar LoS channel with optional NLoS blocking.

    中文简释：AP→机器人直连标量信道；可选遮挡衰减 α。
    要点：用于 no-RIS 基线 SNR；与级联 h_eff 独立。

    Parameters
    ----------
    ap_pos, robot_pos :
        Positions in meters, shape (3,).
    lambda_ :
        Wavelength in meters.
    block_direct_link :
        If ``None``, uses ``config.BLOCK_DIRECT_LINK``.
        ``True`` → multiply by ``DIRECT_PATH_ATTENUATION_BLOCKED`` (very weak).
        ``False`` → open LoS (``α = DIRECT_PATH_ATTENUATION_OPEN``, default 1).
    """
def direct_ap_robot_channel(
    ap_pos: np.ndarray,
    robot_pos: np.ndarray,
    lambda_: float,
    *,
    block_direct_link: bool | None = None,
) -> complex:
    # 步骤1：按开关选取直连衰减系数 α（遮挡 / 开通）
    if block_direct_link is None:
        block_direct_link = config.BLOCK_DIRECT_LINK
    alpha = (
        config.DIRECT_PATH_ATTENUATION_BLOCKED
        if block_direct_link
        else config.DIRECT_PATH_ATTENUATION_OPEN
    )
    ap = np.asarray(ap_pos, dtype=float).reshape(3)
    robot = np.asarray(robot_pos, dtype=float).reshape(3)
    # 步骤2：几何距离 + 自由空间增益，再乘 α
    dist = float(np.linalg.norm(robot - ap))
    return complex(alpha * path_loss_fs(dist, lambda_))



    """Build AP→RIS→Robot cascaded LoS channels.

    中文简释：构造 AP→RIS→机器人级联 LoS 信道。
    要点：h_eff = h_ar ⊙ h_ru，供后续 φᴴ h_eff 相干增益使用。
    注意：robot_pos 是“这组信道对应的几何位姿”；上层应用 p_plan 配相、
    用 p_true 评估时，必须分别调用本函数，不可混用同一份 h_eff。

    Returns
    -------
    CascadedChannel
        ``h_ar``: AP → RIS, shape (N,)
        ``h_ru``: RIS → Robot, shape (N,)
        ``h_eff``: ``h_ar * h_ru`` (element-wise), for later ``φ^H h_eff``.
    """
def cascaded_channel(
    ap_pos: np.ndarray,
    ris_pos: np.ndarray,
    robot_pos: np.ndarray,
    lambda_: float,
) -> CascadedChannel:

    ris = np.asarray(ris_pos, dtype=float)
    if ris.ndim != 2 or ris.shape[1] != 3:
        raise ValueError("ris_pos must have shape (N, 3)")

    # 步骤1：AP → 各 RIS 阵元 → h_ar
    h_ar = los_steering_vector(ap_pos, ris, lambda_)
    # 步骤2：机器人 → 各 RIS 阵元 → h_ru（互易自由空间，与 RIS→机器人等价）
    h_ru = los_steering_vector(robot_pos, ris, lambda_)
    # 步骤3：逐元级联 h_eff = h_ar ⊙ h_ru（供 φᴴ h_eff）
    h_eff = h_ar * h_ru

    return {"h_ar": h_ar, "h_ru": h_ru, "h_eff": h_eff}


def _ris_element_grid() -> np.ndarray:
    """Minimal X–Z URA centered at config.POS_RIS_CENTER_M (for local checks).

    中文简释：本地自检用的 RIS X–Z 平面 URA 阵元坐标（不依赖 geometry 模块）。
    """
    n_x, n_z = config.RIS_NUM_X, config.RIS_NUM_Z
    spacing = config.RIS_ELEMENT_SPACING_M
    center = np.asarray(config.POS_RIS_CENTER_M, dtype=float)

    positions: list[np.ndarray] = []
    for ix in range(n_x):
        for iz in range(n_z):
            x_off = (ix - (n_x - 1) / 2.0) * spacing
            z_off = (iz - (n_z - 1) / 2.0) * spacing
            positions.append(center + np.array([x_off, 0.0, z_off]))
    return np.asarray(positions, dtype=float)


def _sanity_check_robot_in_front() -> float:
    """Assert |h_eff| is not near-zero when the robot faces the RIS; return ||h_eff||_2.

    中文简释：机器人正对 RIS 时检查 ||h_eff||₂ 非近零，避免信道实现错误。
    """
    ap_pos = np.asarray(config.POS_AP_M, dtype=float)
    ris_pos = _ris_element_grid()
    # 机器人置于 RIS 正前方（+Y），略低于面板高度，用于几何自检
    robot_pos = np.array(
        [config.POS_RIS_CENTER_M[0], config.ROBOT_Y_M, config.ROBOT_Z_M],
        dtype=float,
    )

    ch = cascaded_channel(ap_pos, ris_pos, robot_pos, config.WAVELENGTH_M)
    h_eff = ch["h_eff"]
    h_eff_norm = float(np.linalg.norm(h_eff))

    if not np.all(np.isfinite(h_eff)):
        raise AssertionError("h_eff contains non-finite values")
    if np.max(np.abs(h_eff)) < 1e-20:
        raise AssertionError("|h_eff| is near zero for robot in front of RIS")
    if h_eff_norm < 1e-20:
        raise AssertionError("||h_eff||_2 is near zero for robot in front of RIS")

    return h_eff_norm


if __name__ == "__main__":
    norm_h_eff = _sanity_check_robot_in_front()
    print(f"||h_eff||_2 = {norm_h_eff:.6e}")
