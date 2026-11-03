"""SNR modes for planner-assisted RIS control.

Glossary
--------
- no_ris      -- direct AP->robot only (blocked Friis baseline)
- ideal_ris   -- phi from h_eff(p_true) (oracle; upper bound)
- planner_ris -- phi from h_eff(p_plan); SNR always on h_eff(p_true)

Physics reuse phase1_mvp: cascade h_eff = h_ar * h_ru (element-wise),
unit-modulus deterministic_phase, Hermitian gain
phi^H h_eff = np.vdot(phi, h_eff).

Hard rule: default path never imports phase2_traj or loads .pth.

中文职责：规划器辅助 RIS 的三种 SNR 模式（业务编排）。
输入：成对位姿 p_plan / p_true；
输出：no_ris / ideal_ris / planner_ris 的 SNR_dB 时间序列。
在流程中的位置：轨迹之后 →【本模块：配相+评估】→ Stage 出图/扫参。
物理层复用自 phase1_mvp（channel/beamforming/snr），本文件只做业务封装。
注意：配相只用 p_plan；SNR 评估只用 p_true（design vs evaluation 分离）；主线无 ML。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from phase1_mvp.beamforming import deterministic_phase
from phase1_mvp.channel import cascaded_channel, direct_ap_robot_channel
from phase1_mvp.geometry import get_ap_position, get_ris_element_positions
from phase1_mvp.snr import compute_snr, dbm_to_watt

from thesis_planner_ris import config


class ModeSnrResult(TypedDict):
    """Per-time-step SNR curves [dB] evaluated on p_true.

    中文简释：各模式 SNR 曲线；评估几何一律为 p_true。
    """

    time: np.ndarray
    snr_no_ris: np.ndarray
    snr_ideal_ris: np.ndarray
    snr_planner_ris: np.ndarray
    planner_pose_error_m: np.ndarray


def _power() -> tuple[float, float]:
    """中文简释：读取 Pt、sigma^2（线性瓦特），供 SNR 公式使用。"""
    pt = dbm_to_watt(config.TX_POWER_DBM)
    sigma2 = dbm_to_watt(config.NOISE_POWER_DBM)
    return pt, sigma2


def _h_eff_at(robot_pos: np.ndarray) -> np.ndarray:
    """Cascaded effective channel at a robot pose (phase-1 physics).

    中文简释：在给定机器人位姿处构造 h_eff = h_ar ⊙ h_ru（复用 phase1）。
    注意：传入 p_plan 则得到设计信道；传入 p_true 则得到评估信道。
    """
    ap = get_ap_position()
    ris = get_ris_element_positions()
    ch = cascaded_channel(
        ap,
        ris,
        np.asarray(robot_pos, dtype=float).reshape(3),
        config.WAVELENGTH_M,
    )
    return ch["h_eff"]


def design_phi_from_plan(p_plan: np.ndarray) -> np.ndarray:
    """Design unit-modulus RIS phases from the planned pose.

    phi = deterministic_phase(h_eff(p_plan)) with |phi_n| = 1.

    中文简释：用计划位姿 p_plan 的 h_eff 做单位模相位对齐，得到 phi。
    注意：此处只设计相位，不计算 SNR。
    """
    return deterministic_phase(_h_eff_at(p_plan))


def snr_no_ris(p_true: np.ndarray) -> float:
    """No-RIS baseline SNR [dB] at the true / executed pose.

    中文简释：无 RIS 基线——直连信道在 p_true 上评估 SNR_dB。
    """
    pt, sigma2 = _power()
    ap = get_ap_position()
    h_direct = direct_ap_robot_channel(
        ap,
        np.asarray(p_true, dtype=float).reshape(3),
        config.WAVELENGTH_M,
        block_direct_link=config.BLOCK_DIRECT_LINK,
    )
    dummy = np.ones(1, dtype=np.complex128)
    return compute_snr(dummy, dummy, pt, sigma2, mode="no_ris", no_ris_gain=h_direct)


def snr_ideal_ris(p_true: np.ndarray) -> float:
    """Oracle RIS SNR [dB]: phi from p_true, evaluate on p_true.

    中文简释：理想/上界——用 p_true 配相且在同一 p_true 信道上评估。
    """
    pt, sigma2 = _power()
    h_true = _h_eff_at(p_true)
    phi = deterministic_phase(h_true)
    return compute_snr(h_true, phi, pt, sigma2, mode="ris")


def snr_planner_ris(p_plan: np.ndarray, p_true: np.ndarray) -> float:
    """Planner-assisted RIS SNR [dB].

    Design phi from p_plan; always evaluate on h_eff(p_true).

    中文简释：规划器辅助主模式。
    步骤1：用 p_plan 构造设计信道并求 phi（|phi_n|=1）
    步骤2：用 p_true 的 h_eff 评估 SNR（注意不是 p_plan）
    """
    pt, sigma2 = _power()
    # 步骤1：phi <- h_eff(p_plan)
    phi = design_phi_from_plan(p_plan)
    # 步骤2：评估信道 h_eff(p_true)；design vs evaluation 分离
    h_true = _h_eff_at(p_true)
    return compute_snr(h_true, phi, pt, sigma2, mode="ris")


def simulate_modes(
    p_plan: np.ndarray,
    p_true: np.ndarray,
    *,
    planner_pose_error_m: np.ndarray | None = None,
) -> ModeSnrResult:
    """Compute no_ris / ideal_ris / planner_ris over a trajectory.

    Parameters
    ----------
    p_plan, p_true :
        Shape (T, 3).
    planner_pose_error_m :
        Optional precomputed ||p_true - p_plan||; else computed here.

    中文简释：沿轨迹逐时刻计算三种模式 SNR，并附带规划误差序列。
    """
    plan = np.asarray(p_plan, dtype=float)
    true = np.asarray(p_true, dtype=float)
    if plan.shape != true.shape or plan.ndim != 2 or plan.shape[1] != 3:
        raise ValueError("p_plan and p_true must both have shape (T, 3)")

    T = plan.shape[0]
    if planner_pose_error_m is None:
        err = np.linalg.norm(true - plan, axis=1)
    else:
        err = np.asarray(planner_pose_error_m, dtype=float).reshape(T)

    snr_nr = np.empty(T, dtype=float)
    snr_id = np.empty(T, dtype=float)
    snr_pl = np.empty(T, dtype=float)

    # 步骤：逐时刻分别算 no_ris / ideal / planner（评估位姿均为 true[t]）
    for t in range(T):
        snr_nr[t] = snr_no_ris(true[t])
        snr_id[t] = snr_ideal_ris(true[t])
        snr_pl[t] = snr_planner_ris(plan[t], true[t])

    return {
        "time": np.arange(T, dtype=float),
        "snr_no_ris": snr_nr,
        "snr_ideal_ris": snr_id,
        "snr_planner_ris": snr_pl,
        "planner_pose_error_m": err,
    }
