"""Closed-loop demo frames: p_plan designs phi; SNR on p_true.

No phase2_traj / torch / .pth. Synthetic p_plan stands in for a motion planner.

中文职责：Demo 闭环帧数据：p_plan 配相，在 p_true 上评估 SNR。
输入：时间步数、执行误差 sigma（默认非零以便可视化）；
输出：DemoFrame 列表（位姿 + 三模式 SNR）。
在流程中的位置：Stage-4 数据源；animate / export / run_demo 的上游。
注意：无 ML；合成 p_plan 仅替代运动规划器。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from thesis_planner_ris import config
from thesis_planner_ris.modes import (
    snr_ideal_ris,
    snr_no_ris,
    snr_planner_ris,
)
from thesis_planner_ris.planner_traj import make_paired_trajectories


@dataclass
class DemoFrame:
    """One time step for the planner-label demo.

    中文简释：单帧：计划/真实位姿、规划误差、三模式 SNR_dB。
    """

    t: int
    p_true: np.ndarray
    p_plan: np.ndarray
    planner_pose_error_m: float
    snr_no_ris: float
    snr_ideal_ris: float
    snr_planner_ris: float


def run_demo_loop(
    *,
    n_time_steps: int | None = None,
    error_std_m: float | None = None,
    seed: int | None = None,
) -> list[DemoFrame]:
    """Build per-step frames for animation / export.

    Parameters
    ----------
    n_time_steps :
        Trajectory length (default config.DEMO_NUM_TIME_STEPS).
    error_std_m :
        Execution noise so p_true differs from p_plan for viz
        (default config.DEMO_ERROR_STD_M).
    seed :
        RNG seed for execute_with_error.

    中文简释：生成成对轨迹，逐时刻调 modes 计算三模式 SNR。
    注意：snr_planner_ris(p_plan, p_true)：配相用计划位姿，评估用真实位姿。
    """
    if n_time_steps is None:
        n_time_steps = config.DEMO_NUM_TIME_STEPS
    if error_std_m is None:
        error_std_m = config.DEMO_ERROR_STD_M
    if seed is None:
        seed = config.RANDOM_SEED

    # 步骤1：合成 p_plan + 执行误差 -> p_true（默认 sigma>0 以便区分显示）
    rng = np.random.default_rng(seed)
    paired = make_paired_trajectories(
        T=n_time_steps,
        error_std_m=error_std_m,
        rng=rng,
    )
    p_plan = paired["p_plan"]
    p_true = paired["p_true"]
    err = paired["planner_pose_error_m"]

    # 步骤2：逐帧调用 modes（物理层在 modes 内复用 phase1）
    frames: list[DemoFrame] = []
    for t in range(n_time_steps):
        frames.append(
            DemoFrame(
                t=t,
                p_true=p_true[t].copy(),
                p_plan=p_plan[t].copy(),
                planner_pose_error_m=float(err[t]),
                snr_no_ris=snr_no_ris(p_true[t]),
                snr_ideal_ris=snr_ideal_ris(p_true[t]),
                snr_planner_ris=snr_planner_ris(p_plan[t], p_true[t]),
            )
        )
    return frames


def frames_to_arrays(frames: Sequence[DemoFrame]) -> dict[str, np.ndarray]:
    """Stack frame fields into arrays for plotting.

    中文简释：将帧列表堆叠为数组，供 animate_mpl 绘图。
    """
    return {
        "t": np.asarray([f.t for f in frames], dtype=float),
        "p_true": np.stack([f.p_true for f in frames], axis=0),
        "p_plan": np.stack([f.p_plan for f in frames], axis=0),
        "planner_pose_error_m": np.asarray(
            [f.planner_pose_error_m for f in frames], dtype=float
        ),
        "snr_no_ris": np.asarray([f.snr_no_ris for f in frames], dtype=float),
        "snr_ideal_ris": np.asarray([f.snr_ideal_ris for f in frames], dtype=float),
        "snr_planner_ris": np.asarray(
            [f.snr_planner_ris for f in frames], dtype=float
        ),
    }
