"""Planned trajectories and execution with pose error.

Supports:
- ``TRAJ_SOURCE="arc"``: legacy synthetic arc in front of the RIS
- ``TRAJ_SOURCE="grid_planner"``: 2D occupancy A* path (see motion_planner/)

The main line only needs p_plan / p_true; no ML predictor.

中文职责：生成计划轨迹 p_plan，并模拟执行误差得到 p_true。
输入：时间步 T、执行误差标准差 sigma、config.TRAJ_SOURCE；
输出：成对的 p_plan、p_true，以及逐步 ||p_true - p_plan||_2。
在流程中的位置：主线第一步——坐标/轨迹；其后 modes 用 p_plan 配相、p_true 评估。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from Physical_model import config


class PlannerTrajectories(TypedDict):
    """Paired planned and executed pose sequences.

    中文简释：计划/执行位姿对；配相看 p_plan，评估 SNR 看 p_true。
    """

    p_plan: np.ndarray  # (T, 3) motion-planner planned poses [m]
    p_true: np.ndarray  # (T, 3) true / executed poses [m]
    planner_pose_error_m: np.ndarray  # (T,) ||p_true - p_plan||_2 [m]


def _generate_arc_trajectory(T: int) -> np.ndarray:
    """Legacy synthetic arc (kept for对照 / TRAJ_SOURCE=arc)."""
    theta = np.linspace(
        config.TRAJ_THETA_START_RAD,
        config.TRAJ_THETA_END_RAD,
        T,
    )
    x = config.TRAJ_ARC_CX_M + config.TRAJ_ARC_RADIUS_M * np.sin(theta)
    y = config.TRAJ_ARC_CY_M + config.TRAJ_ARC_RADIUS_M * np.cos(theta)
    z = np.full(T, config.ROBOT_Z_M, dtype=float)
    return np.column_stack([x, y, z])

#T 时间戳

def generate_planner_trajectory(
    T: int | None = None,
    *,
    source: str | None = None,
) -> np.ndarray:
    """Generate planned trajectory p_plan of shape (T, 3).

    ``source`` overrides ``config.TRAJ_SOURCE`` when given
    (``"arc"`` or ``"grid_planner"``).

    中文简释：按 TRAJ_SOURCE 生成计划轨迹；grid_planner 走占用栅格 A*。
    """
    if T is None:
        T = config.NUM_TIME_STEPS
    if T < 1:
        raise ValueError("T must be >= 1")

    src = (config.TRAJ_SOURCE if source is None else source).strip().lower()
    if src == "arc":
        return _generate_arc_trajectory(T)
    if src == "grid_planner":
        from Physical_model.motion_planner import plan_robot_trajectory

        return plan_robot_trajectory(T)
    raise ValueError(
        f"unknown TRAJ_SOURCE={src!r}; expected 'arc' or 'grid_planner'"
    )


def execute_with_error(
    p_plan: np.ndarray,
    error_std_m: float | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> PlannerTrajectories:
    """Simulate execution: p_true = p_plan + noise.

    Models tracking / localization / execution mismatch relative to the
    planned path. RIS phases are designed from p_plan; SNR is evaluated
    on p_true.

    中文简释：模拟执行——p_true = p_plan + 各向同性高斯噪声。
    要点：sigma=0 时完美跟踪（Stage-1：p_true = p_plan）。
    注意：配相应用 p_plan，SNR 评估应用 p_true（本函数只产位姿，不配相）。

    Parameters
    ----------
    p_plan :
        Planned poses, shape (T, 3).
    error_std_m :
        Isotropic Gaussian std [m] for each axis. Defaults to
        config.PLANNER_POSE_ERROR_STD_M. Use 0 for perfect tracking
        (Stage-1 baseline: p_true = p_plan).
    rng :
        NumPy generator; if None, seeded from config.RANDOM_SEED.

    Returns
    -------
    PlannerTrajectories
        p_plan, p_true, and per-step planner_pose_error_m.
    """
    plan = np.asarray(p_plan, dtype=float)
    if plan.ndim != 2 or plan.shape[1] != 3:
        raise ValueError("p_plan must have shape (T, 3)")

    if error_std_m is None:
        error_std_m = config.PLANNER_POSE_ERROR_STD_M
    sigma = float(error_std_m)
    if sigma < 0.0:
        raise ValueError("error_std_m must be >= 0")

    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)

    # 步骤1：按 sigma 得到执行位姿 p_true（sigma=0 则与计划重合）
    if sigma == 0.0:
        p_true = plan.copy()
    else:
        noise = rng.normal(0.0, sigma, size=plan.shape)
        p_true = plan + noise

    # 步骤2：逐步规划误差 ||p_true - p_plan||_2（供 Stage-2 横轴）
    err = np.linalg.norm(p_true - plan, axis=1)
    return {
        "p_plan": plan,
        "p_true": p_true,
        "planner_pose_error_m": err,
    }


def make_paired_trajectories(
    T: int | None = None,
    error_std_m: float | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> PlannerTrajectories:
    """Convenience: generate p_plan then apply execute_with_error.

    中文简释：一键生成成对轨迹——先 p_plan，再叠加执行误差得 p_true。
    """
    p_plan = generate_planner_trajectory(T)
    return execute_with_error(p_plan, error_std_m, rng=rng)
