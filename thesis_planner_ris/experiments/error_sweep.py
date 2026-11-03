"""Sweep mean planner_pose_error_m and aggregate SNR modes.

For each execution-noise std in config.ERROR_SWEEP_STD_M, draw Monte Carlo
executions p_true = p_plan + noise, compute no_ris / ideal_ris / planner_ris
(SNR always on p_true), and average over time and trials.

X-axis quantity for thesis plots is the empirical mean
planner_pose_error_m = ||p_true - p_plan||_2 (not ML ADE/FDE).

中文职责：Stage-2 误差扫描——对不同执行噪声 sigma 求三模式均值 SNR。
输入：error_std 列表、蒙卡次数、轨迹长度；
输出：每个 sigma 一行：mean_planner_pose_error_m 与各模式 SNR_dB。
在流程中的位置：run_stage2 调用；横轴用实测 ||p_true-p_plan||，非 ML ADE/FDE。
注意：配相用 p_plan，评估用 p_true（经 modes.simulate_modes）；无 ML。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from thesis_planner_ris import config
from thesis_planner_ris.modes import simulate_modes
from thesis_planner_ris.planner_traj import (
    execute_with_error,
    generate_planner_trajectory,
)


class SweepRow(TypedDict):
    """One aggregated point on the error sweep curve.

    中文简释：扫描曲线上一个聚合点（某 sigma 下蒙卡+时间均值）。
    """

    error_std_m: float
    mean_planner_pose_error_m: float
    snr_no_ris_db: float
    snr_ideal_ris_db: float
    snr_planner_ris_db: float
    snr_planner_std_db: float
    n_monte_carlo: int
    n_time_steps: int


class SweepResult(TypedDict):
    """中文简释：完整扫描结果：rows + 对齐数组（绘图/CSV 用）。"""

    rows: list[SweepRow]
    # Arrays aligned with rows (convenient for plotting / CSV)
    error_std_m: np.ndarray
    mean_planner_pose_error_m: np.ndarray
    snr_no_ris_db: np.ndarray
    snr_ideal_ris_db: np.ndarray
    snr_planner_ris_db: np.ndarray
    snr_planner_std_db: np.ndarray


def run_error_sweep(
    error_stds: tuple[float, ...] | list[float] | None = None,
    *,
    n_monte_carlo: int | None = None,
    n_time_steps: int | None = None,
    seed: int | None = None,
) -> SweepResult:
    """Run the Stage-2 planner-pose-error sweep.

    Parameters
    ----------
    error_stds :
        Isotropic Gaussian std [m] for execute_with_error. Defaults to
        config.ERROR_SWEEP_STD_M (0 .. ~2 m).
    n_monte_carlo :
        Trials per std (independent noise draws).
    n_time_steps :
        Planned traj length for the sweep.
    seed :
        Base RNG seed; trial k uses a derived child seed.

    中文简释：对每个执行噪声 sigma：蒙卡抽 p_true，跑三模式，再对试验/时间求均。
    """
    if error_stds is None:
        error_stds = config.ERROR_SWEEP_STD_M
    if n_monte_carlo is None:
        n_monte_carlo = config.ERROR_SWEEP_N_MONTE_CARLO
    if n_time_steps is None:
        n_time_steps = config.ERROR_SWEEP_NUM_TIME_STEPS
    if seed is None:
        seed = config.RANDOM_SEED

    # 步骤1：生成固定计划轨迹 p_plan（各 sigma 共用）
    p_plan = generate_planner_trajectory(n_time_steps)
    rows: list[SweepRow] = []

    for sigma in error_stds:
        sigma_f = float(sigma)
        means_nr: list[float] = []
        means_id: list[float] = []
        means_pl: list[float] = []
        means_err: list[float] = []

        # 步骤2：该 sigma 下蒙卡：p_true = p_plan + noise，再 simulate_modes
        for k in range(int(n_monte_carlo)):
            rng = np.random.default_rng(seed + 10007 * k + int(round(sigma_f * 1000)))
            paired = execute_with_error(p_plan, error_std_m=sigma_f, rng=rng)
            result = simulate_modes(
                paired["p_plan"],
                paired["p_true"],
                planner_pose_error_m=paired["planner_pose_error_m"],
            )
            means_nr.append(float(np.mean(result["snr_no_ris"])))
            means_id.append(float(np.mean(result["snr_ideal_ris"])))
            means_pl.append(float(np.mean(result["snr_planner_ris"])))
            means_err.append(float(np.mean(result["planner_pose_error_m"])))

        # 步骤3：聚合为扫描曲线一点（横坐标用实测均值规划误差）
        row: SweepRow = {
            "error_std_m": sigma_f,
            "mean_planner_pose_error_m": float(np.mean(means_err)),
            "snr_no_ris_db": float(np.mean(means_nr)),
            "snr_ideal_ris_db": float(np.mean(means_id)),
            "snr_planner_ris_db": float(np.mean(means_pl)),
            "snr_planner_std_db": float(np.std(means_pl)),
            "n_monte_carlo": int(n_monte_carlo),
            "n_time_steps": int(n_time_steps),
        }
        rows.append(row)

    return {
        "rows": rows,
        "error_std_m": np.asarray([r["error_std_m"] for r in rows], dtype=float),
        "mean_planner_pose_error_m": np.asarray(
            [r["mean_planner_pose_error_m"] for r in rows], dtype=float
        ),
        "snr_no_ris_db": np.asarray([r["snr_no_ris_db"] for r in rows], dtype=float),
        "snr_ideal_ris_db": np.asarray(
            [r["snr_ideal_ris_db"] for r in rows], dtype=float
        ),
        "snr_planner_ris_db": np.asarray(
            [r["snr_planner_ris_db"] for r in rows], dtype=float
        ),
        "snr_planner_std_db": np.asarray(
            [r["snr_planner_std_db"] for r in rows], dtype=float
        ),
    }


def check_acceptance(sweep: SweepResult, *, tol_db: float = 0.05) -> dict[str, bool | str]:
    """Stage-2 acceptance: e->0 Planner~Ideal; larger e => Planner SNR drops.

    中文简释：验收：零误差时 Planner~Ideal；误差增大时 planner SNR 不升/总体下降。
    """
    err = sweep["mean_planner_pose_error_m"]
    snr_id = sweep["snr_ideal_ris_db"]
    snr_pl = sweep["snr_planner_ris_db"]

    # Zero-error index (prefer exact 0 std)
    i0 = int(np.argmin(np.abs(sweep["error_std_m"])))
    match_at_zero = abs(float(snr_pl[i0] - snr_id[i0])) <= tol_db

    # Planner should not increase with error (allow tiny MC noise)
    drops = True
    for i in range(1, len(snr_pl)):
        if float(snr_pl[i]) > float(snr_pl[i - 1]) + 0.5:
            drops = False
            break

    # Overall: last planner clearly below first (zero-error) planner
    overall_drop = float(snr_pl[-1]) < float(snr_pl[i0]) - 1.0

    ok = match_at_zero and drops and overall_drop
    return {
        "pass": ok,
        "match_at_zero": match_at_zero,
        "monotonic_nonincrease": drops,
        "overall_drop": overall_drop,
        "detail": (
            f"e0={err[i0]:.4f}m Ideal={snr_id[i0]:.2f} Planner={snr_pl[i0]:.2f}; "
            f"e_last={err[-1]:.4f}m Planner={snr_pl[-1]:.2f}"
        ),
    }
