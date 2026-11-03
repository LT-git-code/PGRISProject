"""Stage 1: baseline planner-informed RIS with zero pose error.

Baseline: p_true = p_plan  =>  ideal_ris ~= planner_ris >> no_ris

Run from repo root::

    python -m thesis_planner_ris.run_stage1

Saves::

    thesis_planner_ris/outputs/snr_planner_baseline.png
    thesis_planner_ris/outputs/stage1_summary.txt

中文职责：Stage-1 基线——零执行误差下验证 planner_ris 与 ideal 重合。
输入：默认轨迹，error_std=0（p_true = p_plan）；
输出：SNR 对比图 + stage1_summary.txt。
在流程中的位置：planner_traj → modes.simulate_modes → 出图/验收。
注意：无 ML；配相用 p_plan，评估用 p_true（本阶段二者相等）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thesis_planner_ris import config
from thesis_planner_ris.modes import simulate_modes
from thesis_planner_ris.planner_traj import make_paired_trajectories


def _ensure_outputs() -> Path:
    """中文简释：确保 outputs 目录存在。"""
    out = config.OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_baseline(result: dict, out_path: Path) -> None:
    """中文简释：绘制 no_ris / ideal_ris / planner_ris 三条 SNR 曲线。"""
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    t = result["time"]
    ax.plot(t, result["snr_no_ris"], label=config.PLOT_LABEL_NO_RIS, lw=1.5)
    ax.plot(t, result["snr_ideal_ris"], label=config.PLOT_LABEL_IDEAL_RIS, lw=1.5)
    ax.plot(t, result["snr_planner_ris"], label=config.PLOT_LABEL_PLANNER_RIS, lw=1.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(
        "Stage-1 baseline: phi from p_plan, SNR on p_true "
        "(p_true = p_plan, zero error)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_summary(result: dict, out_path: Path) -> dict[str, float]:
    """中文简释：写入均值/误差/验收结果到 summary 文本。"""
    mean_nr = float(np.mean(result["snr_no_ris"]))
    mean_id = float(np.mean(result["snr_ideal_ris"]))
    mean_pl = float(np.mean(result["snr_planner_ris"]))
    max_abs_diff_id_pl = float(
        np.max(np.abs(result["snr_ideal_ris"] - result["snr_planner_ris"]))
    )
    mean_err = float(np.mean(result["planner_pose_error_m"]))

    accept_match = max_abs_diff_id_pl < 1e-6
    accept_gain = (mean_id - mean_nr) > 10.0 and (mean_pl - mean_nr) > 10.0
    accept = accept_match and accept_gain and mean_err == 0.0

    lines = [
        "thesis_planner_ris Stage-1 summary",
        "=================================",
        "Baseline: p_true = p_plan (zero planner pose error).",
        "Design phi from p_plan; evaluate SNR on p_true.",
        "Synthetic planned traj stands in for a motion planner (no ML).",
        f"mean planner_pose_error_m     = {mean_err:.6e}",
        f"mean snr_no_ris      [dB]      = {mean_nr:.3f}",
        f"mean snr_ideal_ris   [dB]      = {mean_id:.3f}",
        f"mean snr_planner_ris [dB]      = {mean_pl:.3f}",
        f"max |ideal - planner| [dB]    = {max_abs_diff_id_pl:.6e}",
        f"acceptance Ideal~Planner>>NoRIS = {accept}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "mean_no_ris": mean_nr,
        "mean_ideal": mean_id,
        "mean_planner": mean_pl,
        "max_abs_diff_id_pl": max_abs_diff_id_pl,
        "mean_err": mean_err,
        "accept": float(accept),
    }


def main() -> None:
    """中文简释：Stage-1 入口：零误差轨迹 → 三模式 SNR → 出图与验收。"""
    out = _ensure_outputs()

    # 步骤1：零执行误差：p_true = p_plan
    paired = make_paired_trajectories(error_std_m=0.0)
    assert np.allclose(paired["p_plan"], paired["p_true"])
    assert np.allclose(paired["planner_pose_error_m"], 0.0)

    # 步骤2：三模式 SNR（本阶段 ideal 与 planner 应重合）
    result = simulate_modes(
        paired["p_plan"],
        paired["p_true"],
        planner_pose_error_m=paired["planner_pose_error_m"],
    )

    # 步骤3：出图 + summary，打印验收
    plot_path = out / config.STAGE1_BASELINE_PLOT
    summary_path = out / config.STAGE1_SUMMARY_TXT
    _plot_baseline(result, plot_path)
    stats = _write_summary(result, summary_path)

    print("DONE Stage 1")
    print(f"  {plot_path}")
    print(f"  {summary_path}")
    print(
        f"  mean SNR [dB]: no_ris={stats['mean_no_ris']:.2f}, "
        f"ideal_ris={stats['mean_ideal']:.2f}, "
        f"planner_ris={stats['mean_planner']:.2f}"
    )
    print(
        f"  acceptance Ideal~Planner>>NoRIS: "
        f"{'PASS' if stats['accept'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
