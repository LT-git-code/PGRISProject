"""Paper-style figures for Stage-2 (and later stages).

中文职责：Stage-2 论文风格图：均值 SNR vs 均值规划位姿误差。
输入：SweepResult；输出：PNG 路径。
在流程中的位置：扫描之后可视化；横轴为 ||p_true-p_plan||，非 ADE/FDE。
注意：图例用 no_ris / ideal_ris / planner_ris（勿写成 Predicted）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from thesis_planner_ris import config
from thesis_planner_ris.experiments.error_sweep import SweepResult


def plot_snr_vs_planner_error(
    sweep: SweepResult,
    out_path: str | Path,
) -> Path:
    """Fig: mean SNR vs mean planner_pose_error_m for three modes.

    中文简释：绘制三模式均值 SNR；planner 带蒙卡标准差误差棒。
    要点：标题已写明 design phi from p_plan, eval on p_true。
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 步骤1：横轴 = 实测均值规划误差（非执行噪声 sigma 本身）
    x = sweep["mean_planner_pose_error_m"]
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    # 步骤2：三模式曲线；planner 加 errorbar
    ax.plot(
        x,
        sweep["snr_no_ris_db"],
        "o-",
        label=config.PLOT_LABEL_NO_RIS,
        lw=1.5,
    )
    ax.plot(
        x,
        sweep["snr_ideal_ris_db"],
        "s-",
        label=config.PLOT_LABEL_IDEAL_RIS,
        lw=1.5,
    )
    ax.errorbar(
        x,
        sweep["snr_planner_ris_db"],
        yerr=sweep["snr_planner_std_db"],
        fmt="^-",
        label=config.PLOT_LABEL_PLANNER_RIS,
        lw=1.5,
        capsize=3,
    )
    ax.set_xlabel("mean planner_pose_error_m  ||p_true - p_plan|| [m]")
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title(
        "Stage-2: SNR vs planner pose error "
        "(design phi from p_plan, eval on p_true)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
