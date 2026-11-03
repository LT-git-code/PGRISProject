"""Stage 2: planner pose-error sweep + CSV + paper figure.

Run from repo root::

    python -m thesis_planner_ris.run_stage2

Saves::

    thesis_planner_ris/outputs/snr_vs_planner_error.csv
    thesis_planner_ris/outputs/fig_snr_vs_planner_error.png
    thesis_planner_ris/outputs/stage2_summary.txt

中文职责：Stage-2 入口——规划执行误差扫描 + CSV/论文图。
输入：config 中的 error_std 列表与蒙卡次数；
输出：snr_vs_planner_error.csv、fig、stage2_summary.txt。
在流程中的位置：调用 experiments.error_sweep / export_csv / paper_plots。
注意：仍是 p_plan 配相、p_true 评估；无 ML / phase2 / .pth。
"""

from __future__ import annotations

from pathlib import Path

from thesis_planner_ris import config
from thesis_planner_ris.experiments.error_sweep import (
    check_acceptance,
    run_error_sweep,
)
from thesis_planner_ris.experiments.export_csv import export_sweep_csv
from thesis_planner_ris.experiments.paper_plots import plot_snr_vs_planner_error


def _ensure_outputs() -> Path:
    """中文简释：确保 outputs 目录存在。"""
    out = config.OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_summary(
    sweep,
    accept: dict,
    out_path: Path,
) -> None:
    """中文简释：把扫描表与验收结果写入 stage2_summary.txt。"""
    lines = [
        "thesis_planner_ris Stage-2 summary",
        "=================================",
        "Sweep execution noise std -> mean planner_pose_error_m.",
        "Design phi from p_plan; evaluate SNR always on p_true.",
        "No ML / phase2_traj / .pth in this path.",
        f"error_stds_m = {list(config.ERROR_SWEEP_STD_M)}",
        f"n_monte_carlo = {config.ERROR_SWEEP_N_MONTE_CARLO}",
        f"n_time_steps  = {config.ERROR_SWEEP_NUM_TIME_STEPS}",
        "",
        "error_std_m | mean_err_m | no_ris | ideal | planner",
        "-----------------------------------------------------",
    ]
    for row in sweep["rows"]:
        lines.append(
            f"{row['error_std_m']:10.3f} | "
            f"{row['mean_planner_pose_error_m']:10.4f} | "
            f"{row['snr_no_ris_db']:6.2f} | "
            f"{row['snr_ideal_ris_db']:5.2f} | "
            f"{row['snr_planner_ris_db']:7.2f}"
        )
    lines.extend(
        [
            "",
            f"acceptance detail: {accept['detail']}",
            f"match_at_zero (Planner~Ideal): {accept['match_at_zero']}",
            f"monotonic_nonincrease:         {accept['monotonic_nonincrease']}",
            f"overall_drop:                  {accept['overall_drop']}",
            f"acceptance PASS:               {accept['pass']}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """中文简释：跑扫描 → 验收 → 导出 CSV/图/摘要。"""
    out = _ensure_outputs()
    print("Stage 2: running planner pose-error sweep ...")
    # 步骤1：按执行噪声 sigma 扫描，得到均值 SNR vs 规划误差
    sweep = run_error_sweep()
    accept = check_acceptance(sweep)

    # 步骤2：落盘 CSV、论文图、summary
    csv_path = export_sweep_csv(sweep, out / config.STAGE2_CSV)
    fig_path = plot_snr_vs_planner_error(sweep, out / config.STAGE2_FIG)
    summary_path = out / config.STAGE2_SUMMARY_TXT
    _write_summary(sweep, accept, summary_path)

    print("DONE Stage 2")
    print(f"  {csv_path}")
    print(f"  {fig_path}")
    print(f"  {summary_path}")
    print(f"  acceptance: {'PASS' if accept['pass'] else 'FAIL'} ({accept['detail']})")


if __name__ == "__main__":
    main()
