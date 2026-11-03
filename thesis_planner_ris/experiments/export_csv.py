"""Export Stage-2 sweep tables to CSV (numpy/matplotlib stack only).

中文职责：将 Stage-2 扫描结果导出为 CSV（供论文/后处理）。
输入：SweepResult；输出：utf-8 CSV 文件路径。
在流程中的位置：run_stage2 在扫描之后调用；不参与配相/信道。
"""

from __future__ import annotations

import csv
from pathlib import Path

from thesis_planner_ris.experiments.error_sweep import SweepResult


CSV_COLUMNS = [
    "error_std_m",
    "mean_planner_pose_error_m",
    "snr_no_ris_db",
    "snr_ideal_ris_db",
    "snr_planner_ris_db",
    "snr_planner_std_db",
    "n_monte_carlo",
    "n_time_steps",
]


def export_sweep_csv(sweep: SweepResult, path: str | Path) -> Path:
    """Write snr_vs_planner_error.csv from a SweepResult.

    中文简释：按固定列名写入扫描行；列含规划误差与三模式 SNR。
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 步骤1：写表头；步骤2：逐行写入 sweep['rows']
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in sweep["rows"]:
            writer.writerow({k: row[k] for k in CSV_COLUMNS})  # type: ignore[literal-required]

    return out
