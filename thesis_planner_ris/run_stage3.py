"""Stage 3 entry (optional robust) - skipped in this thesis main line.

Supervisor main method is planner-assisted RIS (Stages 1/2/4). Robust
optimization over a planner-uncertainty ball R is documented as future work;
phase3_robust remains available as legacy code but is not required.

中文职责：Stage-3 可选/鲁棒入口（本论文主线跳过）。
说明：导师主线为规划器辅助 RIS（Stage 1/2/4）；
基于规划不确定性球半径 R 的鲁棒优化仅作为未来工作，不写输出。
注意：不导入 phase3_robust；跑本文件仅打印跳过说明。
"""

from __future__ import annotations


def main() -> None:
    """中文简释：打印 Stage-3 已跳过；不产生任何文件。"""
    print("DONE Stage 3 (skipped)")
    print("  Robust RIS wrapper was skipped by design.")
    print("  Main line: planner-assisted RIS (Stages 1, 2, 4).")
    print("  Future work: wrap phase3_robust with planner uncertainty R")
    print("  (see thesis_planner_ris/README.md). No outputs written.")


if __name__ == "__main__":
    main()
