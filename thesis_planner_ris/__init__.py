"""Thesis main line: planner-assisted / location-aware RIS control.

Default pipeline uses a motion-planner pose p_plan to design RIS phases
and evaluates SNR on the executed pose p_true. No ML / torch / phase2.

中文职责：论文主线包：规划器辅助 / 位置感知 RIS。
默认流程：p_plan 设计相位 phi，在 p_true 上评估 SNR。
物理层复用 phase1_mvp；主线无 ML / torch / phase2。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
