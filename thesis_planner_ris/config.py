"""Local constants for the planner-assisted RIS main line.

RF / geometry defaults mirror phase1_mvp.config so physics imports stay
aligned. Naming here uses p_plan / p_true (not prediction vocabulary).

中文职责：论文主线（规划器辅助 RIS）的本地常量与输出路径。
输入/输出：无函数；射频/几何多数从 phase1_mvp.config 再导出，保证物理一致。
在流程中的位置：stage1/2/demo/experiments 的参数源。
注意：术语是 p_plan / p_true（运动规划），不是 AI 预测；主线无 ML。
"""

from __future__ import annotations

from pathlib import Path

from phase1_mvp import config as p1

# --- Package paths ---
PKG_DIR: Path = Path(__file__).resolve().parent
OUTPUTS_DIR: Path = PKG_DIR / "outputs"

# --- Re-export phase-1 RF / geometry (single source of truth) ---
# 物理层参数复用自 phase1_mvp，本文件只做主线业务侧配置封装
SPEED_OF_LIGHT: float = p1.SPEED_OF_LIGHT
CARRIER_FREQ_HZ: float = p1.CARRIER_FREQ_HZ
WAVELENGTH_M: float = p1.WAVELENGTH_M
TX_POWER_DBM: float = p1.TX_POWER_DBM
NOISE_POWER_DBM: float = p1.NOISE_POWER_DBM
RIS_NUM_X: int = p1.RIS_NUM_X
RIS_NUM_Z: int = p1.RIS_NUM_Z
RIS_ELEMENT_SPACING_M: float = p1.RIS_ELEMENT_SPACING_M
POS_AP_M: tuple[float, float, float] = p1.POS_AP_M
POS_RIS_CENTER_M: tuple[float, float, float] = p1.POS_RIS_CENTER_M
BLOCK_DIRECT_LINK: bool = p1.BLOCK_DIRECT_LINK

# --- Trajectory (synthetic stand-in for a motion planner) ---
# 合成规划轨迹几何（与 phase1 演示圆弧同族），当作已知的 p_plan
NUM_TIME_STEPS: int = p1.NUM_TIME_STEPS
ROBOT_Z_M: float = p1.ROBOT_Z_M
TRAJ_ARC_CX_M: float = p1.TRAJ_ARC_CX_M
TRAJ_ARC_CY_M: float = p1.TRAJ_ARC_CY_M
TRAJ_ARC_RADIUS_M: float = p1.TRAJ_ARC_RADIUS_M
TRAJ_THETA_START_RAD: float = p1.TRAJ_THETA_START_RAD
TRAJ_THETA_END_RAD: float = p1.TRAJ_THETA_END_RAD

# --- Planner pose error (execution vs plan) ---
# 执行误差：p_true = p_plan + 噪声；Stage-1 默认 sigma=0（完美跟踪）
RANDOM_SEED: int = 42
# Stage-1 baseline uses zero error (p_true = p_plan).
PLANNER_POSE_ERROR_STD_M: float = 0.0
# Stage-2 error sweep: execution noise std [m] -> planner_pose_error_m
ERROR_SWEEP_STD_M: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0, 2.0)
ERROR_SWEEP_N_MONTE_CARLO: int = 8
# Shorter traj for sweep speed; physics unchanged (same arc family).
ERROR_SWEEP_NUM_TIME_STEPS: int = 40

# --- Optional robust (planner uncertainty ball radius R) ---
# 可选鲁棒配相的不确定性球半径 R（Stage-3 / robust，非主线必跑）
ROBUST_RADIUS_M: float = 0.5

# --- Plot / export labels (glossary) ---
# 图例口径：no_ris / ideal_ris / planner_ris（勿写成 Predicted）
PLOT_LABEL_NO_RIS: str = "no_ris"
PLOT_LABEL_IDEAL_RIS: str = "ideal_ris"
PLOT_LABEL_PLANNER_RIS: str = "planner_ris"
FIGURE_SIZE: tuple[float, float] = (8.0, 4.5)

# --- Stage-1 outputs ---
STAGE1_BASELINE_PLOT: str = "snr_planner_baseline.png"
STAGE1_SUMMARY_TXT: str = "stage1_summary.txt"

# --- Stage-2 outputs ---
STAGE2_CSV: str = "snr_vs_planner_error.csv"
STAGE2_FIG: str = "fig_snr_vs_planner_error.png"
STAGE2_SUMMARY_TXT: str = "stage2_summary.txt"

# --- Stage-4 demo (planner labels; no ML) ---
# Demo：故意用非零误差，便于可视化区分 p_plan 与 p_true
DEMO_NUM_TIME_STEPS: int = 120
DEMO_ERROR_STD_M: float = 0.5  # nonzero so p_plan / p_true are visually distinct
DEMO_FPS: int = 8
DEMO_DPI: int = 100
DEMO_MAX_FRAMES: int = 40
DEMO_FIGURE_SIZE: tuple[float, float] = (11.0, 5.0)
DEMO_GIF: str = "demo_planner_ris.gif"
DEMO_PREVIEW: str = "demo_preview.png"
DEMO_BEAM_RADIUS_FRAC: float = 0.08  # illustrative cone (not EM pattern)
