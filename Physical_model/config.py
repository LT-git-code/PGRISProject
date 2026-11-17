"""Unified configuration: physics + experiment constants.

中文职责：物理模型与实验共用常量；输出统一到仓库根目录 output/。
射频/几何为物理真源；Stage/Demo 等实验旋钮也集中于此，避免双包分叉。
注意：术语主线是 p_plan / p_true（运动规划），不是 AI 预测。
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---
PKG_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PKG_DIR.parent
OUTPUTS_DIR: Path = REPO_ROOT / "output"

# --- Physical constants / RF ---
SPEED_OF_LIGHT: float = 3.0e8  # m/s
CARRIER_FREQ_HZ: float = 5.0e9  # Hz
WAVELENGTH_M: float = SPEED_OF_LIGHT / CARRIER_FREQ_HZ

# --- Power ---
TX_POWER_DBM: float = 30.0
NOISE_POWER_DBM: float = -90.0

# --- RIS array ---
RIS_NUM_X: int = 8
RIS_NUM_Z: int = 8
RIS_ELEMENT_SPACING_M: float = WAVELENGTH_M / 2.0

# --- Geometry (meters, world frame) ---
POS_AP_M: tuple[float, float, float] = (0.0, 0.0, 10.0)
POS_RIS_CENTER_M: tuple[float, float, float] = (50.0, 0.0, 13.5)

# AP→robot direct path (Friis × α). α depends on BLOCK_DIRECT_LINK.
BLOCK_DIRECT_LINK: bool = True  # True → NLoS/blocked (very weak direct)
DIRECT_PATH_ATTENUATION_BLOCKED: float = 1.0e-3
DIRECT_PATH_ATTENUATION_OPEN: float = 1.0

# --- Phase-2 predictor interface sizes (legacy optional path) ---
T_HIST: int = 20
T_PRED: int = 10

# --- Trajectory source ---
# "arc" = legacy synthetic arc; "grid_planner" = 2D occupancy A* + resample
TRAJ_SOURCE: str = "grid_planner"

# --- Trajectory (synthetic arc stand-in; used when TRAJ_SOURCE == "arc") ---
NUM_TIME_STEPS: int = 100
ROBOT_Z_M: float = 1.0
TRAJ_ARC_CX_M: float = 50.0
TRAJ_ARC_CY_M: float = 5.0
TRAJ_ARC_RADIUS_M: float = 35.0
TRAJ_THETA_START_RAD: float = -1.0
TRAJ_THETA_END_RAD: float = 1.0

# --- Grid motion planner (TRAJ_SOURCE == "grid_planner") ---
# World-axis-aligned map covering AP / RIS / robot workspace (top-down x–y).
GRID_X_MIN_M: float = 0.0
GRID_X_MAX_M: float = 100.0
GRID_Y_MIN_M: float = -5.0
GRID_Y_MAX_M: float = 45.0
GRID_RESOLUTION_M: float = 0.5
PLANNER_ROBOT_RADIUS_M: float = 0.8
PLAN_START_XY_M: tuple[float, float] = (15.0, 34.0)
PLAN_GOAL_XY_M: tuple[float, float] = (85.0, 34.0)
# Axis-aligned boxes: (xmin, xmax, ymin, ymax) in meters.
# Wall A: between AP and workspace (NLoS-ready for direct AP→robot).
# Wall B: blocks the straight high-y corridor so A* detours toward the RIS.
OBSTACLE_BOXES_M: tuple[tuple[float, float, float, float], ...] = (
    (20.0, 28.0, -5.0, 22.0),
    (45.0, 55.0, 26.0, 42.0),
)
# Reserved for later spatial NLoS on the direct link (unused by planner).
NLOS_USE_SPATIAL_VISIBILITY: bool = False

# Legacy aliases
ROBOT_X_START_M: float = 10.0
ROBOT_X_END_M: float = 90.0
ROBOT_Y_M: float = 30.0

# --- Simulation / plotting (phase-1 MVP legacy) ---
RANDOM_SEED: int = 42
POS_NOISE_STD_M: float = 0.3
FIGURE_SIZE: tuple[float, float] = (8.0, 4.5)
SNR_PLOT_TITLE: str = "Phase-1 MVP: SNR vs Time"
SNR_PLOT_XLABEL: str = "Time step"
SNR_PLOT_YLABEL: str = "SNR (dB)"
# Filenames under OUTPUTS_DIR (repo output/)
SNR_PLOT_SAVE_PATH: str = "snr_comparison.png"
SNR_PLOT_SAVE_PATH_PHASE2: str = "snr_comparison_phase2.png"
SNR_PLOT_TITLE_PHASE2: str = "Phase-1+2: SNR vs Time (with Predicted-RIS)"
# Phase-1 MVP plot labels (run_mvp / plot_snr)
PLOT_LABEL_NO_RIS_MVP: str = "No RIS"
PLOT_LABEL_IDEAL: str = "Ideal RIS"
PLOT_LABEL_NOISY: str = "Noisy-position RIS"
PLOT_LABEL_PREDICTED: str = "Predicted-RIS"

# Planner main-line glossary (Stage 1/2/demo)
PLOT_LABEL_NO_RIS: str = "no_ris"

# --- Planner pose error (execution vs plan) ---
PLANNER_POSE_ERROR_STD_M: float = 0.0
ERROR_SWEEP_STD_M: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 1.0, 2.0)
ERROR_SWEEP_N_MONTE_CARLO: int = 8
ERROR_SWEEP_NUM_TIME_STEPS: int = 40

# --- Optional robust ---
ROBUST_RADIUS_M: float = 0.5

# --- Plot / export labels (planner main line glossary) ---
PLOT_LABEL_IDEAL_RIS: str = "ideal_ris"
PLOT_LABEL_PLANNER_RIS: str = "planner_ris"

# --- Stage-1 outputs ---
STAGE1_BASELINE_PLOT: str = "snr_planner_baseline.png"
STAGE1_SUMMARY_TXT: str = "stage1_summary.txt"

# --- Stage-2 outputs ---
STAGE2_CSV: str = "snr_vs_planner_error.csv"
STAGE2_FIG: str = "fig_snr_vs_planner_error.png"
STAGE2_SUMMARY_TXT: str = "stage2_summary.txt"

# --- Stage-4 demo ---
DEMO_NUM_TIME_STEPS: int = 120
DEMO_ERROR_STD_M: float = 0.5
DEMO_FPS: int = 8
DEMO_DPI: int = 100
DEMO_MAX_FRAMES: int = 40
DEMO_FIGURE_SIZE: tuple[float, float] = (11.0, 5.0)
DEMO_GIF: str = "demo_planner_ris.gif"
DEMO_PREVIEW: str = "demo_preview.png"
DEMO_BEAM_RADIUS_FRAC: float = 0.08
