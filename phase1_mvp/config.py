"""Central configuration: all magic numbers live here.

中文职责：Phase-1 全局常量集中处（射频、几何、轨迹、绘图标签）。
输入/输出：无函数；被 channel / geometry / simulation / plot 等直接读取。
在流程中的位置：整条 Phase-1 管线的参数源；改默认值会改变仿真行为（本文件只注释，不改数值）。
注意：T_HIST/T_PRED 对齐 phase2 接口尺寸；论文主线 thesis_planner_ris 另有自有 config。
"""

from __future__ import annotations

# --- Physical constants / RF ---
# 载波与波长：后续自由空间相位 / Friis 幅度都依赖 WAVELENGTH_M
SPEED_OF_LIGHT: float = 3.0e8  # m/s
CARRIER_FREQ_HZ: float = 5.0e9  # Hz
WAVELENGTH_M: float = SPEED_OF_LIGHT / CARRIER_FREQ_HZ

# --- Power ---
# 发射功率 Pt、噪声功率 σ²（dBm）；snr.dbm_to_watt 转为线性瓦特后进 SNR 公式
TX_POWER_DBM: float = 30.0
NOISE_POWER_DBM: float = -90.0

# --- RIS array ---
# X–Z 平面 URA：N = RIS_NUM_X * RIS_NUM_Z；阵元间距默认 λ/2
RIS_NUM_X: int = 8
RIS_NUM_Z: int = 8
RIS_ELEMENT_SPACING_M: float = WAVELENGTH_M / 2.0

# --- Geometry (meters, world frame) ---
# 静态场景：AP 与 RIS 面板中心（世界坐标，米）
POS_AP_M: tuple[float, float, float] = (0.0, 0.0, 10.0)
POS_RIS_CENTER_M: tuple[float, float, float] = (50.0, 0.0, 10.0)

# AP→robot direct path (Friis × α). α depends on BLOCK_DIRECT_LINK.
# 直连衰减：True 时 NLoS/遮挡（弱直连），突出 RIS 级联链路增益
BLOCK_DIRECT_LINK: bool = True  # True → NLoS/blocked (very weak direct)
DIRECT_PATH_ATTENUATION_BLOCKED: float = 1.0e-3  # α when blocked
DIRECT_PATH_ATTENUATION_OPEN: float = 1.0  # α when LoS open (full Friis)

# --- Phase-2 predictor interface sizes (aligned with phase2_traj.config) ---
# 历史窗 / 预测地平线长度；仅可选 Predicted 支路使用，非 thesis 主线
T_HIST: int = 20  # history window length for trajectory_predictor
T_PRED: int = 10  # future horizon length

# --- Demo trajectory (arc in front of RIS, z-plane) ---
# 演示圆弧：生成 p_true 序列；θ=0 时机器人在 RIS 正前方（+Y）
NUM_TIME_STEPS: int = 100
# Arc in the z = ROBOT_Z_M plane: center (TRAJ_ARC_CX, TRAJ_ARC_CY),
# radius TRAJ_ARC_RADIUS_M, angle from TRAJ_THETA_START to TRAJ_THETA_END.
# At θ=0 the robot sits at (CX, CY+R) ≈ directly in front of the RIS (+Y).
ROBOT_Z_M: float = 1.0
TRAJ_ARC_CX_M: float = 50.0
TRAJ_ARC_CY_M: float = 5.0
TRAJ_ARC_RADIUS_M: float = 28.0
TRAJ_THETA_START_RAD: float = -0.6
TRAJ_THETA_END_RAD: float = 0.6

# Legacy aliases (kept for older stubs / docs)
# 旧直线轨迹别名；自检/文档可能仍引用 ROBOT_Y_M
ROBOT_X_START_M: float = 10.0
ROBOT_X_END_M: float = 90.0
ROBOT_Y_M: float = 30.0

# --- Simulation / plotting ---
# POS_NOISE_STD_M：设计位姿噪声 σ，对应 p_noisy = p_true + N(0, σ² I)
RANDOM_SEED: int = 42
POS_NOISE_STD_M: float = 0.3  # default σ for p_noisy = p_true + N(0, σ² I)
FIGURE_SIZE: tuple[float, float] = (8.0, 4.5)
SNR_PLOT_TITLE: str = "Phase-1 MVP: SNR vs Time"
SNR_PLOT_XLABEL: str = "Time step"
SNR_PLOT_YLABEL: str = "SNR (dB)"
# Relative to repo package dir; run_mvp resolves to an absolute path.
SNR_PLOT_SAVE_PATH: str = "outputs/snr_comparison.png"
SNR_PLOT_SAVE_PATH_PHASE2: str = "outputs/snr_comparison_phase2.png"
SNR_PLOT_TITLE_PHASE2: str = "Phase-1+2: SNR vs Time (with Predicted-RIS)"
PLOT_LABEL_NO_RIS: str = "No RIS"
PLOT_LABEL_IDEAL: str = "Ideal RIS"
PLOT_LABEL_NOISY: str = "Noisy-position RIS"
PLOT_LABEL_PREDICTED: str = "Predicted-RIS"
