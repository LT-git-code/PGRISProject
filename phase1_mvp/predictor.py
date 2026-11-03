"""Phase-2 trajectory predictor interface (draft; no model yet).

中文职责：旧版 / 占位的轨迹预测接口（本文件始终 NotImplemented）。
输入：历史窗 history (T_hist, 3)；契约输出 future (T_pred, 3)。
在流程中的位置：早期为 phase2 预留的签名桩；Phase-1 主仿真实际懒加载
phase2_traj.predict，并不调用本文件。
注意：论文主线 thesis_planner_ris 用运动规划器 p_plan，无 ML 预测；
请勿把本接口当成当前主方案。
"""

from __future__ import annotations

import numpy as np

from phase1_mvp import config


def trajectory_predictor(history: np.ndarray) -> np.ndarray:
    """Predict future robot positions from a recent history window.

    Phase-2 interface (not implemented — do not call from phase-1 MVP).

    Contract
    --------
    - **Input** ``history``: shape ``(T_hist, 3)``, meters, oldest → newest.
      Default ``T_hist = config.T_HIST``.
    - **Output** ``future``: shape ``(T_pred, 3)``, meters, next ``T_pred``
      steps after the last history sample. Default ``T_pred = config.T_PRED``.

    Integration with phase-1 simulation
    -----------------------------------
    In ``simulation.simulate_snr_curves``, the design position

        ``p_design = p_true + noise``   (phase-1 “Noisy-position RIS”)

    will be replaced by a predicted coordinate, e.g.

        ``p_design = trajectory_predictor(history)[k]``

    while **evaluation** must still use the true pose ``p_true``:

    - ``h_eff_design = cascaded_channel(..., p_design)`` → design ``φ``
    - ``h_eff_true   = cascaded_channel(..., p_true)``   → compute SNR

    LSTM / other sequence models belong here in phase-2; keep this signature.

    中文简释：占位函数——校验 history 形状后直接 NotImplementedError。
    要点：契约上设计位姿与评估位姿必须分离（design vs evaluation）。
    注意：主线请读 thesis_planner_ris（p_plan / p_true），不要依赖本桩。
    """
    # 步骤1：校验输入形状（即便未实现也保持接口契约）
    hist = np.asarray(history, dtype=float)
    if hist.ndim != 2 or hist.shape[1] != 3:
        raise ValueError("history must have shape (T_hist, 3)")
    if hist.shape[0] < 1:
        raise ValueError("history must contain at least one sample")

    # 步骤2：占位——正式实现应在 phase2；本文件永不返回轨迹
    _ = config.T_PRED  # horizon reserved for phase-2 implementations
    raise NotImplementedError(
        "Phase-2 only: implement trajectory_predictor; "
        "phase-1 uses Gaussian p_design as a stand-in for predicted poses."
    )
