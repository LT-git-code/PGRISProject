"""End-to-end phase-1 simulation: demo trajectory and SNR curves.

中文职责：Phase-1 端到端仿真——演示轨迹 + 多条 SNR 曲线。
输入：轨迹（或内部生成）、位姿噪声 σ、可选预测/鲁棒开关；
输出：time 与 snr_no_ris / snr_ideal / snr_noisy（及可选 predicted / robust）。
在流程中的位置：串起 geometry→channel→beamforming→snr；是 Phase-1 主实验循环。
物理层复用自本包 channel / beamforming / snr；本文件做业务编排与 design/eval 分离。
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from phase1_mvp import config
from phase1_mvp.beamforming import deterministic_phase
from phase1_mvp.channel import cascaded_channel, direct_ap_robot_channel
from phase1_mvp.geometry import get_ap_position, get_ris_element_positions
from phase1_mvp.snr import compute_snr, dbm_to_watt


class SnrCurves(TypedDict, total=False):
    """SNR series; cores always set, extras only when flags enabled.

    中文简释：SNR 时间序列字典；核心三条必有，predicted/robust 按开关可选。
    """

    time: np.ndarray
    snr_no_ris: np.ndarray
    snr_ideal: np.ndarray
    snr_noisy: np.ndarray
    snr_predicted: np.ndarray
    snr_robust: np.ndarray  # optional; requires use_robust=True


def generate_demo_trajectory(T: int) -> np.ndarray:
    """Generate a slow planar arc in front of the RIS coverage region.

    The robot stays on the plane ``z = config.ROBOT_Z_M`` and follows a
    circular arc centered at ``(TRAJ_ARC_CX_M, TRAJ_ARC_CY_M)`` with radius
    ``TRAJ_ARC_RADIUS_M``, sweeping ``θ ∈ [TRAJ_THETA_START, TRAJ_THETA_END]``.
    At ``θ = 0`` the robot is directly in front of the RIS along +Y.

    中文简释：生成 RIS 前方缓慢平面圆弧轨迹（演示用真实位姿序列 p_true）。
    要点：z 固定；θ=0 时正对 RIS（+Y）。

    Parameters
    ----------
    T :
        Number of time steps (typically ``config.NUM_TIME_STEPS``).

    Returns
    -------
    np.ndarray
        Shape ``(T, 3)``, positions in meters.
    """
    if T < 1:
        raise ValueError("T must be >= 1")

    # 步骤1：角度采样；步骤2：圆弧参数方程 → (x,y,z)
    theta = np.linspace(
        config.TRAJ_THETA_START_RAD,
        config.TRAJ_THETA_END_RAD,
        T,
    )
    cx = config.TRAJ_ARC_CX_M
    cy = config.TRAJ_ARC_CY_M
    radius = config.TRAJ_ARC_RADIUS_M
    z = config.ROBOT_Z_M

    x = cx + radius * np.sin(theta)
    y = cy + radius * np.cos(theta)
    traj = np.column_stack([x, y, np.full(T, z, dtype=float)])
    return traj


def _history_window(traj: np.ndarray, t: int, t_hist: int) -> np.ndarray:
    """Causal history ending just before time ``t`` (left-pad if needed).

    Returns ``traj[t - T_hist : t]`` with shape ``(T_hist, 3)``.
    If ``t < T_hist``, repeats ``traj[0]`` on the left so the window is full.

    中文简释：取时刻 t 之前的因果历史窗（可选预测支路用）；不足则左端重复首帧。
    """
    if t <= 0:
        return np.repeat(traj[0:1], t_hist, axis=0)

    start = t - t_hist
    if start >= 0:
        return traj[start:t]

    # t < T_hist：左侧用首帧补齐窗口长度
    available = traj[0:t]
    n_pad = t_hist - available.shape[0]
    pad = np.repeat(traj[0:1], n_pad, axis=0)
    return np.vstack([pad, available])


def simulate_snr_curves(
    traj: np.ndarray,
    pos_noise_std: float = 0.3,
    *,
    block_direct_link: bool | None = None,
    use_predictor: bool = False,
    predictor_checkpoint: str | None = None,
    use_robust: bool = False,
    robust_radius_m: float | None = None,
    rng: np.random.Generator | None = None,
) -> SnrCurves:
    """Simulate no-RIS / ideal-RIS / noisy-phase (+ optional Predicted / Robust).

    Phase-design vs evaluation channels (phase-1 key point)
    -------------------------------------------------------
    At each ``t``:

    - ``p_true = traj[t]`` — true robot position (**evaluation** geometry).
    - Noisy design pose (always computed; unchanged vs phase-1)::

          p_noisy = p_true + N(0, σ² I)

    - Optional Predicted-RIS design pose (``use_predictor=True``)::

          history = traj[t - T_hist : t]   # poses at t-T_hist … t-1 (left-padded)
          future  = trajectory_predictor(history)   # (T_pred, 3)
          p_pred  = future[0]               # predicts the pose **at time t**

      Time alignment (matches phase-2 training windows): history ends at
      ``t-1``, so ``future[0]`` is the model's estimate of ``traj[t]`` — the
      instant being served. Evaluation **always** uses ``h_eff(p_true)``.

    - Optional Robust-RIS (``use_robust=True``, **default off**): designs
      ``φ`` via ``phase3_robust.integration.configure_ris("robust", ...)``
      on ``U(p_hat, R)`` with ``p_hat = p_pred`` if predictor is on, else
      ``p_noisy``. SNR still on ``h_eff(p_true)``.

    - ``h_eff_true`` from ``p_true`` — **evaluation** channel.
    - Design channels from ``p_noisy`` / ``p_pred`` — **phase design only**.

    Curves
    ------
    1. ``snr_no_ris``: direct link only.
    2. ``snr_ideal``: ``φ`` from ``h_eff_true``, SNR on ``h_eff_true``.
    3. ``snr_noisy``: ``φ`` from noisy design pose, SNR on ``h_eff_true``.
    4. ``snr_predicted`` (if ``use_predictor``): ``φ`` from ``p_pred``, SNR on
       ``h_eff_true``.
    5. ``snr_robust`` (if ``use_robust``): robust ``φ``, SNR on ``h_eff_true``.

    When ``use_predictor=False`` and ``use_robust=False``, outputs and numerics
    for (1)–(3) match the original phase-1 behaviour.

    中文简释：逐时刻仿真多条 SNR；核心是 design 配相 vs evaluation 评估分离。
    要点（与 thesis 口径对应）：
      - p_true = traj[t]：真实位姿 → 只用于评估信道 h_eff_true / SNR
      - p_noisy（及可选 p_pred）：设计位姿 → 只用于求 φ
      - 理想曲线：design=eval=p_true；噪声曲线：用 p_noisy 配相、p_true 评估
    注意：可选 use_predictor / use_robust 依赖 phase2/phase3，默认关闭以保持
    Phase-1 行为；论文主线 thesis_planner_ris 不以本处 ML 预测为准。
    """
    traj_arr = np.asarray(traj, dtype=float)
    if traj_arr.ndim != 2 or traj_arr.shape[1] != 3:
        raise ValueError("traj must have shape (T, 3)")
    if pos_noise_std < 0.0:
        raise ValueError("pos_noise_std must be >= 0")
    if block_direct_link is None:
        block_direct_link = config.BLOCK_DIRECT_LINK

    t_steps = traj_arr.shape[0]
    if rng is None:
        rng = np.random.default_rng(config.RANDOM_SEED)

    predict_fn = None
    if use_predictor:
        # 懒加载：仅 Phase-1 运行时可不装 phase2；非 thesis 主线
        from phase2_traj.predict import trajectory_predictor as predict_fn

    configure_ris_fn = None
    robust_R = 0.0
    if use_robust:
        # 懒加载 phase3 鲁棒配相；默认关闭
        from phase3_robust.integration import configure_ris as configure_ris_fn
        from phase3_robust.simulation import resolve_R_design

        robust_R = float(resolve_R_design(robust_radius_m))

    # 固定几何与功率：AP、RIS 阵元、λ、Pt、σ²
    ap_pos = get_ap_position()
    ris_pos = get_ris_element_positions()
    lambda_ = config.WAVELENGTH_M
    pt = dbm_to_watt(config.TX_POWER_DBM)
    sigma2 = dbm_to_watt(config.NOISE_POWER_DBM)
    t_hist = config.T_HIST

    time = np.arange(t_steps, dtype=float)
    snr_no_ris = np.empty(t_steps, dtype=float)
    snr_ideal = np.empty(t_steps, dtype=float)
    snr_noisy = np.empty(t_steps, dtype=float)
    snr_predicted = np.empty(t_steps, dtype=float) if use_predictor else None
    snr_robust = np.empty(t_steps, dtype=float) if use_robust else None

    dummy_phi: np.ndarray | None = None

    for t in range(t_steps):
        # 步骤1：真实评估位姿 p_true；设计用噪声位姿 p_noisy
        p_true = traj_arr[t]
        p_noisy = p_true + rng.normal(0.0, pos_noise_std, size=3)

        # 步骤2：分别构造评估信道与设计信道（注意：不可混用）
        h_eff_true = cascaded_channel(ap_pos, ris_pos, p_true, lambda_)["h_eff"]
        h_eff_noisy = cascaded_channel(ap_pos, ris_pos, p_noisy, lambda_)["h_eff"]

        if dummy_phi is None:
            dummy_phi = np.ones_like(h_eff_true)

        # 步骤3：no-RIS 基线——直连增益在 p_true 上评估
        h_direct = direct_ap_robot_channel(
            ap_pos,
            p_true,
            lambda_,
            block_direct_link=block_direct_link,
        )
        snr_no_ris[t] = compute_snr(
            h_eff_true,
            dummy_phi,
            pt,
            sigma2,
            mode="no_ris",
            no_ris_gain=h_direct,
        )

        # 步骤4：理想 RIS——用 h_eff_true 配相，再用同一信道评估 SNR
        phi_ideal = deterministic_phase(h_eff_true)
        snr_ideal[t] = compute_snr(
            h_eff_true, phi_ideal, pt, sigma2, mode="ris"
        )

        # 步骤5：噪声设计——φ 来自 h_eff_noisy，SNR 仍在 h_eff_true 上评估
        # 注意：这是 Phase-1 展示“配相误差”的关键对比
        phi_noisy = deterministic_phase(h_eff_noisy)
        snr_noisy[t] = compute_snr(
            h_eff_true, phi_noisy, pt, sigma2, mode="ris"
        )

        # 步骤6（可选）：轨迹预测位姿仅用于配相，评估仍用 p_true
        p_pred: np.ndarray | None = None
        if predict_fn is not None:
            history = _history_window(traj_arr, t, t_hist)
            # future[0] ≈ traj[t]（服务时刻）；见上方英文 docstring
            p_pred = predict_fn(history, checkpoint_path=predictor_checkpoint)[0]

        if use_predictor and snr_predicted is not None and p_pred is not None:
            h_eff_pred = cascaded_channel(ap_pos, ris_pos, p_pred, lambda_)["h_eff"]
            phi_pred = deterministic_phase(h_eff_pred)
            snr_predicted[t] = compute_snr(
                h_eff_true, phi_pred, pt, sigma2, mode="ris"
            )

        # 步骤7（可选）：鲁棒配相；p_hat 为设计中心，SNR 仍在 h_eff_true
        if use_robust and snr_robust is not None and configure_ris_fn is not None:
            p_hat = p_pred if p_pred is not None else p_noisy
            phi_rob = configure_ris_fn(
                "robust",
                p_true,
                p_hat,
                robust_R,
                rng=rng,
            )
            snr_robust[t] = compute_snr(
                h_eff_true, phi_rob, pt, sigma2, mode="ris"
            )

    out: SnrCurves = {
        "time": time,
        "snr_no_ris": snr_no_ris,
        "snr_ideal": snr_ideal,
        "snr_noisy": snr_noisy,
    }
    if snr_predicted is not None:
        out["snr_predicted"] = snr_predicted
    if snr_robust is not None:
        out["snr_robust"] = snr_robust
    return out


def run_simulation(
    pos_noise_std: float | None = None,
    *,
    block_direct_link: bool | None = None,
    use_predictor: bool = False,
    predictor_checkpoint: str | None = None,
    use_robust: bool = False,
    robust_radius_m: float | None = None,
    traj: np.ndarray | None = None,
) -> SnrCurves:
    """Convenience entry: trajectory + SNR curves.

    If ``traj`` is omitted and ``use_predictor=True``, a synthetic path from
    ``phase2_traj.data.generator`` is used (in-distribution for the LSTM).
    Otherwise the phase-1 RIS-front arc demo is used (unchanged default).

    ``use_robust`` defaults to ``False`` so phase-1 demos stay unchanged.

    中文简释：便捷入口——选/生成轨迹后调用 simulate_snr_curves。
    要点：默认演示圆弧；仅 use_predictor=True 且未给 traj 时才用 phase2 轨迹。
    """
    std = config.POS_NOISE_STD_M if pos_noise_std is None else pos_noise_std

    if traj is None:
        if use_predictor:
            from phase2_traj.data.generator import generate_trajectory

            traj = generate_trajectory(
                config.NUM_TIME_STEPS,
                np.random.default_rng(config.RANDOM_SEED),
            )
        else:
            traj = generate_demo_trajectory(config.NUM_TIME_STEPS)

    return simulate_snr_curves(
        traj,
        pos_noise_std=std,
        block_direct_link=block_direct_link,
        use_predictor=use_predictor,
        predictor_checkpoint=predictor_checkpoint,
        use_robust=use_robust,
        robust_radius_m=robust_radius_m,
    )


if __name__ == "__main__":
    results = run_simulation()
    print(f"T = {results['time'].size}")
    print(
        "mean SNR [dB]: "
        f"no_ris={results['snr_no_ris'].mean():.2f}, "
        f"ideal={results['snr_ideal'].mean():.2f}, "
        f"noisy={results['snr_noisy'].mean():.2f}"
    )
    if not np.all(results["snr_noisy"] <= results["snr_ideal"] + 1e-9):
        raise AssertionError("snr_noisy exceeded snr_ideal (unexpected)")
    print("sanity: snr_noisy <= snr_ideal for all t")
