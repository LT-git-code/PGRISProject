"""SNR utilities for phase-1 RIS / no-RIS baselines (pure functions).

中文职责：端到端 SNR（dB）计算工具；纯函数，无状态。
输入：h_eff、φ（RIS 模式）或直连增益（no-RIS）、发射功率 Pt、噪声功率 σ²；
输出：SNR_dB = 10 log10(SNR_lin)。
在流程中的位置：配相（beamforming）之后 →【本模块评估链路质量】。
注意：RIS 模式用 |φᴴ h_eff|²；评估时应传入“真实几何”下的 h_eff，而非配相用的那份。
"""

from __future__ import annotations

import numpy as np

from phase1_mvp import config


def dbm_to_watt(power_dbm: float) -> float:
    """Convert dBm to linear watts: ``10^((P_dBm - 30) / 10)``.

    中文简释：dBm → 线性瓦特，供 Pt / σ² 进入 SNR 公式。
    """
    return float(10.0 ** ((power_dbm - 30.0) / 10.0))


def linear_to_db(value: float) -> float:
    """Convert a strictly positive linear value to dB: ``10 log10(value)``.

    中文简释：线性 SNR → SNR_dB；value 必须严格为正。
    """
    if value <= 0.0:
        raise ValueError("linear_to_db requires a strictly positive value")
    return float(10.0 * np.log10(value))


def compute_snr(
    h_eff: np.ndarray,
    phi: np.ndarray,
    pt: float,
    sigma2: float,
    mode: str = "ris",
    *,
    no_ris_gain: float | complex | None = None,
) -> float:
    """Compute end-to-end SNR in dB.

    Vector convention
    -----------------
    ``h_eff`` and ``phi`` are treated as length-``N`` **column** vectors.
    The RIS coherent scalar channel is the Hermitian inner product

        φᴴ h_eff  =  Σ_n conj(φ_n) · h_eff,n   (= ``np.vdot(phi, h_eff)``).

    Modes
    -----
    ``mode="ris"``
        Cascaded RIS link only (no direct path added here)::

            SNR_lin = Pt · |φᴴ h_eff|² / σ²

    ``mode="no_ris"``
        Direct AP→robot only. ``no_ris_gain`` is ``h_direct`` (complex or real
        amplitude). Prefer building it via
        ``channel.direct_ap_robot_channel(..., block_direct_link=...)``::

            SNR_lin = Pt · |h_direct|² / σ²

        If ``no_ris_gain`` is ``None``, falls back to the bare blocked constant
        ``config.DIRECT_PATH_ATTENUATION_BLOCKED`` (distance-agnostic; prefer
        passing an explicit ``h_direct``). ``h_eff`` / ``phi`` are ignored.

    中文简释：计算端到端 SNR_dB；RIS 用相干增益 |φᴴ h_eff|，no-RIS 用直连。
    要点：
      - ris：SNR_lin = Pt · |φᴴ h_eff|² / σ²
      - no_ris：SNR_lin = Pt · |h_direct|² / σ²（忽略 h_eff/φ）
    注意：评估信道应与配相信道分离——此处 h_eff 一般应为 p_true 下的
    h_eff_eval；φ 可来自 p_plan / 噪声位姿等设计信道。

    Parameters
    ----------
    h_eff :
        Cascaded channel, shape (N,). Used when ``mode="ris"``.
    phi :
        Unit-modulus RIS phases, shape (N,). Used when ``mode="ris"``.
    pt :
        Transmit power Pt in watts (linear).
    sigma2 :
        Noise power σ² in watts (linear).
    mode :
        ``"ris"`` or ``"no_ris"``.
    no_ris_gain :
        Optional override for the fixed direct gain in ``no_ris`` mode.
        If ``None``, uses ``config.DIRECT_PATH_ATTENUATION_BLOCKED``.

    Returns
    -------
    float
        SNR in dB.
    """
    if pt < 0.0 or sigma2 <= 0.0:
        raise ValueError("pt must be >= 0 and sigma2 must be > 0")

    if mode == "ris":
        # 步骤1（RIS）：校验形状，计算相干标量 φᴴ h_eff
        h = np.asarray(h_eff, dtype=np.complex128).reshape(-1)
        p = np.asarray(phi, dtype=np.complex128).reshape(-1)
        if h.shape != p.shape:
            raise ValueError("h_eff and phi must have the same shape")
        coherent = np.vdot(p, h)  # φᴴ h_eff
        # 步骤2：SNR_lin = Pt · |φᴴ h_eff|² / σ²
        snr_lin = pt * (np.abs(coherent) ** 2) / sigma2
    elif mode == "no_ris":
        # 步骤1（no-RIS）：取直连增益（优先显式 h_direct）
        g_dir = (
            float(config.DIRECT_PATH_ATTENUATION_BLOCKED)
            if no_ris_gain is None
            else no_ris_gain
        )
        # 步骤2：SNR_lin = Pt · |h_direct|² / σ²
        snr_lin = pt * (np.abs(g_dir) ** 2) / sigma2
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'ris' or 'no_ris'")

    # 步骤3：线性 → dB
    return linear_to_db(float(snr_lin))
