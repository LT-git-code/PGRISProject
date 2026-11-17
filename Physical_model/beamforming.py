"""RIS passive beamforming: deterministic conjugate phase alignment.

中文职责：RIS 被动波束赋形——确定性单位模相位对齐。
输入：级联有效信道 h_eff（通常由设计位姿 p_plan / p_noisy 等构造）；
输出：相位向量 φ，满足 |φ_n|=1，使 |φᴴ h_eff| 最大。
在流程中的位置：信道（channel）→【本模块配相】→ SNR（snr）。
注意：本模块只根据给定 h_eff 求 φ，不区分该 h_eff 来自 design 还是 evaluation。
"""

from __future__ import annotations

import numpy as np



def deterministic_phase(h_eff: np.ndarray) -> np.ndarray:
    """Unit-modulus RIS phases that maximize ``|φᴴ h_eff|``.

    Vector convention (column vectors)
    ----------------------------------
    The coherent cascade scalar is the Hermitian inner product

        φᴴ h_eff = Σ_n conj(φ_n) · h_eff,n  (= ``np.vdot(phi, h_eff)``).

    Under the unit-modulus constraint ``|φ_n| = 1``, the maximizer is

        φ_n = h_eff,n / |h_eff,n| = exp(+j ∠ h_eff,n)   (h_eff,n ≠ 0)

    which yields the tight upper bound

        φᴴ h_eff = ||h_eff||₁ .

    Zero entries of ``h_eff`` map to ``φ_n = 1``.

    Equivalence to “conjugate phase compensation”
    ---------------------------------------------
    Many RIS notes write the cascade as ``Σ_n φ'_n h_eff,n`` with
    ``φ'_n = exp(-j ∠ h_eff,n)``. That ``φ'`` equals ``conj(φ)`` here; both
    models give the same coherent gain ``||h_eff||₁``. This function returns
    ``φ`` for the ``φᴴ h_eff`` convention used by ``snr.compute_snr``.

    中文简释：在 |φ_n|=1 约束下，求使 |φᴴ h_eff| 最大的确定性相位。
    要点：φ_n = h_eff,n / |h_eff,n|；理想时 |φᴴ h_eff| = ||h_eff||₁。
    注意：上层若用 p_plan 的 h_eff_design 求 φ，评估 SNR 时必须改用
    p_true 的 h_eff_eval（本函数本身不做 design/evaluation 分离）。

    Parameters
    ----------
    h_eff :
        Cascaded effective channel, shape (N,).

    Returns
    -------
    np.ndarray
        Complex phase vector φ of shape (N,) with |φ_n| = 1.
    """
    h = np.asarray(h_eff, dtype=np.complex128).reshape(-1)

    # 步骤1：单位模相位对齐 φ_n = h_eff,n / |h_eff,n|（零元处置 φ_n=1）
    abs_h = np.abs(h)
    phi = np.ones(h.shape, dtype=np.complex128)
    nonzero = abs_h > 0.0
    phi[nonzero] = h[nonzero] / abs_h[nonzero]

    if not np.allclose(np.abs(phi), 1.0):
        raise AssertionError("deterministic_phase: |φ_n| must equal 1")

    # 步骤2：自检理想对齐 |φᴴ h_eff| = ||h_eff||₁（与 snr.compute_snr 的 φᴴ 约定一致）
    coherent_gain = float(np.abs(np.vdot(phi, h)))
    l1_norm = float(np.linalg.norm(h, ord=1))
    if not np.isclose(coherent_gain, l1_norm, rtol=1e-9, atol=1e-15):
        raise AssertionError(
            f"ideal alignment failed: |φᴴ h_eff|={coherent_gain} != ||h_eff||_1={l1_norm}"
        )

    return phi
