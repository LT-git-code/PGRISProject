"""Plotting helpers for phase-1 / phase-1+2 SNR curves (2D only).

中文职责：将仿真得到的 SNR 时间序列画成 2D 对比图并落盘。
输入：time 与各曲线 SNR_dB（及可选 snr_predicted）；
输出：PNG 图 + 各曲线平均 SNR 字典（便于验收打印）。
在流程中的位置：simulation 之后的可视化；不参与配相或信道计算。
注意：本模块只绘图，不区分 p_plan / p_true；曲线语义由上游 simulation 保证。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from phase1_mvp import config


def plot_snr_curves(
    time: np.ndarray,
    snr_no_ris: np.ndarray,
    snr_ideal: np.ndarray,
    snr_noisy: np.ndarray,
    save_path: str | Path,
    *,
    snr_predicted: np.ndarray | None = None,
    title: str | None = None,
) -> dict[str, float]:
    """Plot SNR baselines (and optional Predicted-RIS) vs time.

    中文简释：绘制 No-RIS / Ideal / Noisy（及可选 Predicted）的 SNR–时间曲线。
    要点：先算均值并打印，再保存 PNG；标签文案来自 config。

    Parameters
    ----------
    time :
        Time-step indices, shape (T,).
    snr_no_ris, snr_ideal, snr_noisy :
        SNR series in dB, each shape (T,).
    snr_predicted :
        Optional Predicted-RIS SNR [dB].
    save_path :
        Output PNG path (parent directories are created if needed).
    title :
        Optional figure title override.

    Returns
    -------
    dict[str, float]
        Mean SNR [dB] for each plotted curve.
    """
    t = np.asarray(time, dtype=float)
    y_no = np.asarray(snr_no_ris, dtype=float)
    y_id = np.asarray(snr_ideal, dtype=float)
    y_ns = np.asarray(snr_noisy, dtype=float)

    # 步骤1：各基线均值（供控制台与验收）
    means: dict[str, float] = {
        "no_ris": float(np.mean(y_no)),
        "ideal": float(np.mean(y_id)),
        "noisy": float(np.mean(y_ns)),
    }
    msg = (
        "Mean SNR [dB]: "
        f"No RIS={means['no_ris']:.2f}, "
        f"Ideal RIS={means['ideal']:.2f}, "
        f"Noisy-position RIS={means['noisy']:.2f}"
    )

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 步骤2：绘制三条核心曲线
    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)
    ax.plot(t, y_no, label=config.PLOT_LABEL_NO_RIS, linewidth=1.8)
    ax.plot(t, y_id, label=config.PLOT_LABEL_IDEAL, linewidth=1.8)
    ax.plot(t, y_ns, label=config.PLOT_LABEL_NOISY, linewidth=1.8)

    # 步骤3（可选）：Predicted-RIS 曲线（phase1+2 对比用，非 thesis 主线）
    if snr_predicted is not None:
        y_pr = np.asarray(snr_predicted, dtype=float)
        means["predicted"] = float(np.nanmean(y_pr))
        ax.plot(t, y_pr, label=config.PLOT_LABEL_PREDICTED, linewidth=1.8)
        msg += f", Predicted-RIS={means['predicted']:.2f}"

    # 步骤4：标注、落盘、关闭 figure
    print(msg)
    ax.set_title(title or config.SNR_PLOT_TITLE)
    ax.set_xlabel(config.SNR_PLOT_XLABEL)
    ax.set_ylabel(config.SNR_PLOT_YLABEL)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved figure: {out.resolve()}")

    return means


def plot_snr_from_results(
    results: Mapping[str, np.ndarray],
    save_path: str | Path,
    *,
    title: str | None = None,
) -> dict[str, float]:
    """Convenience wrapper around :func:`plot_snr_curves` for simulation dicts.

    中文简释：从 simulate_snr_curves 返回的字典直接取键绘图。
    """
    return plot_snr_curves(
        results["time"],
        results["snr_no_ris"],
        results["snr_ideal"],
        results["snr_noisy"],
        save_path=save_path,
        snr_predicted=results.get("snr_predicted"),  # type: ignore[arg-type]
        title=title,
    )
