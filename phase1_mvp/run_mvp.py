"""Entry point for phase-1 MVP (+ optional Predicted-RIS curve).

Run from repo root::

    python -m phase1_mvp.run_mvp

中文职责：Phase-1 MVP 命令行入口——跑仿真、出图、做验收断言。
输入：无 CLI 参数；读 config 默认噪声 σ 与输出路径。
输出：outputs 下 SNR 对比 PNG，以及控制台均值 / 增益损耗。
在流程中的位置：整条 Phase-1 的可运行入口（simulation → plot）。
注意：后半段 Predicted-RIS 依赖 phase2，属可选对比；论文主线以 thesis_planner_ris 为准。
"""

from __future__ import annotations

from pathlib import Path

from phase1_mvp import config
from phase1_mvp.plot_snr import plot_snr_from_results
from phase1_mvp.simulation import run_simulation


def _pkg_dir() -> Path:
    """中文简释：本包根目录，用于把相对输出路径解析为绝对路径。"""
    return Path(__file__).resolve().parent


def main() -> None:
    """Run baseline MVP, then Predicted-RIS comparison if phase2 is available.

    中文简释：先跑纯 Phase-1（无预测），验收 Ideal > No-RIS 且 Noisy ≤ Ideal；
    再可选跑 Predicted 对比图（需 phase2）。
    """
    # --- Original phase-1 behaviour (unchanged curves) ---
    # 步骤1：默认噪声位姿设计，评估用 p_true（见 simulation）
    results = run_simulation(pos_noise_std=config.POS_NOISE_STD_M, use_predictor=False)
    save_path = _pkg_dir() / config.SNR_PLOT_SAVE_PATH
    # 步骤2：绘制并保存 Phase-1 三条基线
    means = plot_snr_from_results(results, save_path=save_path)

    # 步骤3：打印增益/损耗，并做 Phase-1 验收
    gain_ideal_vs_no_ris = means["ideal"] - means["no_ris"]
    loss_noisy_vs_ideal = means["ideal"] - means["noisy"]
    print(f"Average gain (Ideal - No RIS): {gain_ideal_vs_no_ris:.2f} dB")
    print(f"Average loss (Ideal - Noisy): {loss_noisy_vs_ideal:.2f} dB")

    if means["ideal"] <= means["no_ris"]:
        raise RuntimeError(
            "Acceptance failed: Ideal RIS should be clearly above No RIS."
        )
    if means["noisy"] > means["ideal"] + 1e-6:
        raise RuntimeError(
            "Acceptance failed: Noisy-position RIS exceeded Ideal RIS."
        )
    print("Acceptance (phase-1): Ideal > No RIS, and Noisy <= Ideal.")

    # --- Phase-1+2: add Predicted-RIS (in-distribution traj for the LSTM) ---
    # 步骤4（可选）：Predicted-RIS 对比；非 thesis_planner_ris 主线
    print("--- Predicted-RIS comparison ---")
    results_p2 = run_simulation(
        pos_noise_std=config.POS_NOISE_STD_M,
        use_predictor=True,
    )
    save_p2 = _pkg_dir() / config.SNR_PLOT_SAVE_PATH_PHASE2
    means_p2 = plot_snr_from_results(
        results_p2,
        save_path=save_p2,
        title=config.SNR_PLOT_TITLE_PHASE2,
    )

    if "predicted" not in means_p2:
        raise RuntimeError("snr_predicted missing from phase-2 run")

    # 步骤5：检查 Ideal / Predicted / Noisy 均值排序是否合理
    ideal_m = means_p2["ideal"]
    pred_m = means_p2["predicted"]
    noisy_m = means_p2["noisy"]
    print(
        f"Mean ordering check [dB]: Ideal={ideal_m:.2f}, "
        f"Predicted={pred_m:.2f}, Noisy={noisy_m:.2f}"
    )
    print(f"Ideal - Predicted: {ideal_m - pred_m:.2f} dB")
    print(f"Predicted - Noisy: {pred_m - noisy_m:.2f} dB")

    if pred_m > ideal_m + 0.5:
        raise RuntimeError(
            "Predicted SNR unexpectedly above Ideal — check time alignment "
            "(history should end at t-1; future[0] serves time t)."
        )
    if pred_m + 0.25 < noisy_m:
        print(
            "WARNING: mean Predicted < Noisy; check OOD trajectory / off-by-one."
        )
    else:
        print("Acceptance (phase-2): Ideal >= Predicted >= Noisy (mean).")


if __name__ == "__main__":
    main()
