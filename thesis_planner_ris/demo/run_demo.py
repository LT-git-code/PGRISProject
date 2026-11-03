"""CLI for the planner-assisted RIS demo (no ML).

Examples (from repo root)::

    python -m thesis_planner_ris.demo.run_demo --gif
    python -m thesis_planner_ris.demo.run_demo --preview-only

中文职责：Stage-4 Demo 命令行入口（无 ML）。
输入：CLI（--gif / --preview-only / --error-std / --steps / --show）；
输出：demo_preview.png，可选 demo_planner_ris.gif。
在流程中的位置：pipeline -> animate/export 的组装入口。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from thesis_planner_ris import config
from thesis_planner_ris.demo.animate_mpl import build_animation, save_preview_png
from thesis_planner_ris.demo.export_video import export_gif, subsample_frames
from thesis_planner_ris.demo.pipeline import run_demo_loop


def _parse_args() -> argparse.Namespace:
    """中文简释：解析 Demo CLI 参数。"""
    p = argparse.ArgumentParser(
        description="Planner-assisted RIS demo (p_plan / p_true; no ML)."
    )
    p.add_argument(
        "--gif",
        action="store_true",
        help=f"Export {config.DEMO_GIF} under outputs/",
    )
    p.add_argument(
        "--preview-only",
        action="store_true",
        help=f"Only write {config.DEMO_PREVIEW} (no GIF)",
    )
    p.add_argument(
        "--error-std",
        type=float,
        default=None,
        help="Execution noise std [m] (default: config.DEMO_ERROR_STD_M)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Trajectory length (default: config.DEMO_NUM_TIME_STEPS)",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window (optional)",
    )
    return p.parse_args()


def main() -> None:
    """中文简释：建帧 -> preview -> 可选 GIF/交互窗口。"""
    args = _parse_args()
    out_dir = config.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Demo: building frames (p_plan design, SNR on p_true; no ML) ...")
    # 步骤1：生成帧（p_plan 配相， p_true 评估）
    frames = run_demo_loop(
        n_time_steps=args.steps,
        error_std_m=args.error_std,
    )

    # 步骤2：静态预览图
    preview_path = out_dir / config.DEMO_PREVIEW
    save_preview_png(frames, preview_path)
    print(f"  preview: {preview_path}")

    # 步骤3（可选）：抽稀帧后导出 GIF
    gif_path: Path | None = None
    if args.gif and not args.preview_only:
        frames_gif = subsample_frames(frames)
        fig, anim, _meta = build_animation(frames_gif)
        gif_path = export_gif(anim, out_dir / config.DEMO_GIF)
        plt.close(fig)
        print(f"  gif:     {gif_path}")

    if args.show:
        fig, anim, _meta = build_animation(frames)
        plt.show()
        plt.close(fig)

    print("DONE Stage 4")
    print(f"  {preview_path}")
    if gif_path is not None:
        print(f"  {gif_path}")


if __name__ == "__main__":
    main()
