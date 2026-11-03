"""Export GIF from a matplotlib FuncAnimation (pillow writer).

中文职责：将 FuncAnimation 导出为 GIF（pillow，无需 ffmpeg）。
输入：动画或 DemoFrame；输出：outputs 下 GIF 路径。
在流程中的位置：animate 之后的落盘；可对帧抽稀以控制文件大小。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from matplotlib.animation import FuncAnimation

from thesis_planner_ris import config
from thesis_planner_ris.demo.pipeline import DemoFrame


def resolve_out(path: str | Path) -> Path:
    """中文简释：相对路径解析到 config.OUTPUTS_DIR 下。"""
    p = Path(path)
    if not p.is_absolute():
        p = config.OUTPUTS_DIR / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def subsample_frames(
    frames: Sequence[DemoFrame],
    max_frames: int | None = None,
) -> list[DemoFrame]:
    """Stride frames so GIF stays small.

    中文简释：按 DEMO_MAX_FRAMES 对帧抽稀，避免 GIF 过大。
    """
    n = len(frames)
    cap = int(config.DEMO_MAX_FRAMES if max_frames is None else max_frames)
    if n <= cap:
        return list(frames)
    stride = max(1, (n + cap - 1) // cap)
    return list(frames[::stride])


def export_gif(
    anim: FuncAnimation,
    out_path: str | Path,
    *,
    fps: int | None = None,
    dpi: int | None = None,
) -> Path:
    """Save GIF via pillow (no ffmpeg required).

    中文简释：用 pillow writer 保存 GIF。
    """
    out = resolve_out(out_path)
    fps_v = int(config.DEMO_FPS if fps is None else fps)
    dpi_v = int(config.DEMO_DPI if dpi is None else dpi)
    anim.save(out, writer="pillow", fps=fps_v, dpi=dpi_v)
    return out


def export_gif_from_frames(
    frames: Sequence[DemoFrame],
    out_path: str | Path,
    *,
    build_animation: Any = None,
) -> Path:
    """Build animation from (possibly subsampled) frames and write GIF.

    中文简释：抽稀帧 -> build_animation -> export_gif 的一键封装。
    """
    import matplotlib.pyplot as plt

    from thesis_planner_ris.demo.animate_mpl import build_animation as _build

    builder = build_animation or _build
    frames_use = subsample_frames(frames)
    fig, anim, _meta = builder(frames_use)
    path = export_gif(anim, out_path)
    plt.close(fig)
    return path
