"""Matplotlib dual-panel demo: scene (left) + SNR (right).

Colors (Stage-4 brief):
  green  = p_true
  orange = p_plan
  blue   = illustrative beam cone (toward p_plan; NOT an EM pattern)

中文职责：Matplotlib 双栏动画：左 3D 场景，右 SNR 曲线。
输入：DemoFrame 列表；输出：Figure + FuncAnimation（或静态 preview PNG）。
在流程中的位置：pipeline 之后的可视化；不改变配相/信道逻辑。
注意：蓝色光锥仅示意（指向 p_plan），不是真实电磁波束图样。
几何复用自 phase1_mvp.geometry（AP/RIS）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from phase1_mvp.geometry import (
    get_ap_position,
    get_ris_center,
    get_ris_element_positions,
)

from thesis_planner_ris import config
from thesis_planner_ris.demo.pipeline import DemoFrame, frames_to_arrays


def _beam_cone_mesh(
    apex: np.ndarray,
    aim: np.ndarray,
    radius: float,
    n_circle: int = 16,
) -> dict[str, np.ndarray]:
    """Stylized cone: apex at RIS, base around aim (demo only).

    中文简释：示意光锥网格（顶点 RIS，底面绕 aim=p_plan）；仅演示用。
    """
    apex = np.asarray(apex, dtype=float).reshape(3)
    aim = np.asarray(aim, dtype=float).reshape(3)
    axis = aim - apex
    nrm = float(np.linalg.norm(axis))
    if nrm < 1e-9:
        axis = np.array([0.0, 1.0, 0.0])
        nrm = 1.0
    axis = axis / nrm
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)

    angles = np.linspace(0.0, 2.0 * np.pi, n_circle, endpoint=False)
    circle = aim + radius * (
        np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    )
    verts = np.vstack([apex.reshape(1, 3), circle])
    faces = np.array(
        [[0, i + 1, ((i + 1) % n_circle) + 1] for i in range(n_circle)]
    )
    return {"vertices": verts, "faces": faces}


def build_animation(
    frames: Sequence[DemoFrame],
) -> tuple[Figure, FuncAnimation, dict[str, Any]]:
    """Build FuncAnimation from precomputed DemoFrame list.

    meta['update'] is the frame drawer (also used for static preview).

    中文简释：构造双栏动画：绿=p_true、橙=p_plan、蓝=示意光锥；右侧 SNR 在 p_true 上评估。
    """
    if not frames:
        raise ValueError("frames must be non-empty")

    # 步骤1：堆叠序列 + 静态几何（AP/RIS）；预计算指向 p_plan 的示意锥
    series = frames_to_arrays(frames)
    ap = get_ap_position()
    ris = get_ris_center()
    elems = get_ris_element_positions()
    p_true = series["p_true"]
    p_plan = series["p_plan"]
    t_idx = series["t"]

    cones: list[dict[str, np.ndarray]] = []
    for i in range(len(frames)):
        dist = float(np.linalg.norm(p_plan[i] - ris))
        r = max(config.DEMO_BEAM_RADIUS_FRAC * max(dist, 1e-3), 0.3)
        cones.append(_beam_cone_mesh(ris, p_plan[i], r))

    # 步骤2：左 3D 场景静态层 + 右 SNR 轴
    fig = plt.figure(figsize=config.DEMO_FIGURE_SIZE)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    fig.suptitle(
        "Planner-assisted RIS demo (phi from p_plan, SNR on p_true)",
        fontsize=12,
    )

    ax3d.scatter(
        elems[:, 0], elems[:, 1], elems[:, 2], c="steelblue", s=6, depthshade=False
    )
    ax3d.scatter(
        [ap[0]], [ap[1]], [ap[2]], c="purple", s=70, marker="^", label="AP"
    )
    ax3d.scatter(
        [ris[0]], [ris[1]], [ris[2]], c="navy", s=40, marker="s", label="RIS"
    )

    pad = 5.0
    ax3d.set_xlim(
        float(min(ap[0], p_true[:, 0].min()) - pad),
        float(p_true[:, 0].max() + pad),
    )
    ax3d.set_ylim(
        float(min(0.0, p_true[:, 1].min()) - pad),
        float(p_true[:, 1].max() + pad),
    )
    ax3d.set_zlim(0.0, max(12.0, float(ap[2]) + 2.0))
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("Scene (beam cone = illustrative)")

    (traj_line,) = ax3d.plot([], [], [], c="0.5", lw=1.5, label="traj")
    (true_pt,) = ax3d.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=10,
        c="green",
        label="p_true",
    )
    (plan_pt,) = ax3d.plot(
        [],
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=8,
        c="darkorange",
        label="p_plan",
    )
    beam_holder: list[Any] = [None]
    ax3d.legend(loc="upper left", fontsize=7)

    y_stack = np.concatenate(
        [series["snr_no_ris"], series["snr_ideal_ris"], series["snr_planner_ris"]]
    )
    (ln_nr,) = ax2d.plot([], [], c="C0", lw=1.5, label=config.PLOT_LABEL_NO_RIS)
    (ln_id,) = ax2d.plot([], [], c="C1", lw=1.6, label=config.PLOT_LABEL_IDEAL_RIS)
    (ln_pl,) = ax2d.plot([], [], c="C2", lw=1.6, label=config.PLOT_LABEL_PLANNER_RIS)
    vline = ax2d.axvline(t_idx[0], color="0.3", ls="--", lw=1.0, alpha=0.8)
    readouts = ax2d.text(
        0.98,
        0.97,
        "",
        transform=ax2d.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax2d.set_xlim(float(t_idx[0]), float(t_idx[-1]))
    ax2d.set_ylim(float(y_stack.min()) - 1.0, float(y_stack.max()) + 1.0)
    ax2d.set_xlabel("Time step")
    ax2d.set_ylabel("SNR (dB)")
    ax2d.set_title("SNR (eval on p_true)")
    ax2d.grid(True, ls="--", alpha=0.4)
    ax2d.legend(loc="lower left", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    n = len(frames)

    def _set_beam(i: int) -> None:
        if beam_holder[0] is not None:
            beam_holder[0].remove()
            beam_holder[0] = None
        mesh = cones[i]
        verts = mesh["vertices"]
        tris = [[verts[j] for j in face] for face in mesh["faces"]]
        coll = Poly3DCollection(
            tris,
            alpha=0.35,
            facecolor="dodgerblue",
            edgecolor="royalblue",
            linewidths=0.3,
        )
        ax3d.add_collection3d(coll)
        beam_holder[0] = coll

    # 步骤3：每帧更新 p_true/p_plan 位置、示意锥与 SNR 曲线
    def update(i: int):
        i = int(i) % n
        traj_line.set_data(p_true[: i + 1, 0], p_true[: i + 1, 1])
        traj_line.set_3d_properties(p_true[: i + 1, 2])
        true_pt.set_data([p_true[i, 0]], [p_true[i, 1]])
        true_pt.set_3d_properties([p_true[i, 2]])
        plan_pt.set_data([p_plan[i, 0]], [p_plan[i, 1]])
        plan_pt.set_3d_properties([p_plan[i, 2]])
        _set_beam(i)

        tt = t_idx[: i + 1]
        ln_nr.set_data(tt, series["snr_no_ris"][: i + 1])
        ln_id.set_data(tt, series["snr_ideal_ris"][: i + 1])
        ln_pl.set_data(tt, series["snr_planner_ris"][: i + 1])
        vline.set_xdata([t_idx[i], t_idx[i]])
        readouts.set_text(
            f"planner_ris {series['snr_planner_ris'][i]:.1f} dB\n"
            f"err {series['planner_pose_error_m'][i]:.2f} m"
        )
        return ()

    anim = FuncAnimation(
        fig, update, frames=n, interval=1000.0 / config.DEMO_FPS, blit=False
    )
    update(0)
    return fig, anim, {"n": n, "series": series, "update": update}


def save_preview_png(
    frames: Sequence[DemoFrame],
    out_path: str | Path,
    *,
    frame_index: int | None = None,
) -> Path:
    """Save a static mid-run preview (demo_preview.png).

    中文简释：保存静态预览图（默认取中间帧）。
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, anim, meta = build_animation(frames)
    idx = len(frames) // 2 if frame_index is None else int(frame_index)
    update_fn: Callable[[int], Any] = meta["update"]
    update_fn(idx)
    fig.savefig(out, dpi=config.DEMO_DPI)
    # Keep `anim` referenced until after save to avoid Matplotlib warning.
    del anim
    plt.close(fig)
    return out
