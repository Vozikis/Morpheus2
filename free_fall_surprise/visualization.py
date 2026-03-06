from __future__ import annotations

import io
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency for GIF output only
    imageio = None


def save_overlay_png(path: Path, time: np.ndarray, actual: np.ndarray, pred: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(time, actual[:, 0], label="actual_pos", linewidth=2.0)
    axes[0].plot(time, pred[:, 0], label="pred_pos", linewidth=1.8, alpha=0.85)
    axes[0].set_ylabel("Position")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time, actual[:, 1], label="actual_vel", linewidth=2.0)
    axes[1].plot(time, pred[:, 1], label="pred_vel", linewidth=1.8, alpha=0.85)
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Velocity")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_overlay_gif(
    path: Path,
    time: np.ndarray,
    actual: np.ndarray,
    pred: np.ndarray,
    title: str,
    frame_stride: int = 1,
) -> None:
    if imageio is None:
        raise RuntimeError("Cannot save GIF: imageio is not available.")
    total_steps = len(time)
    if total_steps <= 0:
        return

    stride = max(1, int(frame_stride))
    frames: List[np.ndarray] = []
    last_t_end = 0
    for t_end in range(1, total_steps + 1, stride):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(time[:t_end], actual[:t_end, 0], label="actual_pos", linewidth=2.0)
        axes[0].plot(time[:t_end], pred[:t_end, 0], label="pred_pos", linewidth=1.8, alpha=0.85)
        axes[0].set_ylabel("Position")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(time[:t_end], actual[:t_end, 1], label="actual_vel", linewidth=2.0)
        axes[1].plot(time[:t_end], pred[:t_end, 1], label="pred_vel", linewidth=1.8, alpha=0.85)
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Velocity")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="best")

        fig.suptitle(f"{title} (t={t_end}/{total_steps})")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        last_t_end = t_end

    if last_t_end != total_steps:
        # Ensure final state is included even with stride > 1.
        t_end = total_steps
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        axes[0].plot(time[:t_end], actual[:t_end, 0], label="actual_pos", linewidth=2.0)
        axes[0].plot(time[:t_end], pred[:t_end, 0], label="pred_pos", linewidth=1.8, alpha=0.85)
        axes[0].set_ylabel("Position")
        axes[0].grid(alpha=0.3)
        axes[0].legend(loc="best")
        axes[1].plot(time[:t_end], actual[:t_end, 1], label="actual_vel", linewidth=2.0)
        axes[1].plot(time[:t_end], pred[:t_end, 1], label="pred_vel", linewidth=1.8, alpha=0.85)
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Velocity")
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc="best")
        fig.suptitle(f"{title} (t={t_end}/{total_steps})")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

    if not frames:
        return
    imageio.mimsave(path, frames, duration=0.08)
