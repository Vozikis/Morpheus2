#!/usr/bin/env python3
"""
Free-fall violation-of-expectation experiment using a causal transformer.

The script:
1) Simulates many free-fall trajectories under broad initial conditions/bounds.
2) Trains on in-distribution physical trajectories only.
3) Scores surprise (mean trajectory NLL) for:
   - physical but out-of-distribution trajectories
   - clearly non-physical trajectories
4) Saves summary metrics, per-trajectory CSV, and visual artifacts (PNG + GIF).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency for GIF output only
    imageio = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LOG_2PI = math.log(2.0 * math.pi)


def log_progress(label: str, current: int, total: int) -> None:
    pct = 100.0 * float(current) / float(max(total, 1))
    print(f"[PROGRESS] {label}: {current}/{total} ({pct:.1f}%)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Free-fall violation-of-expectation with transformer surprise scoring."
    )
    parser.add_argument("--x", type=int, default=10, help="Eval trajectories per class.")
    parser.add_argument(
        "--train_n", type=int, default=5000, help="Number of physical train trajectories."
    )
    parser.add_argument(
        "--val_n", type=int, default=1000, help="Number of physical validation trajectories."
    )
    parser.add_argument("--seq_len", type=int, default=80, help="Sequence length.")
    parser.add_argument("--dt", type=float, default=0.05, help="Timestep size.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=8,
        help="Minimum batch size allowed when auto-retrying after CUDA OOM.",
    )
    parser.add_argument(
        "--oom_retries",
        type=int,
        default=4,
        help="Maximum number of CUDA OOM retries with reduced batch size.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model.")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument(
        "--amp",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable AMP mixed precision on CUDA.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/free_fall_surprise",
        help="Output root directory.",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to save model checkpoint.",
    )
    parser.add_argument(
        "--save_gifs",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to generate per-trajectory GIF overlays.",
    )
    parser.add_argument(
        "--save_pngs",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to generate per-trajectory PNG overlays.",
    )
    parser.add_argument(
        "--gif_stride",
        type=int,
        default=1,
        help="Frame stride for GIF rendering (higher is faster/smaller).",
    )
    parser.add_argument(
        "--quality_eval_samples",
        type=int,
        default=256,
        help="Validation samples for prediction-quality sanity check.",
    )
    parser.add_argument(
        "--max_visualizations_per_group",
        type=int,
        default=0,
        help="Safety cap on PNG/GIF renders per evaluation group. 0 means auto (= x).",
    )
    args = parser.parse_args()
    if args.x <= 0:
        raise ValueError("--x must be > 0")
    if args.train_n <= 0 or args.val_n <= 0:
        raise ValueError("--train_n and --val_n must be > 0")
    if args.seq_len < 4:
        raise ValueError("--seq_len must be >= 4")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.min_batch_size <= 0:
        raise ValueError("--min_batch_size must be > 0")
    if args.min_batch_size > args.batch_size:
        raise ValueError("--min_batch_size cannot be larger than --batch_size")
    if args.oom_retries < 0:
        raise ValueError("--oom_retries must be >= 0")
    if args.d_model % args.nhead != 0:
        raise ValueError("--d_model must be divisible by --nhead")
    if args.gif_stride <= 0:
        raise ValueError("--gif_stride must be > 0")
    if args.quality_eval_samples <= 0:
        raise ValueError("--quality_eval_samples must be > 0")
    if args.max_visualizations_per_group < 0:
        raise ValueError("--max_visualizations_per_group must be >= 0")
    if args.save_gifs == 1 and imageio is None:
        raise RuntimeError("GIF saving requested but imageio is not installed.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs(output_dir: Path, save_pngs: bool, save_gifs: bool) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "plots_physical": output_dir / "plots" / "physical_ood",
        "plots_nonphysical": output_dir / "plots" / "non_physical",
        "gifs_physical": output_dir / "gifs" / "physical_ood",
        "gifs_nonphysical": output_dir / "gifs" / "non_physical",
        "checkpoints": output_dir / "checkpoints",
    }
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    if save_pngs:
        paths["plots_physical"].mkdir(parents=True, exist_ok=True)
        paths["plots_nonphysical"].mkdir(parents=True, exist_ok=True)
    if save_gifs:
        paths["gifs_physical"].mkdir(parents=True, exist_ok=True)
        paths["gifs_nonphysical"].mkdir(parents=True, exist_ok=True)
    return paths


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def sample_disjoint_range(
    rng: np.random.Generator, low_range: Tuple[float, float], high_range: Tuple[float, float]
) -> float:
    if rng.random() < 0.5:
        return float(rng.uniform(low_range[0], low_range[1]))
    return float(rng.uniform(high_range[0], high_range[1]))


def sample_physical_params(rng: np.random.Generator, mode: str) -> Dict[str, float]:
    if mode == "train":
        params = {
            "g": float(rng.uniform(8.5, 10.2)),
            "y0": float(rng.uniform(5.0, 50.0)),
            "v0": float(rng.uniform(-5.0, 5.0)),
            "drag": float(rng.uniform(0.0, 0.08)),
            "wind": float(rng.uniform(-0.3, 0.3)),
            "noise_std": float(rng.uniform(0.0, 0.03)),
            "floor": float(rng.uniform(-2.0, 2.0)),
            "ceiling": float(rng.uniform(55.0, 70.0)),
            "restitution": float(rng.uniform(0.55, 0.90)),
        }
    elif mode == "ood":
        params = {
            "g": sample_disjoint_range(rng, (5.5, 7.2), (11.2, 13.5)),
            "y0": float(rng.uniform(60.0, 100.0)),
            "v0": sample_disjoint_range(rng, (-20.0, -8.0), (8.0, 20.0)),
            "drag": float(rng.uniform(0.12, 0.30)),
            "wind": sample_disjoint_range(rng, (-1.6, -0.6), (0.6, 1.6)),
            "noise_std": float(rng.uniform(0.05, 0.15)),
            "floor": float(rng.uniform(-10.0, -5.0)),
            "ceiling": float(rng.uniform(110.0, 140.0)),
            "restitution": float(rng.uniform(0.15, 0.45)),
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Wall bounds included as metadata hooks for future >1D extensions.
    params["wall_left"] = -100.0
    params["wall_right"] = 100.0
    return params


def simulate_physical_trajectory(
    seq_len: int, dt: float, params: Dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    y = np.zeros(seq_len, dtype=np.float32)
    v = np.zeros(seq_len, dtype=np.float32)
    y[0] = float(params["y0"])
    v[0] = float(params["v0"])
    floor = float(params["floor"])
    ceiling = float(params["ceiling"])

    for t in range(1, seq_len):
        prev_y = float(y[t - 1])
        prev_v = float(v[t - 1])
        g = float(params["g"])
        drag = float(params["drag"])
        wind = float(params["wind"])
        noise_std = float(params["noise_std"])

        acc = -g - drag * prev_v + wind
        v_new = prev_v + acc * dt + float(rng.normal(0.0, noise_std))
        y_new = prev_y + v_new * dt + float(rng.normal(0.0, noise_std * 0.5))

        if y_new < floor:
            y_new = floor + (floor - y_new)
            v_new = abs(v_new) * float(params["restitution"])
        elif y_new > ceiling:
            y_new = ceiling - (y_new - ceiling)
            v_new = -abs(v_new) * float(params["restitution"])

        y[t] = np.float32(y_new)
        v[t] = np.float32(v_new)

    return np.stack([y, v], axis=-1).astype(np.float32)


def generate_physical_dataset(
    n: int,
    seq_len: int,
    dt: float,
    mode: str,
    rng: np.random.Generator,
    progress_label: Optional[str] = None,
    return_metadata: bool = True,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    trajectories = np.zeros((n, seq_len, 2), dtype=np.float32)
    metadata: List[Dict[str, float]] = []
    progress_stride = max(1, n // 10)
    for i in range(n):
        params = sample_physical_params(rng, mode=mode)
        traj = simulate_physical_trajectory(seq_len=seq_len, dt=dt, params=params, rng=rng)
        trajectories[i] = traj
        if return_metadata:
            metadata.append(params)
        if progress_label and ((i + 1) == 1 or (i + 1) % progress_stride == 0 or (i + 1) == n):
            log_progress(progress_label, i + 1, n)
    return trajectories, metadata


def generate_random_walk(seq_len: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    steps = rng.normal(0.0, 2.8, size=seq_len).astype(np.float32)
    y = np.cumsum(steps, dtype=np.float32) + np.float32(rng.uniform(-20.0, 20.0))
    v = np.gradient(y, dt).astype(np.float32)
    return np.stack([y, v], axis=-1).astype(np.float32)


def generate_anti_gravity(seq_len: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(seq_len, dtype=np.float32) * np.float32(dt)
    y0 = float(rng.uniform(-20.0, 20.0))
    v0 = float(rng.uniform(3.0, 18.0))
    upward_acc = float(rng.uniform(6.0, 16.0))
    y = y0 + v0 * t + 0.5 * upward_acc * (t**2)
    y += rng.normal(0.0, 1.0, size=seq_len).astype(np.float32)
    v = np.gradient(y, dt).astype(np.float32)
    return np.stack([y.astype(np.float32), v], axis=-1).astype(np.float32)


def generate_sinusoidal_forcing(seq_len: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(seq_len, dtype=np.float32) * np.float32(dt)
    amp = float(rng.uniform(5.0, 40.0))
    freq = float(rng.uniform(0.4, 2.5))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    drift = float(rng.uniform(-5.0, 5.0))
    offset = float(rng.uniform(-25.0, 25.0))
    y = amp * np.sin(2.0 * math.pi * freq * t + phase) + drift * t + offset
    y += rng.normal(0.0, 0.8, size=seq_len).astype(np.float32)
    v = np.gradient(y, dt).astype(np.float32)
    return np.stack([y.astype(np.float32), v], axis=-1).astype(np.float32)


def generate_teleport_jump(seq_len: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    params = sample_physical_params(rng, mode="train")
    base = simulate_physical_trajectory(seq_len, dt, params, rng)
    y = base[:, 0].copy()
    jump_count = int(rng.integers(2, 6))
    jump_points = rng.choice(np.arange(4, seq_len - 2), size=jump_count, replace=False)
    jump_points.sort()
    for jp in jump_points:
        jump_mag = float(rng.uniform(-30.0, 30.0))
        y[jp:] += np.float32(jump_mag)
    y += rng.normal(0.0, 0.5, size=seq_len).astype(np.float32)
    v = np.gradient(y, dt).astype(np.float32)
    return np.stack([y.astype(np.float32), v], axis=-1).astype(np.float32)


def generate_nonphysical_dataset(
    n: int,
    seq_len: int,
    dt: float,
    rng: np.random.Generator,
    progress_label: Optional[str] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    trajectories = np.zeros((n, seq_len, 2), dtype=np.float32)
    metadata: List[Dict[str, float]] = []
    regimes = ["random_walk", "anti_gravity", "sinusoidal_forcing", "teleport_jump"]
    progress_stride = max(1, n // 10)

    for i in range(n):
        regime = str(rng.choice(regimes))
        if regime == "random_walk":
            traj = generate_random_walk(seq_len, dt, rng)
        elif regime == "anti_gravity":
            traj = generate_anti_gravity(seq_len, dt, rng)
        elif regime == "sinusoidal_forcing":
            traj = generate_sinusoidal_forcing(seq_len, dt, rng)
        else:
            traj = generate_teleport_jump(seq_len, dt, rng)

        trajectories[i] = traj
        metadata.append(
            {
                "regime": regime,
                "is_physical": 0,
            }
        )
        if progress_label and ((i + 1) == 1 or (i + 1) % progress_stride == 0 or (i + 1) == n):
            log_progress(progress_label, i + 1, n)
    return trajectories, metadata


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories: np.ndarray):
        # Input: t=0..T-2 ; Target: t=1..T-1
        self.inputs = torch.from_numpy(trajectories[:, :-1, :]).float()
        self.targets = torch.from_numpy(trajectories[:, 1:, :]).float()

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


class CausalTrajectoryTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 4)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, 2]
        bsz, steps, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:, :steps, :]
        causal_mask = torch.triu(
            torch.ones(steps, steps, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(h, mask=causal_mask)
        out = self.head(h)
        mu = out[..., :2]
        log_sigma = out[..., 2:].clamp(min=-6.0, max=4.0)
        return mu, log_sigma


def gaussian_nll_per_timestep(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-2.0 * log_sigma)
    sq = (y - mu) ** 2
    # Per feature NLL.
    nll_feat = 0.5 * sq * inv_var + log_sigma + 0.5 * LOG_2PI
    # Sum features, keep [B, T]
    return nll_feat.sum(dim=-1)


def gaussian_nll_loss(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return gaussian_nll_per_timestep(mu, log_sigma, y).mean()


def normalize_trajectories(trajectories: np.ndarray, stats: NormStats) -> np.ndarray:
    return ((trajectories - stats.mean) / stats.std).astype(np.float32)


def compute_norm_stats(train_trajectories: np.ndarray) -> NormStats:
    flat = train_trajectories.reshape(-1, train_trajectories.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return NormStats(mean=mean, std=std)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    amp: bool,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": []}
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(
        "[INFO] Training setup | epochs={} train_batches={} val_batches={} lr={} amp={}".format(
            epochs, len(train_loader), len(val_loader), lr, int(use_amp)
        ),
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                mu, log_sigma = model(x_batch)
                loss = gaussian_nll_loss(mu, log_sigma, y_batch)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = x_batch.shape[0]
            train_loss_sum += float(loss.item()) * bs
            train_count += bs

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    mu, log_sigma = model(x_batch)
                    loss = gaussian_nll_loss(mu, log_sigma, y_batch)
                bs = x_batch.shape[0]
                val_loss_sum += float(loss.item()) * bs
                val_count += bs

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            pct = 100.0 * float(epoch) / float(max(epochs, 1))
            print(
                f"[Epoch {epoch:03d}/{epochs}] ({pct:.1f}%) train_nll={train_loss:.5f} val_nll={val_loss:.5f}",
                flush=True,
            )

    return history


def score_single_trajectory(
    model: nn.Module,
    trajectory: np.ndarray,
    stats: NormStats,
    device: torch.device,
) -> Dict[str, np.ndarray | float]:
    norm_traj = normalize_trajectories(trajectory[None, ...], stats=stats)[0]
    x = torch.from_numpy(norm_traj[:-1, :]).float().unsqueeze(0).to(device)
    y = torch.from_numpy(norm_traj[1:, :]).float().unsqueeze(0).to(device)

    with torch.no_grad():
        mu, log_sigma = model(x)
        nll_steps = gaussian_nll_per_timestep(mu, log_sigma, y)[0].cpu().numpy().astype(np.float64)
        pred_norm = mu[0].cpu().numpy().astype(np.float64)

    pred = pred_norm * stats.std + stats.mean
    actual = trajectory[1:, :].astype(np.float64)
    time = np.arange(actual.shape[0], dtype=np.float64)
    err = pred - actual
    rmse_pos = float(np.sqrt(np.mean(err[:, 0] ** 2)))
    rmse_vel = float(np.sqrt(np.mean(err[:, 1] ** 2)))
    mae_pos = float(np.mean(np.abs(err[:, 0])))
    mae_vel = float(np.mean(np.abs(err[:, 1])))
    return {
        "surprise_mean_nll": float(np.mean(nll_steps)),
        "surprise_std_nll": float(np.std(nll_steps)),
        "rmse_pos": rmse_pos,
        "rmse_vel": rmse_vel,
        "mae_pos": mae_pos,
        "mae_vel": mae_vel,
        "nll_steps": nll_steps,
        "pred": pred,
        "actual": actual,
        "time": time,
    }


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
    frames: List[np.ndarray] = []
    total_steps = len(time)
    for t_end in range(2, total_steps + 1, frame_stride):
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

    if (total_steps - 1) % frame_stride != 0:
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

    imageio.mimsave(path, frames, duration=0.08)


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.zeros(values.shape[0], dtype=np.float64)
    sorted_vals = values[order]
    n = values.shape[0]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def compute_auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    # Positive class is non-physical, where larger surprise is expected.
    n_neg = neg_scores.shape[0]
    n_pos = pos_scores.shape[0]
    if n_neg == 0 or n_pos == 0:
        return float("nan")
    all_scores = np.concatenate([neg_scores, pos_scores], axis=0)
    ranks = average_ranks(all_scores)
    pos_ranks = ranks[n_neg:]
    rank_sum_pos = float(np.sum(pos_ranks))
    auroc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    q25, q50, q75 = np.percentile(scores, [25.0, 50.0, 75.0])
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def summarize_prediction_errors(records: List[Dict[str, object]], group: str) -> Dict[str, float]:
    group_rows = [r for r in records if str(r.get("group")) == group]
    if not group_rows:
        return {}
    rmse_pos = np.asarray([float(r["rmse_pos"]) for r in group_rows], dtype=np.float64)
    rmse_vel = np.asarray([float(r["rmse_vel"]) for r in group_rows], dtype=np.float64)
    mae_pos = np.asarray([float(r["mae_pos"]) for r in group_rows], dtype=np.float64)
    mae_vel = np.asarray([float(r["mae_vel"]) for r in group_rows], dtype=np.float64)
    return {
        "rmse_pos_mean": float(np.mean(rmse_pos)),
        "rmse_vel_mean": float(np.mean(rmse_vel)),
        "mae_pos_mean": float(np.mean(mae_pos)),
        "mae_vel_mean": float(np.mean(mae_vel)),
        "rmse_pos_median": float(np.median(rmse_pos)),
        "rmse_vel_median": float(np.median(rmse_vel)),
        "mae_pos_median": float(np.median(mae_pos)),
        "mae_vel_median": float(np.median(mae_vel)),
    }


def evaluate_prediction_quality_subset(
    model: nn.Module,
    trajectories: np.ndarray,
    stats: NormStats,
    device: torch.device,
    max_samples: int,
    label: str,
) -> Dict[str, float]:
    n_total = int(trajectories.shape[0])
    n_eval = min(n_total, int(max_samples))
    stride = max(1, n_eval // 10)

    rmse_pos_vals: List[float] = []
    rmse_vel_vals: List[float] = []
    mae_pos_vals: List[float] = []
    mae_vel_vals: List[float] = []

    for i in range(n_eval):
        res = score_single_trajectory(model, trajectories[i], stats, device)
        rmse_pos_vals.append(float(res["rmse_pos"]))
        rmse_vel_vals.append(float(res["rmse_vel"]))
        mae_pos_vals.append(float(res["mae_pos"]))
        mae_vel_vals.append(float(res["mae_vel"]))
        if (i + 1) == 1 or (i + 1) % stride == 0 or (i + 1) == n_eval:
            log_progress(label, i + 1, n_eval)

    rmse_pos_arr = np.asarray(rmse_pos_vals, dtype=np.float64)
    rmse_vel_arr = np.asarray(rmse_vel_vals, dtype=np.float64)
    mae_pos_arr = np.asarray(mae_pos_vals, dtype=np.float64)
    mae_vel_arr = np.asarray(mae_vel_vals, dtype=np.float64)
    return {
        "sample_count": int(n_eval),
        "rmse_pos_mean": float(np.mean(rmse_pos_arr)),
        "rmse_vel_mean": float(np.mean(rmse_vel_arr)),
        "mae_pos_mean": float(np.mean(mae_pos_arr)),
        "mae_vel_mean": float(np.mean(mae_vel_arr)),
    }


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    # Effect size for difference in means: b - a
    var_a = float(np.var(a, ddof=1)) if a.shape[0] > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.shape[0] > 1 else 0.0
    pooled = ((a.shape[0] - 1) * var_a + (b.shape[0] - 1) * var_b) / max(a.shape[0] + b.shape[0] - 2, 1)
    pooled_std = math.sqrt(max(pooled, 1e-12))
    return float((float(np.mean(b)) - float(np.mean(a))) / pooled_std)


def main() -> None:
    run_start = time.time()
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    paths = ensure_dirs(output_dir, save_pngs=bool(args.save_pngs), save_gifs=bool(args.save_gifs))
    rng = np.random.default_rng(args.seed)

    print("[STAGE 1/9] Setup", flush=True)
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Output directory: {output_dir}", flush=True)
    print(
        "[INFO] Config | x={} train_n={} val_n={} seq_len={} dt={} epochs={} batch_size={} lr={} d_model={} nhead={} num_layers={} amp={} save_pngs={} save_gifs={} max_viz_per_group={} oom_retries={} min_batch_size={}".format(
            args.x,
            args.train_n,
            args.val_n,
            args.seq_len,
            args.dt,
            args.epochs,
            args.batch_size,
            args.lr,
            args.d_model,
            args.nhead,
            args.num_layers,
            args.amp,
            args.save_pngs,
            args.save_gifs,
            args.max_visualizations_per_group,
            args.oom_retries,
            args.min_batch_size,
        ),
        flush=True,
    )

    print("[STAGE 2/9] Generating datasets", flush=True)
    print("[INFO] Generating in-distribution physical training trajectories...", flush=True)
    train_traj, _ = generate_physical_dataset(
        args.train_n,
        args.seq_len,
        args.dt,
        mode="train",
        rng=rng,
        progress_label="generate_train_physical",
        return_metadata=False,
    )
    print("[INFO] Generating in-distribution physical validation trajectories...", flush=True)
    val_traj, _ = generate_physical_dataset(
        args.val_n,
        args.seq_len,
        args.dt,
        mode="train",
        rng=rng,
        progress_label="generate_val_physical",
        return_metadata=False,
    )
    print("[INFO] Generating physical out-of-distribution evaluation trajectories...", flush=True)
    phys_ood_traj, phys_ood_meta = generate_physical_dataset(
        args.x,
        args.seq_len,
        args.dt,
        mode="ood",
        rng=rng,
        progress_label="generate_eval_physical_ood",
        return_metadata=True,
    )
    print("[INFO] Generating non-physical evaluation trajectories...", flush=True)
    nonphys_traj, nonphys_meta = generate_nonphysical_dataset(
        args.x,
        args.seq_len,
        args.dt,
        rng=rng,
        progress_label="generate_eval_non_physical",
    )

    print("[STAGE 3/9] Normalization and data loaders", flush=True)
    stats = compute_norm_stats(train_traj)
    train_norm = normalize_trajectories(train_traj, stats)
    val_norm = normalize_trajectories(val_traj, stats)

    train_ds = TrajectoryDataset(train_norm)
    val_ds = TrajectoryDataset(val_norm)
    print(
        f"[INFO] Dataset sizes | train={len(train_ds)} val={len(val_ds)} eval_total={2 * args.x}",
        flush=True,
    )

    print("[STAGE 4/9] Model initialization", flush=True)

    def build_model() -> CausalTrajectoryTransformer:
        return CausalTrajectoryTransformer(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_len=args.seq_len - 1,
        ).to(device)

    def build_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
        loader_gen = torch.Generator().manual_seed(args.seed)
        train_loader_local = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=loader_gen,
            num_workers=0,
        )
        val_loader_local = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        return train_loader_local, val_loader_local

    print("[STAGE 5/9] Training transformer on physical-only trajectories", flush=True)
    current_batch_size = int(args.batch_size)
    model: Optional[CausalTrajectoryTransformer] = None
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    oom_retries_used = 0

    while True:
        model = build_model()
        train_loader, val_loader = build_loaders(current_batch_size)
        print(
            f"[INFO] Training attempt {oom_retries_used + 1} | batch_size={current_batch_size}",
            flush=True,
        )
        try:
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                amp=bool(args.amp),
            )
            break
        except RuntimeError as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or "cuda error: out of memory" in msg
            can_retry = (
                device.type == "cuda"
                and is_oom
                and oom_retries_used < args.oom_retries
                and current_batch_size > args.min_batch_size
            )
            if not can_retry:
                raise
            next_batch_size = max(args.min_batch_size, current_batch_size // 2)
            if next_batch_size == current_batch_size:
                raise
            oom_retries_used += 1
            print(
                f"[WARN] CUDA OOM detected. Retrying with smaller batch size: {current_batch_size} -> {next_batch_size}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current_batch_size = next_batch_size

    print(
        f"[INFO] Training completed with effective_batch_size={current_batch_size} oom_retries_used={oom_retries_used}",
        flush=True,
    )

    print("[STAGE 6/9] Saving checkpoint (if enabled)", flush=True)
    if model is None:
        raise RuntimeError("Model was not initialized successfully.")
    if args.save_checkpoint == 1:
        ckpt_path = paths["checkpoints"] / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "norm_mean": stats.mean.tolist(),
                "norm_std": stats.std.tolist(),
                "args": vars(args),
                "effective_batch_size": current_batch_size,
                "oom_retries_used": oom_retries_used,
                "train_history": history,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved checkpoint: {ckpt_path}", flush=True)
    else:
        print("[INFO] Checkpoint saving disabled.", flush=True)

    print("[STAGE 7/9] Sanity check: prediction quality on in-distribution validation subset", flush=True)
    val_quality = evaluate_prediction_quality_subset(
        model=model,
        trajectories=val_traj,
        stats=stats,
        device=device,
        max_samples=args.quality_eval_samples,
        label="id_val_prediction_quality",
    )
    print(
        "[INFO] ID validation quality | samples={} rmse_pos={:.4f} rmse_vel={:.4f} mae_pos={:.4f} mae_vel={:.4f}".format(
            val_quality["sample_count"],
            val_quality["rmse_pos_mean"],
            val_quality["rmse_vel_mean"],
            val_quality["mae_pos_mean"],
            val_quality["mae_vel_mean"],
        ),
        flush=True,
    )

    print("[STAGE 8/9] Scoring + rendering trajectories", flush=True)
    if args.save_pngs == 0:
        print("[INFO] PNG rendering disabled (--save_pngs=0).", flush=True)
    if args.save_gifs == 0:
        print("[INFO] GIF rendering disabled (--save_gifs=0). PNG overlays will still be generated.", flush=True)
    records: List[Dict[str, object]] = []
    physical_scores: List[float] = []
    nonphysical_scores: List[float] = []
    max_viz = int(args.x if args.max_visualizations_per_group == 0 else min(args.max_visualizations_per_group, args.x))
    print(
        f"[INFO] Visualization plan | per_group={max_viz} (x={args.x}, cap_arg={args.max_visualizations_per_group})",
        flush=True,
    )
    physical_rendered = 0
    nonphysical_rendered = 0
    physical_viz_skip_notified = False
    nonphysical_viz_skip_notified = False
    eval_total = 2 * args.x
    eval_done = 0
    eval_progress_stride = max(1, eval_total // 10)

    # Physical OOD group
    for i in range(args.x):
        res = score_single_trajectory(model, phys_ood_traj[i], stats, device)
        score = float(res["surprise_mean_nll"])
        physical_scores.append(score)
        traj_id = f"physical_ood_{i:04d}"
        png_path = paths["plots_physical"] / f"traj_{i:04d}.png"
        gif_path = paths["gifs_physical"] / f"traj_{i:04d}.gif"
        title = f"Physical OOD | {traj_id} | mean_nll={score:.4f}"
        should_render = i < max_viz
        if should_render:
            if args.save_pngs == 1:
                save_overlay_png(png_path, res["time"], res["actual"], res["pred"], title)
            physical_rendered += int(args.save_pngs == 1)
            if args.save_gifs == 1:
                save_overlay_gif(
                    gif_path,
                    res["time"],
                    res["actual"],
                    res["pred"],
                    title,
                    frame_stride=args.gif_stride,
                )
        elif not physical_viz_skip_notified:
            print(
                f"[WARN] Visualization cap reached for physical_ood group (max={max_viz}). Remaining trajectories will be scored but not rendered.",
                flush=True,
            )
            physical_viz_skip_notified = True

        meta = dict(phys_ood_meta[i])
        meta["group"] = "physical_ood"
        meta["is_physical"] = 1
        records.append(
            {
                "trajectory_id": traj_id,
                "group": "physical_ood",
                "surprise_mean_nll": score,
                "surprise_std_nll": float(res["surprise_std_nll"]),
                "rmse_pos": float(res["rmse_pos"]),
                "rmse_vel": float(res["rmse_vel"]),
                "mae_pos": float(res["mae_pos"]),
                "mae_vel": float(res["mae_vel"]),
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )
        eval_done += 1
        if eval_done == 1 or eval_done % eval_progress_stride == 0 or eval_done == eval_total:
            log_progress("score_and_render", eval_done, eval_total)
            print(f"[INFO] Latest complete: {traj_id} | score={score:.5f}", flush=True)

    # Non-physical group
    for i in range(args.x):
        res = score_single_trajectory(model, nonphys_traj[i], stats, device)
        score = float(res["surprise_mean_nll"])
        nonphysical_scores.append(score)
        traj_id = f"non_physical_{i:04d}"
        png_path = paths["plots_nonphysical"] / f"traj_{i:04d}.png"
        gif_path = paths["gifs_nonphysical"] / f"traj_{i:04d}.gif"
        title = f"Non-Physical | {traj_id} | mean_nll={score:.4f}"
        should_render = i < max_viz
        if should_render:
            if args.save_pngs == 1:
                save_overlay_png(png_path, res["time"], res["actual"], res["pred"], title)
            nonphysical_rendered += int(args.save_pngs == 1)
            if args.save_gifs == 1:
                save_overlay_gif(
                    gif_path,
                    res["time"],
                    res["actual"],
                    res["pred"],
                    title,
                    frame_stride=args.gif_stride,
                )
        elif not nonphysical_viz_skip_notified:
            print(
                f"[WARN] Visualization cap reached for non_physical group (max={max_viz}). Remaining trajectories will be scored but not rendered.",
                flush=True,
            )
            nonphysical_viz_skip_notified = True

        meta = dict(nonphys_meta[i])
        meta["group"] = "non_physical"
        records.append(
            {
                "trajectory_id": traj_id,
                "group": "non_physical",
                "surprise_mean_nll": score,
                "surprise_std_nll": float(res["surprise_std_nll"]),
                "rmse_pos": float(res["rmse_pos"]),
                "rmse_vel": float(res["rmse_vel"]),
                "mae_pos": float(res["mae_pos"]),
                "mae_vel": float(res["mae_vel"]),
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )
        eval_done += 1
        if eval_done == 1 or eval_done % eval_progress_stride == 0 or eval_done == eval_total:
            log_progress("score_and_render", eval_done, eval_total)
            print(f"[INFO] Latest complete: {traj_id} | score={score:.5f}", flush=True)

    print("[STAGE 9/9] Aggregating metrics + writing artifacts", flush=True)
    df = pd.DataFrame.from_records(records)
    csv_path = paths["metrics"] / "per_trajectory_scores.csv"
    df.to_csv(csv_path, index=False)

    physical_scores_np = np.asarray(physical_scores, dtype=np.float64)
    nonphysical_scores_np = np.asarray(nonphysical_scores, dtype=np.float64)
    auroc = compute_auroc(neg_scores=physical_scores_np, pos_scores=nonphysical_scores_np)

    summary = {
        "config": vars(args),
        "device_used": str(device),
        "counts": {
            "physical_ood": int(args.x),
            "non_physical": int(args.x),
            "total_eval": int(2 * args.x),
        },
        "train_history": history,
        "runtime_safety": {
            "effective_batch_size": int(current_batch_size),
            "oom_retries_used": int(oom_retries_used),
            "max_visualizations_per_group": int(max_viz),
        },
        "prediction_quality": {
            "id_validation_subset": val_quality,
            "physical_ood": summarize_prediction_errors(records, "physical_ood"),
            "non_physical": summarize_prediction_errors(records, "non_physical"),
        },
        "surprise_stats": {
            "physical_ood": summarize_scores(physical_scores_np),
            "non_physical": summarize_scores(nonphysical_scores_np),
            "delta_mean_nonphysical_minus_physical": float(
                np.mean(nonphysical_scores_np) - np.mean(physical_scores_np)
            ),
            "delta_median_nonphysical_minus_physical": float(
                np.median(nonphysical_scores_np) - np.median(physical_scores_np)
            ),
            "effect_size_cohen_d": cohen_d(physical_scores_np, nonphysical_scores_np),
            "auroc_nonphysical_as_positive": auroc,
        },
        "artifacts": {
            "per_trajectory_scores_csv": str(csv_path),
            "plots_dir": str(output_dir / "plots") if args.save_pngs == 1 else None,
            "gifs_dir": str(output_dir / "gifs") if args.save_gifs == 1 else None,
            "rendered_png_count_physical_ood": int(physical_rendered),
            "rendered_png_count_non_physical": int(nonphysical_rendered),
            "rendered_gif_count_physical_ood": int(physical_rendered if args.save_gifs == 1 else 0),
            "rendered_gif_count_non_physical": int(nonphysical_rendered if args.save_gifs == 1 else 0),
            "checkpoint": str(paths["checkpoints"] / "model.pt") if args.save_checkpoint == 1 else None,
        },
    }

    summary_path = paths["metrics"] / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed_sec = time.time() - run_start
    print(f"[INFO] Saved metrics: {summary_path}", flush=True)
    print(f"[INFO] Saved per-trajectory CSV: {csv_path}", flush=True)
    print(
        "[INFO] Surprise means | physical_ood={:.5f} | non_physical={:.5f} | delta={:.5f}".format(
            float(np.mean(physical_scores_np)),
            float(np.mean(nonphysical_scores_np)),
            float(np.mean(nonphysical_scores_np) - np.mean(physical_scores_np)),
        ),
        flush=True,
    )
    print(f"[INFO] AUROC(non_physical positive)={auroc:.5f}", flush=True)
    print(f"[INFO] Done. Total runtime: {elapsed_sec:.1f}s", flush=True)


if __name__ == "__main__":
    main()
