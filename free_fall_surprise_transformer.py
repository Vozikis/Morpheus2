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
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LOG_2PI = math.log(2.0 * math.pi)


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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model.")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
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
    args = parser.parse_args()
    if args.x <= 0:
        raise ValueError("--x must be > 0")
    if args.train_n <= 0 or args.val_n <= 0:
        raise ValueError("--train_n and --val_n must be > 0")
    if args.seq_len < 4:
        raise ValueError("--seq_len must be >= 4")
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


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "metrics": output_dir / "metrics",
        "plots_physical": output_dir / "plots" / "physical_ood",
        "plots_nonphysical": output_dir / "plots" / "non_physical",
        "gifs_physical": output_dir / "gifs" / "physical_ood",
        "gifs_nonphysical": output_dir / "gifs" / "non_physical",
        "checkpoints": output_dir / "checkpoints",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
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
    n: int, seq_len: int, dt: float, mode: str, rng: np.random.Generator
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    trajectories = np.zeros((n, seq_len, 2), dtype=np.float32)
    metadata: List[Dict[str, float]] = []
    for i in range(n):
        params = sample_physical_params(rng, mode=mode)
        traj = simulate_physical_trajectory(seq_len=seq_len, dt=dt, params=params, rng=rng)
        trajectories[i] = traj
        metadata.append(params)
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
    n: int, seq_len: int, dt: float, rng: np.random.Generator
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    trajectories = np.zeros((n, seq_len, 2), dtype=np.float32)
    metadata: List[Dict[str, float]] = []
    regimes = ["random_walk", "anti_gravity", "sinusoidal_forcing", "teleport_jump"]

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
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            mu, log_sigma = model(x_batch)
            loss = gaussian_nll_loss(mu, log_sigma, y_batch)
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
            print(
                f"[Epoch {epoch:03d}/{epochs}] train_nll={train_loss:.5f} val_nll={val_loss:.5f}",
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
    return {
        "surprise_mean_nll": float(np.mean(nll_steps)),
        "surprise_std_nll": float(np.std(nll_steps)),
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


def save_overlay_gif(path: Path, time: np.ndarray, actual: np.ndarray, pred: np.ndarray, title: str) -> None:
    frames: List[np.ndarray] = []
    total_steps = len(time)
    for t_end in range(2, total_steps + 1):
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


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    # Effect size for difference in means: b - a
    var_a = float(np.var(a, ddof=1)) if a.shape[0] > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.shape[0] > 1 else 0.0
    pooled = ((a.shape[0] - 1) * var_a + (b.shape[0] - 1) * var_b) / max(a.shape[0] + b.shape[0] - 2, 1)
    pooled_std = math.sqrt(max(pooled, 1e-12))
    return float((float(np.mean(b)) - float(np.mean(a))) / pooled_std)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir).resolve()
    paths = ensure_dirs(output_dir)
    rng = np.random.default_rng(args.seed)

    print(f"[INFO] Device: {device}")
    print("[INFO] Generating datasets...")
    train_traj, _ = generate_physical_dataset(args.train_n, args.seq_len, args.dt, mode="train", rng=rng)
    val_traj, _ = generate_physical_dataset(args.val_n, args.seq_len, args.dt, mode="train", rng=rng)
    phys_ood_traj, phys_ood_meta = generate_physical_dataset(args.x, args.seq_len, args.dt, mode="ood", rng=rng)
    nonphys_traj, nonphys_meta = generate_nonphysical_dataset(args.x, args.seq_len, args.dt, rng=rng)

    stats = compute_norm_stats(train_traj)
    train_norm = normalize_trajectories(train_traj, stats)
    val_norm = normalize_trajectories(val_traj, stats)

    train_ds = TrajectoryDataset(train_norm)
    val_ds = TrajectoryDataset(val_norm)
    loader_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = CausalTrajectoryTransformer(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.seq_len - 1,
    ).to(device)

    print("[INFO] Training transformer on physical-only trajectories...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
    )

    if args.save_checkpoint == 1:
        ckpt_path = paths["checkpoints"] / "model.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "norm_mean": stats.mean.tolist(),
                "norm_std": stats.std.tolist(),
                "args": vars(args),
                "train_history": history,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved checkpoint: {ckpt_path}")

    print("[INFO] Scoring physical OOD and non-physical trajectories...")
    records: List[Dict[str, object]] = []
    physical_scores: List[float] = []
    nonphysical_scores: List[float] = []

    # Physical OOD group
    for i in range(args.x):
        res = score_single_trajectory(model, phys_ood_traj[i], stats, device)
        score = float(res["surprise_mean_nll"])
        physical_scores.append(score)
        traj_id = f"physical_ood_{i:04d}"
        png_path = paths["plots_physical"] / f"traj_{i:04d}.png"
        gif_path = paths["gifs_physical"] / f"traj_{i:04d}.gif"
        title = f"Physical OOD | {traj_id} | mean_nll={score:.4f}"
        save_overlay_png(png_path, res["time"], res["actual"], res["pred"], title)
        save_overlay_gif(gif_path, res["time"], res["actual"], res["pred"], title)

        meta = dict(phys_ood_meta[i])
        meta["group"] = "physical_ood"
        meta["is_physical"] = 1
        records.append(
            {
                "trajectory_id": traj_id,
                "group": "physical_ood",
                "surprise_mean_nll": score,
                "surprise_std_nll": float(res["surprise_std_nll"]),
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )

    # Non-physical group
    for i in range(args.x):
        res = score_single_trajectory(model, nonphys_traj[i], stats, device)
        score = float(res["surprise_mean_nll"])
        nonphysical_scores.append(score)
        traj_id = f"non_physical_{i:04d}"
        png_path = paths["plots_nonphysical"] / f"traj_{i:04d}.png"
        gif_path = paths["gifs_nonphysical"] / f"traj_{i:04d}.gif"
        title = f"Non-Physical | {traj_id} | mean_nll={score:.4f}"
        save_overlay_png(png_path, res["time"], res["actual"], res["pred"], title)
        save_overlay_gif(gif_path, res["time"], res["actual"], res["pred"], title)

        meta = dict(nonphys_meta[i])
        meta["group"] = "non_physical"
        records.append(
            {
                "trajectory_id": traj_id,
                "group": "non_physical",
                "surprise_mean_nll": score,
                "surprise_std_nll": float(res["surprise_std_nll"]),
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )

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
            "plots_dir": str(output_dir / "plots"),
            "gifs_dir": str(output_dir / "gifs"),
            "checkpoint": str(paths["checkpoints"] / "model.pt") if args.save_checkpoint == 1 else None,
        },
    }

    summary_path = paths["metrics"] / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved metrics: {summary_path}")
    print(f"[INFO] Saved per-trajectory CSV: {csv_path}")
    print(
        "[INFO] Surprise means | physical_ood={:.5f} | non_physical={:.5f} | delta={:.5f}".format(
            float(np.mean(physical_scores_np)),
            float(np.mean(nonphysical_scores_np)),
            float(np.mean(nonphysical_scores_np) - np.mean(physical_scores_np)),
        )
    )
    print(f"[INFO] AUROC(non_physical positive)={auroc:.5f}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
