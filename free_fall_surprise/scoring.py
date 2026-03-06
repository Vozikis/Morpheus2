from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .data_generation import compute_valid_prediction_steps
from .dataset import NormStats, normalize_trajectories
from .losses import LOG_2PI, gaussian_nll_per_timestep
from .runtime import log_progress


TARGET_ABS_EPS = 1e-8


def valid_target_step_mask(targets: np.ndarray, eps: float = TARGET_ABS_EPS) -> np.ndarray:
    finite_mask = np.isfinite(targets).all(axis=-1)
    # Keep steps where at least one channel is present/non-zero.
    non_zero_mask = (np.abs(targets) > eps).any(axis=-1)
    return np.logical_and(finite_mask, non_zero_mask)


def score_single_trajectory_teacher_forced(
    model: nn.Module,
    trajectory: np.ndarray,
    stats: NormStats,
    device: torch.device,
    floor: float = 0.0,
    valid_steps: Optional[int] = None,
) -> Dict[str, np.ndarray | float]:
    norm_traj = normalize_trajectories(trajectory[None, ...], stats=stats)[0]
    total_steps = int(norm_traj.shape[0] - 1)
    effective_steps = int(
        max(
            1,
            min(
                total_steps,
                valid_steps
                if valid_steps is not None
                else compute_valid_prediction_steps(trajectory, floor=floor),
            ),
        )
    )
    x = torch.from_numpy(norm_traj[:-1, :]).float().unsqueeze(0).to(device)
    y = torch.from_numpy(norm_traj[1:, :]).float().unsqueeze(0).to(device)

    with torch.no_grad():
        mu, log_sigma = model(x)
        nll_steps = gaussian_nll_per_timestep(mu, log_sigma, y)[0].cpu().numpy().astype(np.float64)
        pred_norm = mu[0].cpu().numpy().astype(np.float64)

    nll_steps = nll_steps[:effective_steps]
    pred = (pred_norm * stats.std + stats.mean)[:effective_steps]
    actual = trajectory[1:, :].astype(np.float64)[:effective_steps]
    time = np.arange(actual.shape[0], dtype=np.float64)
    step_mask = valid_target_step_mask(actual)
    err = pred - actual
    if not np.any(step_mask):
        raise ValueError("Encountered empty trajectory during scoring.")
    nll_eval = nll_steps[step_mask]
    err_eval = err[step_mask]
    rmse_pos = float(np.sqrt(np.mean(err_eval[:, 0] ** 2)))
    rmse_vel = float(np.sqrt(np.mean(err_eval[:, 1] ** 2)))
    mae_pos = float(np.mean(np.abs(err_eval[:, 0])))
    mae_vel = float(np.mean(np.abs(err_eval[:, 1])))
    return {
        "surprise_mean_nll": float(np.mean(nll_eval)),
        "surprise_std_nll": float(np.std(nll_eval)),
        "rmse_pos": rmse_pos,
        "rmse_vel": rmse_vel,
        "mae_pos": mae_pos,
        "mae_vel": mae_vel,
        "nll_steps": nll_eval,
        "pred": pred,
        "actual": actual,
        "time": time,
    }


def score_single_trajectory_rollout(
    model: nn.Module,
    trajectory: np.ndarray,
    stats: NormStats,
    device: torch.device,
    floor: float = 0.0,
    valid_steps: Optional[int] = None,
) -> Dict[str, np.ndarray | float]:
    norm_traj = normalize_trajectories(trajectory[None, ...], stats=stats)[0]
    total_steps = int(norm_traj.shape[0] - 1)
    effective_steps = int(
        max(
            1,
            min(
                total_steps,
                valid_steps
                if valid_steps is not None
                else compute_valid_prediction_steps(trajectory, floor=floor),
            ),
        )
    )

    generated_inputs: List[np.ndarray] = [norm_traj[0].astype(np.float32)]
    pred_norm_steps: List[np.ndarray] = []
    nll_steps_list: List[float] = []

    with torch.no_grad():
        for step in range(effective_steps):
            x_np = np.asarray(generated_inputs, dtype=np.float32)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            mu_seq, log_sigma_seq = model(x)
            mu_next = mu_seq[0, -1, :]
            log_sigma_next = log_sigma_seq[0, -1, :]
            y_true_next = torch.from_numpy(norm_traj[step + 1]).float().to(device)

            inv_var = torch.exp(-2.0 * log_sigma_next)
            sq = (y_true_next - mu_next) ** 2
            nll_step = (0.5 * sq * inv_var + log_sigma_next + 0.5 * LOG_2PI).sum()

            mu_next_np64 = mu_next.detach().cpu().numpy().astype(np.float64)
            mu_next_np32 = mu_next.detach().cpu().numpy().astype(np.float32)
            pred_norm_steps.append(mu_next_np64)
            generated_inputs.append(mu_next_np32)
            nll_steps_list.append(float(nll_step.item()))

    nll_steps = np.asarray(nll_steps_list, dtype=np.float64)
    pred_norm = np.stack(pred_norm_steps, axis=0).astype(np.float64)
    pred = pred_norm * stats.std + stats.mean
    actual = trajectory[1:, :].astype(np.float64)[:effective_steps]
    time = np.arange(actual.shape[0], dtype=np.float64)
    step_mask = valid_target_step_mask(actual)
    err = pred - actual
    if not np.any(step_mask):
        raise ValueError("Encountered empty trajectory during scoring.")
    nll_eval = nll_steps[step_mask]
    err_eval = err[step_mask]
    rmse_pos = float(np.sqrt(np.mean(err_eval[:, 0] ** 2)))
    rmse_vel = float(np.sqrt(np.mean(err_eval[:, 1] ** 2)))
    mae_pos = float(np.mean(np.abs(err_eval[:, 0])))
    mae_vel = float(np.mean(np.abs(err_eval[:, 1])))
    return {
        "surprise_mean_nll": float(np.mean(nll_eval)),
        "surprise_std_nll": float(np.std(nll_eval)),
        "rmse_pos": rmse_pos,
        "rmse_vel": rmse_vel,
        "mae_pos": mae_pos,
        "mae_vel": mae_vel,
        "nll_steps": nll_eval,
        "pred": pred,
        "actual": actual,
        "time": time,
    }


def score_single_trajectory(
    model: nn.Module,
    trajectory: np.ndarray,
    stats: NormStats,
    device: torch.device,
    mode: str,
    floor: float = 0.0,
    valid_steps: Optional[int] = None,
) -> Dict[str, np.ndarray | float]:
    if mode == "teacher_forced":
        return score_single_trajectory_teacher_forced(
            model, trajectory, stats, device, floor=floor, valid_steps=valid_steps
        )
    if mode == "rollout":
        return score_single_trajectory_rollout(
            model, trajectory, stats, device, floor=floor, valid_steps=valid_steps
        )
    raise ValueError(f"Unknown scoring mode: {mode}")


def trajectory_embedding(
    model: nn.Module,
    trajectory: np.ndarray,
    stats: NormStats,
    device: torch.device,
    floor: float = 0.0,
    valid_steps: Optional[int] = None,
) -> np.ndarray:
    norm_traj = normalize_trajectories(trajectory[None, ...], stats=stats)[0]
    total_steps = int(norm_traj.shape[0] - 1)
    effective_steps = int(
        max(
            1,
            min(
                total_steps,
                valid_steps
                if valid_steps is not None
                else compute_valid_prediction_steps(trajectory, floor=floor),
            ),
        )
    )
    x = torch.from_numpy(norm_traj[:effective_steps, :]).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if not hasattr(model, "encode"):
            raise AttributeError("Model must provide an encode(...) method for FTD embeddings.")
        hidden = model.encode(x)
        emb = hidden.mean(dim=1)[0].detach().cpu().numpy().astype(np.float64)
    return emb


def compute_trajectory_embeddings(
    model: nn.Module,
    trajectories: np.ndarray,
    stats: NormStats,
    device: torch.device,
    floor: float = 0.0,
    valid_steps: Optional[int] = None,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for traj in trajectories:
        emb = trajectory_embedding(
            model=model,
            trajectory=traj,
            stats=stats,
            device=device,
            floor=floor,
            valid_steps=valid_steps,
        )
        embeddings.append(emb)
    return np.stack(embeddings, axis=0).astype(np.float64)


def evaluate_prediction_quality_subset(
    model: nn.Module,
    trajectories: np.ndarray,
    stats: NormStats,
    device: torch.device,
    max_samples: int,
    label: str,
    scoring_mode: str,
) -> Dict[str, float]:
    n_total = int(trajectories.shape[0])
    n_eval = min(n_total, int(max_samples))
    stride = max(1, n_eval // 10)

    rmse_pos_vals: List[float] = []
    rmse_vel_vals: List[float] = []
    mae_pos_vals: List[float] = []
    mae_vel_vals: List[float] = []

    for i in range(n_eval):
        res = score_single_trajectory(model, trajectories[i], stats, device, mode=scoring_mode)
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
