from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_generation import compute_valid_prediction_steps


TARGET_ABS_EPS = 1e-8


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        trajectories: np.ndarray,
        floor: float = 0.0,
        mask_source_trajectories: Optional[np.ndarray] = None,
    ):
        # Input: t=0..T-2 ; Target: t=1..T-1
        self.inputs = torch.from_numpy(trajectories[:, :-1, :]).float()
        self.targets = torch.from_numpy(trajectories[:, 1:, :]).float()
        n = trajectories.shape[0]
        horizon = trajectories.shape[1] - 1
        mask_source = trajectories if mask_source_trajectories is None else mask_source_trajectories
        step_masks = np.zeros((n, horizon), dtype=np.float32)
        for i in range(n):
            valid_steps = compute_valid_prediction_steps(mask_source[i], floor=floor)
            if valid_steps > 0:
                step_masks[i, :valid_steps] = 1.0
        # Also drop steps where either target feature is missing/non-finite or zero-valued.
        raw_targets = mask_source[:, 1:, :]
        finite_mask = np.isfinite(raw_targets).all(axis=-1)
        non_zero_mask = (np.abs(raw_targets) > TARGET_ABS_EPS).all(axis=-1)
        step_masks *= np.logical_and(finite_mask, non_zero_mask).astype(np.float32)
        self.step_masks = torch.from_numpy(step_masks).float()

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx], self.step_masks[idx]


def normalize_trajectories(trajectories: np.ndarray, stats: NormStats) -> np.ndarray:
    return ((trajectories - stats.mean) / stats.std).astype(np.float32)


def compute_norm_stats(train_trajectories: np.ndarray) -> NormStats:
    flat = train_trajectories.reshape(-1, train_trajectories.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return NormStats(mean=mean, std=std)
