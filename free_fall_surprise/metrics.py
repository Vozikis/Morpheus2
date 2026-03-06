from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


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


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    # Effect size for difference in means: b - a
    var_a = float(np.var(a, ddof=1)) if a.shape[0] > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.shape[0] > 1 else 0.0
    pooled = ((a.shape[0] - 1) * var_a + (b.shape[0] - 1) * var_b) / max(a.shape[0] + b.shape[0] - 2, 1)
    pooled_std = math.sqrt(max(pooled, 1e-12))
    return float((float(np.mean(b)) - float(np.mean(a))) / pooled_std)
