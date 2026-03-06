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


def _symmetric_matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Matrix contains non-finite values.")
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    sqrt_diag = np.diag(np.sqrt(eigvals))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sqrt_matrix = eigvecs @ sqrt_diag @ eigvecs.T
    return 0.5 * (sqrt_matrix + sqrt_matrix.T)


def fit_gaussian(embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2D [N, D]")
    n, d = embeddings.shape
    mean = np.mean(embeddings, axis=0).astype(np.float64)
    if n > 1:
        cov = np.cov(embeddings, rowvar=False).astype(np.float64)
    else:
        cov = np.zeros((d, d), dtype=np.float64)
    return {"mean": mean, "cov": cov}


def compute_frechet_distance(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
    eps: float = 1e-6,
) -> float:
    if mean_a.shape != mean_b.shape:
        raise ValueError("Mean vectors must have identical shape.")
    if cov_a.shape != cov_b.shape:
        raise ValueError("Covariance matrices must have identical shape.")
    d = cov_a.shape[0]
    cov_a_reg = cov_a + np.eye(d, dtype=np.float64) * eps
    cov_b_reg = cov_b + np.eye(d, dtype=np.float64) * eps
    sqrt_cov_a = _symmetric_matrix_sqrt(cov_a_reg)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        middle = sqrt_cov_a @ cov_b_reg @ sqrt_cov_a
    sqrt_middle = _symmetric_matrix_sqrt(middle)
    diff = mean_a - mean_b
    distance = float(diff.dot(diff) + np.trace(cov_a_reg + cov_b_reg - 2.0 * sqrt_middle))
    return max(0.0, distance)


def frechet_trajectory_distance(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    eps: float = 1e-6,
) -> float:
    ga = fit_gaussian(embeddings_a)
    gb = fit_gaussian(embeddings_b)
    return compute_frechet_distance(
        mean_a=ga["mean"],
        cov_a=ga["cov"],
        mean_b=gb["mean"],
        cov_b=gb["cov"],
        eps=eps,
    )


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
