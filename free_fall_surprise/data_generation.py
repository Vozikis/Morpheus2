from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .runtime import log_progress


def sample_physical_params(rng: np.random.Generator, mode: str) -> Dict[str, float]:
    if mode == "train":
        params = {
            # Broad physical "foundation" range for 1D ball drop on Earth-like conditions.
            "g": float(rng.uniform(9.70, 9.90)),  # m/s^2
            "y0": float(rng.uniform(2.0, 80.0)),  # release height (m)
            "v0": float(rng.uniform(0.0, 15.0)),  # initial upward speed (m/s)
            "mass": float(rng.uniform(0.01, 5.0)),  # kg
            "radius": float(rng.uniform(0.005, 0.25)),  # m
            "drag_coefficient": float(rng.uniform(0.15, 1.20)),  # broad sphere-like Cd range
            "air_density": float(rng.uniform(0.90, 1.35)),  # kg/m^3 (altitude/weather variation)
            "noise_std": float(rng.uniform(0.0, 0.01)),
            "floor": 0.0,
        }
    elif mode == "ood":
        params = {
            # OOD physical tails: still lawful free-fall but shifted to harder extremes.
            "g": float(rng.uniform(9.45, 10.20)),
            "y0": float(rng.uniform(80.0, 180.0)),
            "v0": float(rng.uniform(15.0, 35.0)),
            "mass": float(rng.uniform(0.005, 8.0)),
            "radius": float(rng.uniform(0.003, 0.35)),
            # Keep these IID relative to training regime.
            "drag_coefficient": float(rng.uniform(0.15, 1.20)),
            "air_density": float(rng.uniform(0.90, 1.35)),
            "noise_std": float(rng.uniform(0.008, 0.03)),
            "floor": 0.0,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return params


def simulate_physical_trajectory(
    seq_len: int, dt: float, params: Dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    y = np.zeros(seq_len, dtype=np.float32)
    v = np.zeros(seq_len, dtype=np.float32)
    y[0] = float(params["y0"])
    v[0] = float(params["v0"])
    floor = float(params["floor"])
    g = float(params["g"])
    mass = max(float(params["mass"]), 1e-6)
    radius = max(float(params["radius"]), 1e-6)
    drag_coefficient = max(float(params["drag_coefficient"]), 0.0)
    air_density = max(float(params["air_density"]), 0.0)
    noise_std = float(params["noise_std"])

    # Quadratic drag constant for 1D vertical motion.
    area = math.pi * radius * radius
    drag_k = 0.5 * air_density * drag_coefficient * area / mass

    for t in range(1, seq_len):
        prev_y = float(y[t - 1])
        prev_v = float(v[t - 1])
        if prev_y <= floor + 1e-8 and abs(prev_v) <= 1e-6:
            # Ball has landed and remains at rest on the ground.
            y[t] = np.float32(floor)
            v[t] = np.float32(0.0)
            continue

        # a = -g - k v |v| (drag opposite to velocity).
        acc = -g - drag_k * prev_v * abs(prev_v)
        v_new = prev_v + acc * dt + float(rng.normal(0.0, noise_std))
        y_new = prev_y + v_new * dt + float(rng.normal(0.0, noise_std * 0.25))

        # Ground contact: no bounce for pure drop experiments.
        if y_new <= floor:
            y[t] = np.float32(floor)
            v[t] = np.float32(0.0)
            if t + 1 < seq_len:
                y[t + 1 :] = np.float32(floor)
                v[t + 1 :] = np.float32(0.0)
            break

        y[t] = np.float32(y_new)
        v[t] = np.float32(v_new)

    return np.stack([y, v], axis=-1).astype(np.float32)


def compute_valid_prediction_steps(
    trajectory: np.ndarray, floor: float = 0.0, eps: float = 1e-8
) -> int:
    """
    Number of one-step predictions to evaluate (targets at t=1..t=valid_steps).
    For free-fall, we stop at first ground contact (inclusive).
    """
    total_steps = int(trajectory.shape[0] - 1)
    if total_steps <= 0:
        return 0
    y = trajectory[:, 0]
    hit_idx = np.flatnonzero(y <= floor + eps)
    if hit_idx.size == 0:
        return total_steps
    impact_t = int(hit_idx[0])
    if impact_t <= 0:
        return 1
    return min(impact_t, total_steps)


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
