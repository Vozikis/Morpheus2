from __future__ import annotations

import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def log_progress(label: str, current: int, total: int) -> None:
    pct = 100.0 * float(current) / float(max(total, 1))
    print(f"[PROGRESS] {label}: {current}/{total} ({pct:.1f}%)", flush=True)


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
        "plots_id": output_dir / "plots" / "in_distribution",
        "plots_physical": output_dir / "plots" / "physical_ood",
        "plots_nonphysical": output_dir / "plots" / "non_physical",
        "gifs_id": output_dir / "gifs" / "in_distribution",
        "gifs_physical": output_dir / "gifs" / "physical_ood",
        "gifs_nonphysical": output_dir / "gifs" / "non_physical",
        "checkpoints": output_dir / "checkpoints",
    }
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    if save_pngs:
        paths["plots_id"].mkdir(parents=True, exist_ok=True)
        paths["plots_physical"].mkdir(parents=True, exist_ok=True)
        paths["plots_nonphysical"].mkdir(parents=True, exist_ok=True)
    if save_gifs:
        paths["gifs_id"].mkdir(parents=True, exist_ok=True)
        paths["gifs_physical"].mkdir(parents=True, exist_ok=True)
        paths["gifs_nonphysical"].mkdir(parents=True, exist_ok=True)
    return paths
