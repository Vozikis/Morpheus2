from __future__ import annotations

import math
from typing import Optional

import torch


LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll_per_timestep(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    inv_var = torch.exp(-2.0 * log_sigma)
    sq = (y - mu) ** 2
    # Per feature NLL.
    nll_feat = 0.5 * sq * inv_var + log_sigma + 0.5 * LOG_2PI
    # Sum features, keep [B, T]
    return nll_feat.sum(dim=-1)


def masked_mean(values: torch.Tensor, step_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if step_mask is None:
        return values.mean()
    mask = step_mask.to(dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def gaussian_nll_loss(
    mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor, step_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    per_step = gaussian_nll_per_timestep(mu, log_sigma, y)
    return masked_mean(per_step, step_mask=step_mask)
