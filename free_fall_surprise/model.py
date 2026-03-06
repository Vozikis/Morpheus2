from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CausalTrajectoryTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
        context_window: int = -1,
        log_sigma_min: float = -6.0,
        log_sigma_max: float = 1.5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.context_window = int(context_window)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
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
        idx = torch.arange(steps, device=x.device)
        q_pos = idx.view(-1, 1)
        k_pos = idx.view(1, -1)
        future_mask = k_pos > q_pos
        if self.context_window >= 0:
            too_old_mask = k_pos < (q_pos - self.context_window)
            causal_mask = future_mask | too_old_mask
        else:
            causal_mask = future_mask
        h = self.encoder(h, mask=causal_mask)
        out = self.head(h)
        mu = out[..., :2]
        log_sigma = out[..., 2:].clamp(min=self.log_sigma_min, max=self.log_sigma_max)
        return mu, log_sigma
