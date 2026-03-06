from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .distributed import DistributedContext, allreduce_sum, is_main_process
from .losses import gaussian_nll_loss, masked_mean


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: Optional[DistributedSampler],
    ctx: DistributedContext,
    epochs: int,
    lr: float,
    amp: bool,
    sigma_reg_weight: float,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = {"train_loss": [], "val_loss": [], "train_sigma_reg": [], "val_sigma_reg": []}
    use_amp = bool(amp and ctx.device.type == "cuda")
    try:
        # Preferred API in newer PyTorch versions.
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        # Backward-compatible fallback.
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if is_main_process(ctx):
        print(
            "[INFO] Training setup | epochs={} train_batches={} val_batches={} lr={} amp={} sigma_reg_weight={} ddp={} world_size={}".format(
                epochs,
                len(train_loader),
                len(val_loader),
                lr,
                int(use_amp),
                sigma_reg_weight,
                int(ctx.use_ddp),
                ctx.world_size,
            ),
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        train_sigma_reg_sum = 0.0
        train_count = 0
        for x_batch, y_batch, step_mask in train_loader:
            x_batch = x_batch.to(ctx.device)
            y_batch = y_batch.to(ctx.device)
            step_mask = step_mask.to(ctx.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=use_amp):
                mu, log_sigma = model(x_batch)
                nll_loss = gaussian_nll_loss(mu, log_sigma, y_batch, step_mask=step_mask)
                sigma_per_step = torch.exp(2.0 * log_sigma).mean(dim=-1)
                sigma_reg = masked_mean(sigma_per_step, step_mask=step_mask)
                loss = nll_loss + sigma_reg_weight * sigma_reg

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
            train_loss_sum += float(nll_loss.item()) * bs
            train_sigma_reg_sum += float(sigma_reg.item()) * bs
            train_count += bs

        model.eval()
        val_loss_sum = 0.0
        val_sigma_reg_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x_batch, y_batch, step_mask in val_loader:
                x_batch = x_batch.to(ctx.device)
                y_batch = y_batch.to(ctx.device)
                step_mask = step_mask.to(ctx.device)
                with torch.autocast(device_type=ctx.device.type, dtype=torch.float16, enabled=use_amp):
                    mu, log_sigma = model(x_batch)
                    nll_loss = gaussian_nll_loss(mu, log_sigma, y_batch, step_mask=step_mask)
                    sigma_per_step = torch.exp(2.0 * log_sigma).mean(dim=-1)
                    sigma_reg = masked_mean(sigma_per_step, step_mask=step_mask)
                bs = x_batch.shape[0]
                val_loss_sum += float(nll_loss.item()) * bs
                val_sigma_reg_sum += float(sigma_reg.item()) * bs
                val_count += bs

        train_loss_sum = allreduce_sum(train_loss_sum, ctx.device, ctx.use_ddp)
        train_sigma_reg_sum = allreduce_sum(train_sigma_reg_sum, ctx.device, ctx.use_ddp)
        val_loss_sum = allreduce_sum(val_loss_sum, ctx.device, ctx.use_ddp)
        val_sigma_reg_sum = allreduce_sum(val_sigma_reg_sum, ctx.device, ctx.use_ddp)
        train_count = int(allreduce_sum(float(train_count), ctx.device, ctx.use_ddp))
        val_count = int(allreduce_sum(float(val_count), ctx.device, ctx.use_ddp))

        train_loss = train_loss_sum / max(train_count, 1)
        train_sigma_reg = train_sigma_reg_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        val_sigma_reg = val_sigma_reg_sum / max(val_count, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_sigma_reg"].append(train_sigma_reg)
        history["val_sigma_reg"].append(val_sigma_reg)

        if is_main_process(ctx) and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            pct = 100.0 * float(epoch) / float(max(epochs, 1))
            print(
                f"[Epoch {epoch:03d}/{epochs}] ({pct:.1f}%) train_nll={train_loss:.5f} val_nll={val_loss:.5f} train_sigma={train_sigma_reg:.5f} val_sigma={val_sigma_reg:.5f}",
                flush=True,
            )

    return history
