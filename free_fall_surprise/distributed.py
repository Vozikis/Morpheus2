from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .runtime import resolve_device


@dataclass
class DistributedContext:
    use_ddp: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def is_main_process(ctx: DistributedContext) -> bool:
    return ctx.rank == 0


def init_distributed(args: argparse.Namespace) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = bool(args.multi_gpu == 1 and world_size > 1)

    if use_ddp:
        if args.device == "cpu":
            raise RuntimeError("DDP multi-GPU requires CUDA device mode.")
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=args.ddp_backend, init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = resolve_device(args.device)

    return DistributedContext(
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.use_ddp and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def allreduce_sum(value: float, device: torch.device, enabled: bool) -> float:
    if not enabled:
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())
