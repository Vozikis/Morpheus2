from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import parse_args
from .data_generation import generate_nonphysical_dataset, generate_physical_dataset
from .dataset import TrajectoryDataset, compute_norm_stats, normalize_trajectories
from .distributed import cleanup_distributed, init_distributed, is_main_process
from .metrics import cohen_d, compute_auroc, summarize_prediction_errors, summarize_scores
from .model import CausalTrajectoryTransformer
from .runtime import ensure_dirs, log_progress, set_seed
from .scoring import evaluate_prediction_quality_subset, score_single_trajectory
from .training import train_model
from .visualization import save_overlay_gif, save_overlay_png


def main() -> None:
    run_start = time.time()
    args = parse_args()
    ctx = init_distributed(args)
    set_seed(args.seed)
    device = ctx.device
    main_proc = is_main_process(ctx)
    output_dir = Path(args.output_dir).resolve()
    paths = ensure_dirs(output_dir, save_pngs=bool(args.save_pngs), save_gifs=bool(args.save_gifs))
    rng = np.random.default_rng(args.seed)

    if main_proc:
        print("[STAGE 1/9] Setup", flush=True)
        print(f"[INFO] Device: {device}", flush=True)
        print(f"[INFO] Output directory: {output_dir}", flush=True)
        print(
            "[INFO] Config | x={} train_n={} val_n={} seq_len={} dt={} epochs={} batch_size={} lr={} sigma_reg_weight={} d_model={} nhead={} num_layers={} context_window={} log_sigma_min={} log_sigma_max={} surprise_mode={} amp={} multi_gpu={} world_size={} save_pngs={} save_gifs={} max_viz_per_group={} oom_retries={} min_batch_size={}".format(
                args.x,
                args.train_n,
                args.val_n,
                args.seq_len,
                args.dt,
                args.epochs,
                args.batch_size,
                args.lr,
                args.sigma_reg_weight,
                args.d_model,
                args.nhead,
                args.num_layers,
                args.context_window,
                args.log_sigma_min,
                args.log_sigma_max,
                args.surprise_mode,
                args.amp,
                args.multi_gpu,
                ctx.world_size,
                args.save_pngs,
                args.save_gifs,
                args.max_visualizations_per_group,
                args.oom_retries,
                args.min_batch_size,
            ),
            flush=True,
        )

    if main_proc:
        print("[STAGE 2/9] Generating datasets", flush=True)
        print("[INFO] Generating in-distribution physical training trajectories...", flush=True)
    train_traj, _ = generate_physical_dataset(
        args.train_n,
        args.seq_len,
        args.dt,
        mode="train",
        rng=rng,
        progress_label="generate_train_physical" if main_proc else None,
        return_metadata=False,
    )
    if main_proc:
        print("[INFO] Generating in-distribution physical validation trajectories...", flush=True)
    val_traj, _ = generate_physical_dataset(
        args.val_n,
        args.seq_len,
        args.dt,
        mode="train",
        rng=rng,
        progress_label="generate_val_physical" if main_proc else None,
        return_metadata=False,
    )
    if main_proc:
        print("[INFO] Generating in-distribution evaluation trajectories...", flush=True)
    id_eval_traj, id_eval_meta = generate_physical_dataset(
        args.x,
        args.seq_len,
        args.dt,
        mode="train",
        rng=rng,
        progress_label="generate_eval_in_distribution" if main_proc else None,
        return_metadata=True,
    )
    if main_proc:
        print("[INFO] Generating physical out-of-distribution evaluation trajectories...", flush=True)
    phys_ood_traj, phys_ood_meta = generate_physical_dataset(
        args.x,
        args.seq_len,
        args.dt,
        mode="ood",
        rng=rng,
        progress_label="generate_eval_physical_ood" if main_proc else None,
        return_metadata=True,
    )
    if main_proc:
        print("[INFO] Generating non-physical evaluation trajectories...", flush=True)
    nonphys_traj, nonphys_meta = generate_nonphysical_dataset(
        args.x,
        args.seq_len,
        args.dt,
        rng=rng,
        progress_label="generate_eval_non_physical" if main_proc else None,
    )

    if main_proc:
        print("[STAGE 3/9] Normalization and data loaders", flush=True)
    stats = compute_norm_stats(train_traj)
    train_norm = normalize_trajectories(train_traj, stats)
    val_norm = normalize_trajectories(val_traj, stats)

    train_ds = TrajectoryDataset(train_norm, floor=0.0, mask_source_trajectories=train_traj)
    val_ds = TrajectoryDataset(val_norm, floor=0.0, mask_source_trajectories=val_traj)
    if main_proc:
        print(
            f"[INFO] Dataset sizes | train={len(train_ds)} val={len(val_ds)} eval_total={3 * args.x}",
            flush=True,
        )

    if main_proc:
        print("[STAGE 4/9] Model initialization", flush=True)

    def build_model() -> CausalTrajectoryTransformer:
        return CausalTrajectoryTransformer(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_seq_len=args.seq_len - 1,
            context_window=args.context_window,
            log_sigma_min=args.log_sigma_min,
            log_sigma_max=args.log_sigma_max,
        ).to(device)

    def build_loaders(batch_size: int) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
        train_sampler: Optional[DistributedSampler] = None
        val_sampler: Optional[DistributedSampler] = None
        if ctx.use_ddp:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=ctx.world_size,
                rank=ctx.rank,
                shuffle=False,
                seed=args.seed,
                drop_last=False,
            )
        loader_gen = torch.Generator().manual_seed(args.seed)
        train_loader_local = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=False,
            generator=loader_gen,
            num_workers=0,
        )
        val_loader_local = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            drop_last=False,
            num_workers=0,
        )
        return train_loader_local, val_loader_local, train_sampler

    if main_proc:
        print("[STAGE 5/9] Training transformer on physical-only trajectories", flush=True)
    current_batch_size = int(args.batch_size)
    model: Optional[nn.Module] = None
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    oom_retries_used = 0

    while True:
        model = build_model()
        if ctx.use_ddp:
            model = DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank, find_unused_parameters=False)
        train_loader, val_loader, train_sampler = build_loaders(current_batch_size)
        if main_proc:
            print(
                f"[INFO] Training attempt {oom_retries_used + 1} | batch_size={current_batch_size}",
                flush=True,
            )
        try:
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                train_sampler=train_sampler,
                ctx=ctx,
                epochs=args.epochs,
                lr=args.lr,
                amp=bool(args.amp),
                sigma_reg_weight=float(args.sigma_reg_weight),
            )
            break
        except RuntimeError as exc:
            msg = str(exc).lower()
            is_oom = "out of memory" in msg or "cuda error: out of memory" in msg
            can_retry = (
                device.type == "cuda"
                and not ctx.use_ddp
                and is_oom
                and oom_retries_used < args.oom_retries
                and current_batch_size > args.min_batch_size
            )
            if not can_retry:
                raise
            next_batch_size = max(args.min_batch_size, current_batch_size // 2)
            if next_batch_size == current_batch_size:
                raise
            oom_retries_used += 1
            if main_proc:
                print(
                    f"[WARN] CUDA OOM detected. Retrying with smaller batch size: {current_batch_size} -> {next_batch_size}",
                    flush=True,
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current_batch_size = next_batch_size

    if main_proc:
        print(
            f"[INFO] Training completed with effective_batch_size={current_batch_size} oom_retries_used={oom_retries_used}",
            flush=True,
        )

    if main_proc:
        print("[STAGE 6/9] Saving checkpoint (if enabled)", flush=True)
    if model is None:
        raise RuntimeError("Model was not initialized successfully.")
    model_for_eval: nn.Module = model.module if isinstance(model, DDP) else model
    if main_proc and args.save_checkpoint == 1:
        ckpt_path = paths["checkpoints"] / "model.pt"
        torch.save(
            {
                "state_dict": model_for_eval.state_dict(),
                "norm_mean": stats.mean.tolist(),
                "norm_std": stats.std.tolist(),
                "args": vars(args),
                "effective_batch_size": current_batch_size,
                "oom_retries_used": oom_retries_used,
                "train_history": history,
            },
            ckpt_path,
        )
        print(f"[INFO] Saved checkpoint: {ckpt_path}", flush=True)
    elif main_proc:
        print("[INFO] Checkpoint saving disabled.", flush=True)

    if ctx.use_ddp:
        dist.barrier()
    if not main_proc:
        cleanup_distributed(ctx)
        return

    print("[STAGE 7/9] Sanity check: prediction quality on in-distribution validation subset", flush=True)
    val_quality = evaluate_prediction_quality_subset(
        model=model_for_eval,
        trajectories=val_traj,
        stats=stats,
        device=device,
        max_samples=args.quality_eval_samples,
        label="id_val_prediction_quality",
        scoring_mode=args.surprise_mode,
    )
    print(
        "[INFO] ID validation quality | samples={} rmse_pos={:.4f} rmse_vel={:.4f} mae_pos={:.4f} mae_vel={:.4f}".format(
            val_quality["sample_count"],
            val_quality["rmse_pos_mean"],
            val_quality["rmse_vel_mean"],
            val_quality["mae_pos_mean"],
            val_quality["mae_vel_mean"],
        ),
        flush=True,
    )

    print("[STAGE 8/9] Scoring + rendering trajectories", flush=True)
    if args.save_pngs == 0:
        print("[INFO] PNG rendering disabled (--save_pngs=0).", flush=True)
    if args.save_gifs == 0:
        print("[INFO] GIF rendering disabled (--save_gifs=0). PNG overlays will still be generated.", flush=True)
    records: List[Dict[str, object]] = []
    id_scores: List[float] = []
    physical_scores: List[float] = []
    nonphysical_scores: List[float] = []
    max_viz = int(args.x if args.max_visualizations_per_group == 0 else min(args.max_visualizations_per_group, args.x))
    print(
        f"[INFO] Visualization plan | per_group={max_viz} (x={args.x}, cap_arg={args.max_visualizations_per_group})",
        flush=True,
    )
    id_png_rendered = 0
    id_gif_rendered = 0
    physical_png_rendered = 0
    physical_gif_rendered = 0
    nonphysical_png_rendered = 0
    nonphysical_gif_rendered = 0
    id_viz_skip_notified = False
    physical_viz_skip_notified = False
    nonphysical_viz_skip_notified = False
    eval_total = 3 * args.x
    eval_done = 0
    eval_progress_stride = max(1, eval_total // 10)

    # In-distribution group
    for i in range(args.x):
        res = score_single_trajectory(model_for_eval, id_eval_traj[i], stats, device, mode=args.surprise_mode)
        score = float(res["surprise_mean_nll"])
        id_scores.append(score)
        traj_id = f"in_distribution_{i:04d}"
        png_path = paths["plots_id"] / f"traj_{i:04d}.png"
        gif_path = paths["gifs_id"] / f"traj_{i:04d}.gif"
        title = f"In-Distribution | {traj_id} | mean_nll={score:.4f}"
        should_render = i < max_viz
        if should_render:
            if args.save_pngs == 1:
                save_overlay_png(png_path, res["time"], res["actual"], res["pred"], title)
                id_png_rendered += 1
            if args.save_gifs == 1:
                try:
                    save_overlay_gif(
                        gif_path,
                        res["time"],
                        res["actual"],
                        res["pred"],
                        title,
                        frame_stride=args.gif_stride,
                    )
                    id_gif_rendered += 1
                except Exception as exc:
                    print(
                        f"[WARN] GIF render failed for {traj_id}: {exc}. Continuing without this GIF.",
                        flush=True,
                    )
        elif not id_viz_skip_notified:
            print(
                f"[WARN] Visualization cap reached for in_distribution group (max={max_viz}). Remaining trajectories will be scored but not rendered.",
                flush=True,
            )
            id_viz_skip_notified = True

        meta = dict(id_eval_meta[i])
        meta["group"] = "in_distribution"
        meta["is_physical"] = 1
        records.append(
            {
                "trajectory_id": traj_id,
                "group": "in_distribution",
                "surprise_mean_nll": score,
                "surprise_std_nll": float(res["surprise_std_nll"]),
                "rmse_pos": float(res["rmse_pos"]),
                "rmse_vel": float(res["rmse_vel"]),
                "mae_pos": float(res["mae_pos"]),
                "mae_vel": float(res["mae_vel"]),
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )
        eval_done += 1
        if eval_done == 1 or eval_done % eval_progress_stride == 0 or eval_done == eval_total:
            log_progress("score_and_render", eval_done, eval_total)
            print(f"[INFO] Latest complete: {traj_id} | score={score:.5f}", flush=True)

    # Physical OOD group
    for i in range(args.x):
        res = score_single_trajectory(
            model_for_eval, phys_ood_traj[i], stats, device, mode=args.surprise_mode
        )
        score = float(res["surprise_mean_nll"])
        physical_scores.append(score)
        traj_id = f"physical_ood_{i:04d}"
        png_path = paths["plots_physical"] / f"traj_{i:04d}.png"
        gif_path = paths["gifs_physical"] / f"traj_{i:04d}.gif"
        title = f"Physical OOD | {traj_id} | mean_nll={score:.4f}"
        should_render = i < max_viz
        if should_render:
            if args.save_pngs == 1:
                save_overlay_png(png_path, res["time"], res["actual"], res["pred"], title)
                physical_png_rendered += 1
            if args.save_gifs == 1:
                try:
                    save_overlay_gif(
                        gif_path,
                        res["time"],
                        res["actual"],
                        res["pred"],
                        title,
                        frame_stride=args.gif_stride,
                    )
                    physical_gif_rendered += 1
                except Exception as exc:
                    print(
                        f"[WARN] GIF render failed for {traj_id}: {exc}. Continuing without this GIF.",
                        flush=True,
                    )
        elif not physical_viz_skip_notified:
            print(
                f"[WARN] Visualization cap reached for physical_ood group (max={max_viz}). Remaining trajectories will be scored but not rendered.",
                flush=True,
            )
            physical_viz_skip_notified = True

        meta = dict(phys_ood_meta[i])
        meta["group"] = "physical_ood"
        meta["is_physical"] = 1
        records.append(
            {
                "trajectory_id": traj_id,
                "group": "physical_ood",
                "surprise_mean_nll": score,
                "surprise_std_nll": float(res["surprise_std_nll"]),
                "rmse_pos": float(res["rmse_pos"]),
                "rmse_vel": float(res["rmse_vel"]),
                "mae_pos": float(res["mae_pos"]),
                "mae_vel": float(res["mae_vel"]),
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )
        eval_done += 1
        if eval_done == 1 or eval_done % eval_progress_stride == 0 or eval_done == eval_total:
            log_progress("score_and_render", eval_done, eval_total)
            print(f"[INFO] Latest complete: {traj_id} | score={score:.5f}", flush=True)

    # Non-physical group
    for i in range(args.x):
        res = score_single_trajectory(
            model_for_eval, nonphys_traj[i], stats, device, mode=args.surprise_mode
        )
        score = float(res["surprise_mean_nll"])
        nonphysical_scores.append(score)
        traj_id = f"non_physical_{i:04d}"
        png_path = paths["plots_nonphysical"] / f"traj_{i:04d}.png"
        gif_path = paths["gifs_nonphysical"] / f"traj_{i:04d}.gif"
        title = f"Non-Physical | {traj_id} | mean_nll={score:.4f}"
        should_render = i < max_viz
        if should_render:
            if args.save_pngs == 1:
                save_overlay_png(png_path, res["time"], res["actual"], res["pred"], title)
                nonphysical_png_rendered += 1
            if args.save_gifs == 1:
                try:
                    save_overlay_gif(
                        gif_path,
                        res["time"],
                        res["actual"],
                        res["pred"],
                        title,
                        frame_stride=args.gif_stride,
                    )
                    nonphysical_gif_rendered += 1
                except Exception as exc:
                    print(
                        f"[WARN] GIF render failed for {traj_id}: {exc}. Continuing without this GIF.",
                        flush=True,
                    )
        elif not nonphysical_viz_skip_notified:
            print(
                f"[WARN] Visualization cap reached for non_physical group (max={max_viz}). Remaining trajectories will be scored but not rendered.",
                flush=True,
            )
            nonphysical_viz_skip_notified = True

        meta = dict(nonphys_meta[i])
        meta["group"] = "non_physical"
        records.append(
            {
                "trajectory_id": traj_id,
                "group": "non_physical",
                "surprise_mean_nll": score,
                "surprise_std_nll": float(res["surprise_std_nll"]),
                "rmse_pos": float(res["rmse_pos"]),
                "rmse_vel": float(res["rmse_vel"]),
                "mae_pos": float(res["mae_pos"]),
                "mae_vel": float(res["mae_vel"]),
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )
        eval_done += 1
        if eval_done == 1 or eval_done % eval_progress_stride == 0 or eval_done == eval_total:
            log_progress("score_and_render", eval_done, eval_total)
            print(f"[INFO] Latest complete: {traj_id} | score={score:.5f}", flush=True)

    print("[STAGE 9/9] Aggregating metrics + writing artifacts", flush=True)
    df = pd.DataFrame.from_records(records)
    csv_path = paths["metrics"] / "per_trajectory_scores.csv"
    df.to_csv(csv_path, index=False)
    score_txt_path = paths["metrics"] / "per_trajectory_scores.txt"
    with score_txt_path.open("w", encoding="utf-8") as f:
        for group in ("in_distribution", "physical_ood", "non_physical"):
            group_df = df[df["group"] == group].sort_values("trajectory_id")
            header = f"[SCORES] group={group} count={len(group_df)}"
            print(header, flush=True)
            f.write(header + "\n")
            for row in group_df.itertuples(index=False):
                line = (
                    f"[SCORE] trajectory_id={row.trajectory_id} "
                    f"group={row.group} "
                    f"surprise_mean_nll={float(row.surprise_mean_nll):.6f} "
                    f"surprise_std_nll={float(row.surprise_std_nll):.6f}"
                )
                print(line, flush=True)
                f.write(line + "\n")

    id_scores_np = np.asarray(id_scores, dtype=np.float64)
    physical_scores_np = np.asarray(physical_scores, dtype=np.float64)
    nonphysical_scores_np = np.asarray(nonphysical_scores, dtype=np.float64)
    auroc = compute_auroc(neg_scores=physical_scores_np, pos_scores=nonphysical_scores_np)

    summary = {
        "config": vars(args),
        "device_used": str(device),
        "counts": {
            "in_distribution": int(args.x),
            "physical_ood": int(args.x),
            "non_physical": int(args.x),
            "total_eval": int(3 * args.x),
        },
        "train_history": history,
        "runtime_safety": {
            "effective_batch_size": int(current_batch_size),
            "oom_retries_used": int(oom_retries_used),
            "max_visualizations_per_group": int(max_viz),
        },
        "prediction_quality": {
            "id_validation_subset": val_quality,
            "in_distribution": summarize_prediction_errors(records, "in_distribution"),
            "physical_ood": summarize_prediction_errors(records, "physical_ood"),
            "non_physical": summarize_prediction_errors(records, "non_physical"),
        },
        "surprise_stats": {
            "in_distribution": summarize_scores(id_scores_np),
            "physical_ood": summarize_scores(physical_scores_np),
            "non_physical": summarize_scores(nonphysical_scores_np),
            "delta_mean_physical_ood_minus_in_distribution": float(
                np.mean(physical_scores_np) - np.mean(id_scores_np)
            ),
            "delta_mean_nonphysical_minus_physical": float(
                np.mean(nonphysical_scores_np) - np.mean(physical_scores_np)
            ),
            "delta_mean_nonphysical_minus_in_distribution": float(
                np.mean(nonphysical_scores_np) - np.mean(id_scores_np)
            ),
            "delta_median_nonphysical_minus_physical": float(
                np.median(nonphysical_scores_np) - np.median(physical_scores_np)
            ),
            "effect_size_cohen_d": cohen_d(physical_scores_np, nonphysical_scores_np),
            "auroc_nonphysical_as_positive": auroc,
        },
        "artifacts": {
            "per_trajectory_scores_csv": str(csv_path),
            "per_trajectory_scores_txt": str(score_txt_path),
            "plots_dir": str(output_dir / "plots") if args.save_pngs == 1 else None,
            "gifs_dir": str(output_dir / "gifs") if args.save_gifs == 1 else None,
            "rendered_png_count_in_distribution": int(id_png_rendered),
            "rendered_png_count_physical_ood": int(physical_png_rendered),
            "rendered_png_count_non_physical": int(nonphysical_png_rendered),
            "rendered_gif_count_in_distribution": int(id_gif_rendered),
            "rendered_gif_count_physical_ood": int(physical_gif_rendered),
            "rendered_gif_count_non_physical": int(nonphysical_gif_rendered),
            "checkpoint": str(paths["checkpoints"] / "model.pt") if args.save_checkpoint == 1 else None,
        },
    }

    summary_path = paths["metrics"] / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed_sec = time.time() - run_start
    print(f"[INFO] Saved metrics: {summary_path}", flush=True)
    print(f"[INFO] Saved per-trajectory CSV: {csv_path}", flush=True)
    print(f"[INFO] Saved per-trajectory score text: {score_txt_path}", flush=True)
    print(
        "[INFO] Surprise means | in_distribution={:.5f} | physical_ood={:.5f} | non_physical={:.5f} | delta(non_physical-physical_ood)={:.5f}".format(
            float(np.mean(id_scores_np)),
            float(np.mean(physical_scores_np)),
            float(np.mean(nonphysical_scores_np)),
            float(np.mean(nonphysical_scores_np) - np.mean(physical_scores_np)),
        ),
        flush=True,
    )
    print(f"[INFO] AUROC(non_physical positive)={auroc:.5f}", flush=True)
    print(f"[INFO] Done. Total runtime: {elapsed_sec:.1f}s", flush=True)
    cleanup_distributed(ctx)
