from __future__ import annotations

import argparse

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency for GIF output only
    imageio = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Free-fall violation-of-expectation with transformer surprise scoring."
    )
    parser.add_argument("--x", type=int, default=10, help="Eval trajectories per class.")
    parser.add_argument(
        "--train_n", type=int, default=5000, help="Number of physical train trajectories."
    )
    parser.add_argument(
        "--val_n", type=int, default=1000, help="Number of physical validation trajectories."
    )
    parser.add_argument("--seq_len", type=int, default=80, help="Sequence length.")
    parser.add_argument("--dt", type=float, default=0.05, help="Timestep size.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--min_batch_size",
        type=int,
        default=8,
        help="Minimum batch size allowed when auto-retrying after CUDA OOM.",
    )
    parser.add_argument(
        "--oom_retries",
        type=int,
        default=4,
        help="Maximum number of CUDA OOM retries with reduced batch size.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--sigma_reg_weight",
        type=float,
        default=1e-3,
        help="Regularization weight to penalize overly large predictive variance.",
    )
    parser.add_argument("--d_model", type=int, default=128, help="Transformer d_model.")
    parser.add_argument("--nhead", type=int, default=8, help="Attention heads.")
    parser.add_argument("--num_layers", type=int, default=4, help="Encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument(
        "--log_sigma_min",
        type=float,
        default=-6.0,
        help="Lower clamp bound for predicted log_sigma.",
    )
    parser.add_argument(
        "--log_sigma_max",
        type=float,
        default=1.5,
        help="Upper clamp bound for predicted log_sigma.",
    )
    parser.add_argument(
        "--amp",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable AMP mixed precision on CUDA.",
    )
    parser.add_argument(
        "--multi_gpu",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable DDP multi-GPU when launched with torchrun and WORLD_SIZE>1.",
    )
    parser.add_argument(
        "--ddp_backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend used for DDP.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/free_fall_surprise",
        help="Output root directory.",
    )
    parser.add_argument(
        "--save_checkpoint",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to save model checkpoint.",
    )
    parser.add_argument(
        "--save_gifs",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to generate per-trajectory GIF overlays.",
    )
    parser.add_argument(
        "--save_pngs",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to generate per-trajectory PNG overlays.",
    )
    parser.add_argument(
        "--gif_stride",
        type=int,
        default=1,
        help="Frame stride for GIF rendering (higher is faster/smaller).",
    )
    parser.add_argument(
        "--quality_eval_samples",
        type=int,
        default=256,
        help="Validation samples for prediction-quality sanity check.",
    )
    parser.add_argument(
        "--max_visualizations_per_group",
        type=int,
        default=0,
        help="Safety cap on PNG/GIF renders per evaluation group. 0 means auto (= x).",
    )
    parser.add_argument(
        "--surprise_mode",
        type=str,
        default="teacher_forced",
        choices=["rollout", "teacher_forced"],
        help="How surprise is computed: open-loop rollout or teacher-forced one-step.",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=-1,
        help="Max self-attention lookback (in steps). -1 uses full history; 5 means each step sees t-5..t only.",
    )
    args = parser.parse_args()
    if args.x <= 0:
        raise ValueError("--x must be > 0")
    if args.train_n <= 0 or args.val_n <= 0:
        raise ValueError("--train_n and --val_n must be > 0")
    if args.seq_len < 4:
        raise ValueError("--seq_len must be >= 4")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")
    if args.min_batch_size <= 0:
        raise ValueError("--min_batch_size must be > 0")
    if args.min_batch_size > args.batch_size:
        raise ValueError("--min_batch_size cannot be larger than --batch_size")
    if args.oom_retries < 0:
        raise ValueError("--oom_retries must be >= 0")
    if args.sigma_reg_weight < 0.0:
        raise ValueError("--sigma_reg_weight must be >= 0")
    if args.d_model % args.nhead != 0:
        raise ValueError("--d_model must be divisible by --nhead")
    if args.log_sigma_max <= args.log_sigma_min:
        raise ValueError("--log_sigma_max must be > --log_sigma_min")
    if args.gif_stride <= 0:
        raise ValueError("--gif_stride must be > 0")
    if args.quality_eval_samples <= 0:
        raise ValueError("--quality_eval_samples must be > 0")
    if args.max_visualizations_per_group < 0:
        raise ValueError("--max_visualizations_per_group must be >= 0")
    if args.context_window < -1:
        raise ValueError("--context_window must be >= -1")
    if args.save_gifs == 1 and imageio is None:
        raise RuntimeError("GIF saving requested but imageio is not installed.")
    return args
