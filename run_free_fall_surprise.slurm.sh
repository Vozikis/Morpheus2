#!/bin/bash
# Submit example:
# sbatch --account=<acct> --partition=<partition> run_free_fall_surprise.slurm.sh
#
# Note:
# - Account/partition are intentionally not hardcoded.
# - Pass them at submit time via sbatch flags.

#SBATCH --job-name=freefall_surprise
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/free_fall_surprise_transformer.py" ]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

# -----------------------------
# Conda setup
# -----------------------------
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-myenv}"

python - <<'PY'
import importlib.util
import torch

required = ["numpy", "torch", "pandas", "matplotlib"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"[ERROR] Missing required Python packages in active env: {missing}")
if importlib.util.find_spec("imageio") is None:
    print("[WARN] imageio not found: GIF rendering requires imageio (set SAVE_GIFS=0 otherwise).")
print(f"[INFO] Python preflight OK | torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

# -----------------------------
# Runtime safety (cluster-friendly defaults)
# -----------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -c 0

# -----------------------------
# User-tunable defaults
# -----------------------------
X="${X:-50}"
TRAIN_N="${TRAIN_N:-100000}"
VAL_N="${VAL_N:-20000}"
SEQ_LEN="${SEQ_LEN:-80}"
DT="${DT:-0.05}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MIN_BATCH_SIZE="${MIN_BATCH_SIZE:-16}"
OOM_RETRIES="${OOM_RETRIES:-4}"
LR="${LR:-5e-5}"
D_MODEL="${D_MODEL:-384}"
NHEAD="${NHEAD:-12}"
NUM_LAYERS="${NUM_LAYERS:-10}"
DROPOUT="${DROPOUT:-0.1}"
AMP="${AMP:-1}"                          # 0 | 1
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"                # auto | cpu | cuda
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}" # 0 | 1
SAVE_GIFS="${SAVE_GIFS:-0}"             # 0 | 1
GIF_STRIDE="${GIF_STRIDE:-2}"           # >=1 (used when SAVE_GIFS=1)
QUALITY_EVAL_SAMPLES="${QUALITY_EVAL_SAMPLES:-512}"
MAX_VIS_PER_GROUP="${MAX_VIS_PER_GROUP:-200}"
OUTPUT_PARENT="${OUTPUT_PARENT:-/ivi/zfs/s0/original_homes/${USER}/free_fall_surprise}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_PARENT}/runs}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}}"
RUN_DIR="${RUN_DIR:-${OUTPUT_DIR}/surprise_${RUN_ID}}"

if [ "$MIN_BATCH_SIZE" -gt "$BATCH_SIZE" ]; then
  echo "[WARN] MIN_BATCH_SIZE ($MIN_BATCH_SIZE) > BATCH_SIZE ($BATCH_SIZE). Clamping MIN_BATCH_SIZE to BATCH_SIZE."
  MIN_BATCH_SIZE="$BATCH_SIZE"
fi

mkdir -p "$OUTPUT_DIR" "$RUN_DIR"

# Save exact script snapshots used for this run.
cp "${SCRIPT_DIR}/free_fall_surprise_transformer.py" "${RUN_DIR}/free_fall_surprise_transformer.py"
cp "${SCRIPT_DIR}/run_free_fall_surprise.slurm.sh" "${RUN_DIR}/run_free_fall_surprise.slurm.sh"

echo "[INFO] Launching free_fall_surprise_transformer.py with:"
echo "  SCRIPT_DIR=$SCRIPT_DIR"
echo "  OUTPUT_PARENT=$OUTPUT_PARENT"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  RUN_ID=$RUN_ID"
echo "  RUN_DIR=$RUN_DIR"
echo "  X=$X TRAIN_N=$TRAIN_N VAL_N=$VAL_N SEQ_LEN=$SEQ_LEN DT=$DT"
echo "  EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE MIN_BATCH_SIZE=$MIN_BATCH_SIZE OOM_RETRIES=$OOM_RETRIES LR=$LR"
echo "  D_MODEL=$D_MODEL NHEAD=$NHEAD NUM_LAYERS=$NUM_LAYERS DROPOUT=$DROPOUT AMP=$AMP"
echo "  SEED=$SEED DEVICE=$DEVICE SAVE_CHECKPOINT=$SAVE_CHECKPOINT SAVE_GIFS=$SAVE_GIFS GIF_STRIDE=$GIF_STRIDE QUALITY_EVAL_SAMPLES=$QUALITY_EVAL_SAMPLES MAX_VIS_PER_GROUP=$MAX_VIS_PER_GROUP"
echo "  CONDA_ENV=${CONDA_ENV:-myenv}"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "  PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

python "${SCRIPT_DIR}/free_fall_surprise_transformer.py" \
  --x "$X" \
  --train_n "$TRAIN_N" \
  --val_n "$VAL_N" \
  --seq_len "$SEQ_LEN" \
  --dt "$DT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --min_batch_size "$MIN_BATCH_SIZE" \
  --oom_retries "$OOM_RETRIES" \
  --lr "$LR" \
  --d_model "$D_MODEL" \
  --nhead "$NHEAD" \
  --num_layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --amp "$AMP" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --output_dir "$RUN_DIR" \
  --save_checkpoint "$SAVE_CHECKPOINT" \
  --save_gifs "$SAVE_GIFS" \
  --gif_stride "$GIF_STRIDE" \
  --quality_eval_samples "$QUALITY_EVAL_SAMPLES" \
  --max_visualizations_per_group "$MAX_VIS_PER_GROUP"

echo "[INFO] Job completed."
