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
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00

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

required = ["numpy", "torch", "pandas", "matplotlib", "imageio"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"[ERROR] Missing required Python packages in active env: {missing}")
print(f"[INFO] Python preflight OK | torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
PY

# -----------------------------
# User-tunable defaults
# -----------------------------
X="${X:-10}"
TRAIN_N="${TRAIN_N:-5000}"
VAL_N="${VAL_N:-1000}"
SEQ_LEN="${SEQ_LEN:-80}"
DT="${DT:-0.05}"
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-4}"
D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"                # auto | cpu | cuda
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-1}" # 0 | 1
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/free_fall_surprise}"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Launching free_fall_surprise_transformer.py with:"
echo "  SCRIPT_DIR=$SCRIPT_DIR"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  X=$X TRAIN_N=$TRAIN_N VAL_N=$VAL_N SEQ_LEN=$SEQ_LEN DT=$DT"
echo "  EPOCHS=$EPOCHS BATCH_SIZE=$BATCH_SIZE LR=$LR"
echo "  D_MODEL=$D_MODEL NHEAD=$NHEAD NUM_LAYERS=$NUM_LAYERS DROPOUT=$DROPOUT"
echo "  SEED=$SEED DEVICE=$DEVICE SAVE_CHECKPOINT=$SAVE_CHECKPOINT"
echo "  CONDA_ENV=${CONDA_ENV:-myenv}"

python "${SCRIPT_DIR}/free_fall_surprise_transformer.py" \
  --x "$X" \
  --train_n "$TRAIN_N" \
  --val_n "$VAL_N" \
  --seq_len "$SEQ_LEN" \
  --dt "$DT" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --d_model "$D_MODEL" \
  --nhead "$NHEAD" \
  --num_layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --output_dir "$OUTPUT_DIR" \
  --save_checkpoint "$SAVE_CHECKPOINT"

echo "[INFO] Job completed."
