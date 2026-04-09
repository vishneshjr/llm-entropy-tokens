#!/usr/bin/env bash
# Train DAPO with entropy masking on multiple GPUs.
#
# Usage:
#   bash scripts/train_multigpu.sh              # fresh start, 2 GPUs
#   bash scripts/train_multigpu.sh --resume     # resume from latest checkpoint
#   NUM_GPUS=4 bash scripts/train_multigpu.sh   # use 4 GPUs
#
# After your first rental ends, your next rental just needs:
#   bash scripts/train_multigpu.sh --resume

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-2}"
MASK_MODE="${MASK_MODE:-topk}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MASK_MODE}}"

# Parse --resume flag
RESUME_FLAG=""
for arg in "$@"; do
    if [ "$arg" = "--resume" ]; then
        RESUME_FLAG="--resume_from_checkpoint latest"
    fi
done

echo "=== DAPO Training ==="
echo "GPUs:       ${NUM_GPUS}"
echo "Mask mode:  ${MASK_MODE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Resume:     ${RESUME_FLAG:-no}"
echo "====================="

# Dynamically set num_processes in the accelerate config
accelerate launch \
    --multi_gpu \
    --mixed_precision bf16 \
    --num_processes "${NUM_GPUS}" \
    -m src.train_dapo \
    --mask_mode "${MASK_MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --save_steps 25 \
    --save_total_limit 3 \
    ${RESUME_FLAG}
