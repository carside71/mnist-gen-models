#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/train_diffusion.sh
#
# 下の変数を編集すればデフォルト値を変更できます。
# 一時的に値を変えたいだけなら、環境変数として渡すこともできます。
#   例) EPOCHS=50 BATCH_SIZE=128 scripts/train_diffusion.sh
#
# また、コマンドライン引数はそのまま python スクリプトへ転送されるので、
# 個別に上書きしたい場合は以下のように指定できます。
#   例) scripts/train_diffusion.sh --epochs 50 --lr 1e-4

DATA_DIR="${DATA_DIR:-/workspace/datasets/mnist}"
OUT_DIR="${OUT_DIR:-/workspace/outputs/diffusion}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-2e-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
TIMESTEPS="${TIMESTEPS:-1000}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python "${PROJECT_ROOT}/src/train_diffusion.py" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --num-workers "${NUM_WORKERS}" \
    --base-channels "${BASE_CHANNELS}" \
    --timesteps "${TIMESTEPS}" \
    --seed "${SEED}" \
    --val-ratio "${VAL_RATIO}" \
    "$@"
