#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/visualize_diffusion_trajectories.sh
#
# 下の変数を編集すればデフォルト値を変更できます。
# 一時的に値を変えたいだけなら、環境変数として渡すこともできます。
#   例) NUM_GEN=32 NUM_DATA=32 STEPS=100 scripts/visualize_diffusion_trajectories.sh
#
# 対象ラベルを絞りたい場合は LABELS にスペース区切りで指定します。
#   例) LABELS="1 3 5" scripts/visualize_diffusion_trajectories.sh
#
# コマンドライン引数はそのまま python スクリプトへ転送されます。
#   例) scripts/visualize_diffusion_trajectories.sh --dim 3 --labels 1 3 5

CHECKPOINT="${CHECKPOINT:-/workspace/outputs/diffusion/exp_02/checkpoints/best.pt}"
OUT_PATH="${OUT_PATH:-/workspace/outputs/diffusion/trajectories/diffusion_traj.png}"
DATASET="${DATASET:-mnist}"
if [ "${DATASET}" = "cifar10" ]; then
    DATA_DIR="${DATA_DIR:-/workspace/datasets/cifar10}"
else
    DATA_DIR="${DATA_DIR:-/workspace/datasets/mnist}"
fi
NUM_GEN="${NUM_GEN:-16}"
NUM_DATA="${NUM_DATA:-16}"
STEPS="${STEPS:-50}"
TIMESTEPS="${TIMESTEPS:-1000}"
DIM="${DIM:-2}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
LABELS="${LABELS:-}"
BASE_CHANNELS="${BASE_CHANNELS:-}"
NUM_CLASSES="${NUM_CLASSES:-}"
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXTRA_ARGS=()
[[ -n "${BASE_CHANNELS}" ]] && EXTRA_ARGS+=(--base-channels "${BASE_CHANNELS}")
[[ -n "${NUM_CLASSES}" ]] && EXTRA_ARGS+=(--num-classes "${NUM_CLASSES}")
if [[ -n "${LABELS}" ]]; then
    # shellcheck disable=SC2206
    LABEL_ARR=(${LABELS})
    EXTRA_ARGS+=(--labels "${LABEL_ARR[@]}")
fi

python "${PROJECT_ROOT}/tools/visualize_diffusion_trajectories.py" \
    --checkpoint "${CHECKPOINT}" \
    --out-path "${OUT_PATH}" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --num-gen "${NUM_GEN}" \
    --num-data "${NUM_DATA}" \
    --steps "${STEPS}" \
    --timesteps "${TIMESTEPS}" \
    --dim "${DIM}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
