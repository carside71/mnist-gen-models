#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/sample_flow.sh
#
# 下の変数を編集すればデフォルト値を変更できます。
# 一時的に値を変えたいだけなら、環境変数として渡すこともできます。
#   例) NUM_SAMPLES=128 STEPS=200 scripts/sample_flow.sh
#
# また、コマンドライン引数はそのまま python スクリプトへ転送されるので、
# 個別に上書きしたい場合は以下のように指定できます。
#   例) scripts/sample_flow.sh --num-samples 128 --steps 200
#
# BASE_CHANNELS は未指定 (空文字) の場合は python 側のデフォルト
# (= チェックポイントから推定) が使われます。

CHECKPOINT="${CHECKPOINT:-/workspace/outputs/flow/checkpoints/best.pt}"
OUT_PATH="${OUT_PATH:-/workspace/outputs/flow/samples/flow_samples.png}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
STEPS="${STEPS:-100}"
BASE_CHANNELS="${BASE_CHANNELS:-}"
NUM_CLASSES="${NUM_CLASSES:-}"
DATASET="${DATASET:-}"
LABEL="${LABEL:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.0}"
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXTRA_ARGS=()
[[ -n "${BASE_CHANNELS}" ]] && EXTRA_ARGS+=(--base-channels "${BASE_CHANNELS}")
[[ -n "${NUM_CLASSES}" ]] && EXTRA_ARGS+=(--num-classes "${NUM_CLASSES}")
[[ -n "${DATASET}" ]] && EXTRA_ARGS+=(--dataset "${DATASET}")
[[ -n "${LABEL}" ]] && EXTRA_ARGS+=(--label "${LABEL}")

python "${PROJECT_ROOT}/src/sample_flow.py" \
    --checkpoint "${CHECKPOINT}" \
    --out-path "${OUT_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --steps "${STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
