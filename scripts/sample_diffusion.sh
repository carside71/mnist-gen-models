#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/sample_diffusion.sh
#
# 下の変数を編集すればデフォルト値を変更できます。
# 一時的に値を変えたいだけなら、環境変数として渡すこともできます。
#   例) NUM_SAMPLES=128 SEED=0 scripts/sample_diffusion.sh
#
# また、コマンドライン引数はそのまま python スクリプトへ転送されるので、
# 個別に上書きしたい場合は以下のように指定できます。
#   例) scripts/sample_diffusion.sh --num-samples 128 --seed 0
#
# BASE_CHANNELS / TIMESTEPS は未指定 (空文字) の場合は python 側のデフォルト
# (= チェックポイントから推定 / 学習時の値) が使われます。

CHECKPOINT="${CHECKPOINT:-/workspace/outputs/diffusion/checkpoints/best.pt}"
OUT_PATH="${OUT_PATH:-/workspace/outputs/diffusion/samples/diffusion_samples.png}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
BASE_CHANNELS="${BASE_CHANNELS:-}"
TIMESTEPS="${TIMESTEPS:-}"
SEED="${SEED:-42}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXTRA_ARGS=()
[[ -n "${BASE_CHANNELS}" ]] && EXTRA_ARGS+=(--base-channels "${BASE_CHANNELS}")
[[ -n "${TIMESTEPS}" ]] && EXTRA_ARGS+=(--timesteps "${TIMESTEPS}")

python "${PROJECT_ROOT}/src/sample_diffusion.py" \
    --checkpoint "${CHECKPOINT}" \
    --out-path "${OUT_PATH}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
