#!/usr/bin/env bash
# Diffusion モデルの学習を実行する。
#
# 使い方:
#   scripts/train_diffusion.sh                              # mnist で学習 (デフォルト)
#   scripts/train_diffusion.sh --dataset cifar10            # cifar10 で学習 (data-dir も自動設定)
#   scripts/train_diffusion.sh --root-dir /path/to/out_root # 出力ルートを変更
#   scripts/train_diffusion.sh --epochs 50 --lr 1e-4        # 任意の引数を python へ pass-through
#
# 出力先は ${ROOT_DIR}/exp_NN として、空き番号を自動採番して作成する。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET=mnist
ROOT_DIR=/workspace/outputs/diffusion
PASS=()
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    case "${args[i]}" in
        --root-dir) ROOT_DIR="${args[i+1]}"; i=$((i + 2)) ;;
        --dataset)  DATASET="${args[i+1]}"; PASS+=("${args[i]}" "${args[i+1]}"); i=$((i + 2)) ;;
        *)          PASS+=("${args[i]}"); i=$((i + 1)) ;;
    esac
done
DATA_DIR="/workspace/datasets/${DATASET}"

mkdir -p "${ROOT_DIR}"
N=1
while [ -e "${ROOT_DIR}/$(printf 'exp_%02d' "${N}")" ]; do
    N=$((N + 1))
done
OUT_DIR="${ROOT_DIR}/$(printf 'exp_%02d' "${N}")"

python src/train_diffusion.py \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --epochs 20 \
    --batch-size 256 \
    --lr 2e-4 \
    --num-workers 8 \
    --base-channels 64 \
    --timesteps 1000 \
    --num-classes 10 \
    --p-uncond 0.1 \
    --seed 42 \
    --val-ratio 0.1 \
    ${PASS[@]+"${PASS[@]}"}
