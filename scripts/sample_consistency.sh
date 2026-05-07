#!/usr/bin/env bash
# 蒸留済み Consistency Model からサンプリングを行う。
#
# 使い方:
#   scripts/sample_consistency.sh                                  # ROOT_DIR の checkpoint を使用
#   scripts/sample_consistency.sh --root-dir /workspace/.../exp_03 # 実験ディレクトリ指定
#   scripts/sample_consistency.sh --dataset cifar10                # データセット指定
#   scripts/sample_consistency.sh --steps 4 --num-samples 128      # 任意引数 pass-through
#
# checkpoint は ${ROOT_DIR}/checkpoints/best.pt、出力は ${ROOT_DIR}/samples/consistency_samples.png。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT_DIR=/workspace/outputs/consistency/exp_01
DATASET=mnist
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

python src/sample_consistency.py \
    --checkpoint "${ROOT_DIR}/checkpoints/best.pt" \
    --out-path "${ROOT_DIR}/samples/consistency_samples.png" \
    --dataset "${DATASET}" \
    --num-samples 64 \
    --steps 1 \
    --seed 42 \
    ${PASS[@]+"${PASS[@]}"}
