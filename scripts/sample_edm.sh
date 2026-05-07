#!/usr/bin/env bash
# 学習済み EDM モデルからサンプリングを行う。
#
# 使い方:
#   scripts/sample_edm.sh                                  # ROOT_DIR のチェックポイントを使用
#   scripts/sample_edm.sh --root-dir /workspace/.../exp_03 # 実験ディレクトリを指定
#   scripts/sample_edm.sh --dataset cifar10                # データセット指定
#   scripts/sample_edm.sh --num-samples 128 --seed 0       # 任意の引数を pass-through
#
# checkpoint は ${ROOT_DIR}/checkpoints/best.pt、出力は ${ROOT_DIR}/samples/edm_samples.png。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT_DIR=/workspace/outputs/edm/exp_01
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

python src/sample_edm.py \
    --checkpoint "${ROOT_DIR}/checkpoints/best.pt" \
    --out-path "${ROOT_DIR}/samples/edm_samples.png" \
    --dataset "${DATASET}" \
    --num-samples 64 \
    --num-steps 18 \
    --guidance-scale 0.0 \
    --seed 42 \
    ${PASS[@]+"${PASS[@]}"}
