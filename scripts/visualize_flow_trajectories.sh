#!/usr/bin/env bash
# 学習済み Flow モデルの生成トラジェクトリを可視化する。
#
# 使い方:
#   scripts/visualize_flow_trajectories.sh                                  # ROOT_DIR を使用
#   scripts/visualize_flow_trajectories.sh --root-dir /workspace/.../exp_03 # 実験ディレクトリを指定
#   scripts/visualize_flow_trajectories.sh --dataset cifar10                # データセット指定 (data-dir も自動)
#   scripts/visualize_flow_trajectories.sh --dim 3 --labels 1 3             # 任意の引数を pass-through
#
# checkpoint は ${ROOT_DIR}/checkpoints/best.pt、出力は ${ROOT_DIR}/trajectories/flow_traj.png。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT_DIR=/workspace/outputs/flow/exp_01
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
DATA_DIR="/workspace/datasets/${DATASET}"

python tools/visualize_flow_trajectories.py \
    --checkpoint "${ROOT_DIR}/checkpoints/best.pt" \
    --out-path "${ROOT_DIR}/trajectories/flow_traj.png" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --num-gen 16 \
    --num-data 16 \
    --steps 50 \
    --dim 2 \
    --guidance-scale 0.0 \
    --seed 42 \
    ${PASS[@]+"${PASS[@]}"}
