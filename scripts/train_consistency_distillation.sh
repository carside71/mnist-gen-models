#!/usr/bin/env bash
# 教師 Diffusion から Consistency Model を蒸留する。
#
# 使い方:
#   scripts/train_consistency_distillation.sh                                     # mnist で蒸留
#   scripts/train_consistency_distillation.sh --dataset cifar10                   # cifar10 で蒸留
#   scripts/train_consistency_distillation.sh --teacher-root /workspace/.../exp_01 # 教師の実験ディレクトリ
#   scripts/train_consistency_distillation.sh --root-dir /path/to/out_root        # 出力ルート変更
#   scripts/train_consistency_distillation.sh --epochs 20 --num-steps 40          # 任意引数 pass-through
#
# 出力先は ${ROOT_DIR}/exp_NN として、空き番号を自動採番して作成する。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DATASET=mnist
ROOT_DIR=/workspace/outputs/consistency
TEACHER_ROOT=/workspace/outputs/edm/exp_01
PASS=()
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    case "${args[i]}" in
        --root-dir)       ROOT_DIR="${args[i+1]}"; i=$((i + 2)) ;;
        --teacher-root)   TEACHER_ROOT="${args[i+1]}"; i=$((i + 2)) ;;
        --dataset)        DATASET="${args[i+1]}"; PASS+=("${args[i]}" "${args[i+1]}"); i=$((i + 2)) ;;
        *)                PASS+=("${args[i]}"); i=$((i + 1)) ;;
    esac
done
DATA_DIR="/workspace/datasets/${DATASET}"

mkdir -p "${ROOT_DIR}"
N=1
while [ -e "${ROOT_DIR}/$(printf 'exp_%02d' "${N}")" ]; do
    N=$((N + 1))
done
OUT_DIR="${ROOT_DIR}/$(printf 'exp_%02d' "${N}")"

python src/train_consistency_distillation.py \
    --teacher-checkpoint "${TEACHER_ROOT}/checkpoints/best.pt" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --epochs 10 \
    --batch-size 256 \
    --lr 1e-4 \
    --num-workers 8 \
    --num-steps 18 \
    --ema-decay 0.999 \
    --loss mse \
    --seed 42 \
    --val-ratio 0.1 \
    ${PASS[@]+"${PASS[@]}"}
