#!/usr/bin/env bash
# Evaluate a VOC core-set trained model on the full VOC test split.
#
# Usage:
#   bash scripts/voc/eval_coreset_voc.sh [feature_mode] [selection_algo] [density_filter]
#
# Arguments must match the train_coreset_voc.sh script.

set -euo pipefail

FEATURE_MODE="${1:-classwise}"
SELECTION_ALGO="${2:-kcenter}"
DENSITY_FILTER="${3:-true}"

OUTPUT_DIR="outputs/train_voc_coreset_${FEATURE_MODE}_${SELECTION_ALGO}"
if [[ "${DENSITY_FILTER}" == "true" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_filtered"
fi
WEIGHT="${OUTPUT_DIR}/checkpoints/best_map50.pt"
DATASET_YAML="data/VOC-YOLO/VOC.yaml"
EVAL_OUTPUT_DIR="outputs/eval_voc_coreset_${FEATURE_MODE}_${SELECTION_ALGO}"
if [[ "${DENSITY_FILTER}" == "true" ]]; then
    EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}_filtered"
fi

cmd=(
    python tools/test_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "${DATASET_YAML}"
    --batch_size 4
    --output_dir "${EVAL_OUTPUT_DIR}"
    --weight "${WEIGHT}"
    --text_mode prompt
    --param_tune lora
)

echo "[eval] weight=${WEIGHT}"
echo "[eval] output_dir=${EVAL_OUTPUT_DIR}"
TOKENIZERS_PARALLELISM=false "${cmd[@]}"
