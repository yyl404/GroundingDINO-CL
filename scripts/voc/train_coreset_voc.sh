#!/usr/bin/env bash
# Train on a VOC core-set subset produced by extract_coreset_voc.sh
#
# Usage:
#   bash scripts/voc/train_coreset_voc.sh [feature_mode] [selection_algo] [density_filter]
#
# Arguments must match the extraction script used to build the dataset.

set -euo pipefail

FEATURE_MODE="${1:-classwise}"
SELECTION_ALGO="${2:-kcenter}"
DENSITY_FILTER="${3:-true}"

DATASET_DIR="data/VOC_coreset_100_80_80_${FEATURE_MODE}_${SELECTION_ALGO}"
if [[ "${DENSITY_FILTER}" == "true" ]]; then
    DATASET_DIR="${DATASET_DIR}_filtered"
fi
DATASET_YAML="${DATASET_DIR}/data.yaml"
OUTPUT_DIR="outputs/train_voc_coreset_${FEATURE_MODE}_${SELECTION_ALGO}"
if [[ "${DENSITY_FILTER}" == "true" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_filtered"
fi

cmd=(
    python tools/train_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "${DATASET_YAML}"
    --epochs 20
    --batch_size 4
    --lr 1e-4
    --output_dir "${OUTPUT_DIR}"
    --text_mode prompt
    --param_tune lora
)

echo "[train] dataset=${DATASET_YAML}"
echo "[train] output_dir=${OUTPUT_DIR}"
TOKENIZERS_PARALLELISM=false "${cmd[@]}"
