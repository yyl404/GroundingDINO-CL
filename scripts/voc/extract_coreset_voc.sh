#!/usr/bin/env bash
# Core-set extraction for VOC-YOLO with configurable pipeline stages.
#
# Usage:
#   bash scripts/voc/extract_coreset_voc.sh [feature_mode] [selection_algo] [density_filter]
#
# Arguments:
#   feature_mode    global | classwise          (default: classwise)
#   selection_algo  kcenter | facility_location | kmeans  (default: kcenter)
#   density_filter  true | false                (default: true)
#
# Examples:
#   bash scripts/voc/extract_coreset_voc.sh classwise kcenter true
#   bash scripts/voc/extract_coreset_voc.sh global facility_location false

set -euo pipefail

FEATURE_MODE="${1:-classwise}"
SELECTION_ALGO="${2:-kcenter}"
DENSITY_FILTER="${3:-true}"

SRC_DATA="data/VOC-YOLO/VOC.yaml"
NUM_SAMPLE="[100,80,80]"
OUT_DIR="data/VOC_coreset_100_80_80_${FEATURE_MODE}_${SELECTION_ALGO}"
if [[ "${DENSITY_FILTER}" == "true" ]]; then
    OUT_DIR="${OUT_DIR}_filtered"
fi

cmd=(
    python coreset_selection/select_coreset.py
    --src_data "${SRC_DATA}"
    --out_dir "${OUT_DIR}"
    --num_sample "${NUM_SAMPLE}"
    --feature_mode "${FEATURE_MODE}"
    --selection_algo "${SELECTION_ALGO}"
    --density_k 20
    --density_outlier_percentile 95
    --save_meta
)

if [[ "${DENSITY_FILTER}" == "true" ]]; then
    cmd+=(--enable_density_filter)
fi

echo "[extract] feature_mode=${FEATURE_MODE}, selection_algo=${SELECTION_ALGO}, density_filter=${DENSITY_FILTER}"
echo "[extract] output_dir=${OUT_DIR}"
TOKENIZERS_PARALLELISM=false "${cmd[@]}"
