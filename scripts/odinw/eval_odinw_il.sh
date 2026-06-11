#!/usr/bin/env bash
# Evaluate each incremental stage model on all OdinW-13 test sets.
# Not tied to training shot count; specialized by TEXT_MODE + PARAM_TUNE.
#
# Environment variables:
#   TEXT_MODE, PARAM_TUNE  : must match the trained run
#   TRAIN_OUTPUT_ROOT      : where stage checkpoints live (default from odinw_config.sh)
#   EVAL_OUTPUT_ROOT       : where eval logs are saved
#   SHOT_MODE              : only used to locate TRAIN_OUTPUT_ROOT if not overridden

set -euo pipefail
cd "$(dirname "$0")/../../"
source scripts/odinw/odinw_config.sh
export TOKENIZERS_PARALLELISM=false

resolve_test_path() {
  python3 - "$1" <<'PY'
import sys
from scripts.odinw.odinw_datasets import ODINW_DATASETS, test_json, image_dir_for_json

name = sys.argv[1]
ds = next(d for d in ODINW_DATASETS if d.name == name)
test_path = test_json(ds)
print(test_path)
print(image_dir_for_json(test_path))
PY
}

for stage_idx in "${!ODINW_DATASETS[@]}"; do
  stage=$((stage_idx + 1))
  trained_dataset="${ODINW_DATASETS[$stage_idx]}"
  weight="${TRAIN_OUTPUT_ROOT}/stage_${stage}_${trained_dataset}/checkpoints/best_map50.pt"

  if [ ! -f "$weight" ]; then
    echo "Skip stage ${stage}: checkpoint not found at ${weight}"
    continue
  fi

  echo "=== Evaluating stage ${stage} model (trained through ${trained_dataset}) ==="

  for eval_dataset in "${ODINW_DATASETS[@]}"; do
    mapfile -t paths < <(resolve_test_path "$eval_dataset")
    test_json_path="${paths[0]}"
    image_dir="${paths[1]}"
    output_dir="${EVAL_OUTPUT_ROOT}/stage_${stage}/eval_on_${eval_dataset}"

    python tools/test_wrapper.py \
      --config_file "$CONFIG_FILE" \
      --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
      --dataset_format coco \
      --test_json "$test_json_path" \
      --image_dir "$image_dir" \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --output_dir "$output_dir" \
      --weight "$weight" \
      --text_mode "$TEXT_MODE" \
      --param_tune "$PARAM_TUNE" \
      --eval_metric "$EVAL_METRIC" \
      --vis-batch "$VIS_BATCH"
  done
done

python scripts/odinw/aggregate_odinw_results.py \
  --eval-root "$EVAL_OUTPUT_ROOT" \
  --metric "$EVAL_METRIC" \
  --output "${EVAL_OUTPUT_ROOT}/results_matrix.csv"

echo "Saved matrix: ${EVAL_OUTPUT_ROOT}/results_matrix.csv"
