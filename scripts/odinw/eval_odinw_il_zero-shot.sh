#!/usr/bin/env bash
# Zero-shot evaluation on all OdinW-13 test sets (no finetuned weight).
# Uses fixed class captions and frozen parameters.

set -euo pipefail
cd "$(dirname "$0")/../../"
source scripts/odinw/odinw_config.sh
export TOKENIZERS_PARALLELISM=false

EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/eval_odinw_il/zero-shot}"

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

echo "=== Zero-shot eval on all OdinW-13 datasets ==="

for eval_dataset in "${ODINW_DATASETS[@]}"; do
  mapfile -t paths < <(resolve_test_path "$eval_dataset")
  test_json_path="${paths[0]}"
  image_dir="${paths[1]}"
  output_dir="${EVAL_OUTPUT_ROOT}/eval_on_${eval_dataset}"

  python tools/test_wrapper.py \
    --config_file "$CONFIG_FILE" \
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
    --dataset_format coco \
    --test_json "$test_json_path" \
    --image_dir "$image_dir" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --output_dir "$output_dir" \
    --text_mode fixed \
    --param_tune frozen \
    --eval_metric "$EVAL_METRIC" \
    --vis-batch "$VIS_BATCH"
done

python scripts/odinw/aggregate_odinw_results.py \
  --eval-root "$EVAL_OUTPUT_ROOT" \
  --metric "$EVAL_METRIC" \
  --flat \
  --output "${EVAL_OUTPUT_ROOT}/results_zero_shot.csv"

echo "Saved: ${EVAL_OUTPUT_ROOT}/results_zero_shot.csv"
