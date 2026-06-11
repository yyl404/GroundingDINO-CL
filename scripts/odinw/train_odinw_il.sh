#!/usr/bin/env bash
# Sequential incremental training on OdinW-13 (dictionary order).
# Each stage loads best_map50.pt from the previous stage.
#
# Environment variables:
#   SHOT_MODE   : 1 | 3 | 5 | 10 | full
#   TEXT_MODE   : prompt | fixed
#   PARAM_TUNE  : full | lora | delta | frozen
#   START_STAGE : optional, resume from this stage (1-based)
#   EPOCHS, BATCH_SIZE, LR, ...

set -euo pipefail
cd "$(dirname "$0")/../../"
source scripts/odinw/odinw_config.sh
export TOKENIZERS_PARALLELISM=false

START_STAGE="${START_STAGE:-1}"

resolve_paths() {
  python3 - "$1" "$2" <<'PY'
import sys
from scripts.odinw.odinw_datasets import ODINW_DATASETS, train_json, val_json, image_dir_for_json

name, shot_mode = sys.argv[1], sys.argv[2]
ds = next(d for d in ODINW_DATASETS if d.name == name)
train_path = train_json(ds, shot_mode)
val_path = val_json(ds)
print(train_path)
print(val_path)
print(image_dir_for_json(train_path))
print(image_dir_for_json(val_path))
PY
}

for idx in "${!ODINW_DATASETS[@]}"; do
  stage=$((idx + 1))
  if [ "$stage" -lt "$START_STAGE" ]; then
    continue
  fi

  dataset_name="${ODINW_DATASETS[$idx]}"
  mapfile -t paths < <(resolve_paths "$dataset_name" "$SHOT_MODE")
  train_json_path="${paths[0]}"
  val_json_path="${paths[1]}"
  train_image_dir="${paths[2]}"
  val_image_dir="${paths[3]}"

  output_dir="${TRAIN_OUTPUT_ROOT}/stage_${stage}_${dataset_name}"

  cmd=(
    python tools/train_wrapper.py
    --config_file "$CONFIG_FILE"
    --pretrained_checkpoint "$PRETRAINED_CHECKPOINT"
    --dataset_format coco
    --train_json "$train_json_path"
    --val_json "$val_json_path"
    --train_image_dir "$train_image_dir"
    --val_image_dir "$val_image_dir"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --lr "$LR"
    --output_dir "$output_dir"
    --text_mode "$TEXT_MODE"
    --param_tune "$PARAM_TUNE"
  )

  if [ "$stage" -gt 1 ]; then
    prev_stage=$((stage - 1))
    prev_dataset="${ODINW_DATASETS[$((stage - 2))]}"
    prev_ckpt="${TRAIN_OUTPUT_ROOT}/stage_${prev_stage}_${prev_dataset}/checkpoints/best_map50.pt"
    cmd+=(--load_wrapper "$prev_ckpt")
  fi

  echo "=== Stage ${stage}/13: ${dataset_name} (shot=${SHOT_MODE}, ${RUN_TAG}) ==="
  "${cmd[@]}"
done
