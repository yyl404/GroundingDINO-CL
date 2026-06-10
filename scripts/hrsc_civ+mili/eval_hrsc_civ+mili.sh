TOKENIZERS_PARALLELISM=false

TASK_DATASETS=(
    data/HRSC2016-YOLO-CIL/task-1-civ/HRSC2016.yaml
    data/HRSC2016-YOLO-CIL/task-2-mili/HRSC2016.yaml
)

WEIGHT=outputs/train_wrapper_hrsc_civ+mili/task_2/checkpoints/best_map50.pt

for idx in "${!TASK_DATASETS[@]}"; do
  task_id=$((idx + 1))
  dataset_yaml="${TASK_DATASETS[$idx]}"
  output_dir="outputs/eval_wrapper_hrsc_civ+mili/task_${task_id}"

  cmd=(
    python tools/test_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "$dataset_yaml"
    --batch_size 4
    --output_dir "$output_dir"
    --weight "$WEIGHT"
    --text_mode prompt
  )

  "${cmd[@]}"
done