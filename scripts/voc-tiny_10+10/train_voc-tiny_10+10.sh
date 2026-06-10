TOKENIZERS_PARALLELISM=false

TASK_DATASETS=(
    data/VOC-TINY_10+10/task_1_cls_10/data.yaml
    data/VOC-TINY_10+10/task_2_cls_10/data.yaml
)

START_TASK="${START_TASK:-1}"

for idx in "${!TASK_DATASETS[@]}"; do
  task_id=$((idx + 1))
  if [ "$task_id" -lt "$START_TASK" ]; then
    continue
  fi
  dataset_yaml="${TASK_DATASETS[$idx]}"
  output_dir="outputs/train_wrapper_voc-tiny_10+10/task_${task_id}"

  cmd=(
    python tools/train_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "$dataset_yaml"
    --epochs 5
    --batch_size 4
    --lr 1e-4
    --output_dir "$output_dir"
    --text_mode prompt
    --param_tune lora
  )

  if [ "$task_id" -gt 1 ]; then
    prev_task_id=$((task_id - 1))
    prev_ckpt="outputs/train_wrapper_voc-tiny_10+10/task_${prev_task_id}/checkpoints/best_map50.pt"
    cmd+=(--load_wrapper "$prev_ckpt")
  fi

  echo "Training task $task_id"
  "${cmd[@]}"
done