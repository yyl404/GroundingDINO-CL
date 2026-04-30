TOKENIZERS_PARALLELISM=false

TASK_DATASETS=(
    data/RSAR-TINY_3+3/task_1_cls_3/data.yaml
    data/RSAR-TINY_3+3/task_2_cls_3/data.yaml
    data/RSAR-TINY_3+3/task_1-2_cls_6/data.yaml
)

for idx in "${!TASK_DATASETS[@]}"; do
  task_id=$((idx + 1))
  dataset_yaml="${TASK_DATASETS[$idx]}"
  output_dir="outputs/eval_wrapper_rsar_3+3_zero-shot/task_${task_id}"

  cmd=(
    python tools/test_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "$dataset_yaml"
    --batch_size 4
    --output_dir "$output_dir"
    --inject_before_encoder
    --zero-shot
  )

  "${cmd[@]}"
done