TOKENIZERS_PARALLELISM=false

TASK_DATASETS=(
    data/VOC-TINY_10+10/task_1_cls_10/data.yaml
    data/VOC-TINY_10+10/task_2_cls_10/data.yaml
    data/VOC-TINY_10+10/task_1-2_cls_20/data.yaml
)

WEIGHT=outputs/train_wrapper_voc-tiny_10+10/task_2/checkpoints/best_map50.pt

for idx in "${!TASK_DATASETS[@]}"; do
  task_id=$((idx + 1))
  dataset_yaml="${TASK_DATASETS[$idx]}"
  output_dir="outputs/eval_wrapper_voc-tiny_10+10/task_${task_id}"

  cmd=(
    python tools/test_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "$dataset_yaml"
    --batch_size 4
    --output_dir "$output_dir"
    --weight "$WEIGHT"
    --inject_before_encoder
  )

  "${cmd[@]}"
done