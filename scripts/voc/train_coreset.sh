TOKENIZERS_PARALLELISM=false

DATASET_YAML="data/VOC_coreset_100_80_80/data.yaml"
OUTPUT_DIR="outputs/train_wrapper_voc_coreset/"

cmd=(
    python tools/train_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "$DATASET_YAML"
    --epochs 20
    --batch_size 4
    --lr 1e-4
    --output_dir "$OUTPUT_DIR"
    --text_mode prompt
    --param_tune lora
)

echo "Training task $task_id"
"${cmd[@]}"