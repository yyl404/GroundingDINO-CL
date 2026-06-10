TOKENIZERS_PARALLELISM=false

WEIGHT=outputs/train_wrapper_voc_random/checkpoints/best_map50.pt
DATASET_YAML="data/VOC-YOLO/VOC.yaml"
OUTPUT_DIR="outputs/eval_wrapper_voc_random"

cmd=(
    python tools/test_wrapper.py
    --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py
    --pretrained_checkpoint weights/groundingdino_swint_ogc.pth
    --dataset_yaml "$DATASET_YAML"
    --batch_size 4
    --output_dir "$OUTPUT_DIR"
    --weight "$WEIGHT"
    --text_mode prompt
    --param_tune lora
)

"${cmd[@]}"