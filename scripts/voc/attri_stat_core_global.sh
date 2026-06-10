TOKENIZERS_PARALLELISM=false

SRC_DATA_YAML="data/VOC-YOLO/VOC.yaml"
CORE_DATA_YAML="data/VOC_coreset_100_80_80_global/data.yaml"
OUTPUT_DIR="outputs/attri_stat_voc_core_global_vs_src"
SPLIT="${1:-train}"

python coreset_selection/attri_stat.py \
    --src_data "$SRC_DATA_YAML" \
    --core_data "$CORE_DATA_YAML" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT"
