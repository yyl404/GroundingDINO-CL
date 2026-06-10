TOKENIZERS_PARALLELISM=false

python coreset_selection/select_coreset.py \
    --src_data data/VOC-YOLO/VOC.yaml \
    --out_dir data/VOC_coreset_100_80_80_global \
    --num_sample "[100,80,80]" \
    --global_feature_selection \
    --save_meta
