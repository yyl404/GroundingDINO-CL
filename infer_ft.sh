CUDA_VISIBLE_DEVICES=0 python tools/inference_on_a_image_wrapper.py \
    -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
    -p weights/groundingdino_swint_ogc.pth \
    -i data/HRSC2016-YOLO-Coarse/images/test/100000012.bmp \
    -o "outputs" \
    -wp outputs/train_wrapper_hrsc/checkpoints/best_map50.pt \
    --aggregation_method mean \
    --box_threshold 0.25 \
    --inject_before_encoder \
    -cls "ship"