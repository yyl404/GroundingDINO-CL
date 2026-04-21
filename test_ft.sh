TOKENIZERS_PARALLELISM=false
python tools/test_wrapper.py \
  --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --pretrained_checkpoint weights/groundingdino_swint_ogc.pth \
  --dataset_yaml data/VOC-TINY-YOLO/VOC.yaml \
  --batch_size 4 \
  --output_dir outputs/test_wrapper_voc \
  --weight outputs/train_wrapper_voc/checkpoints/best_map50.pt \
  --inject_before_encoder \
  --vis-batch 1