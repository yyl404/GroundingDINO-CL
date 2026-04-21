TOKENIZERS_PARALLELISM=false
python tools/train_wrapper.py \
  --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --pretrained_checkpoint weights/groundingdino_swint_ogc.pth \
  --dataset_yaml data/VOC-TINY-YOLO/VOC.yaml \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-4 \
  --output_dir outputs/train_wrapper_voc \
  --inject_before_encoder