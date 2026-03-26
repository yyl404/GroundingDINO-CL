CUDA_VISIBLE_DEVICES=0 python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino_swint_ogc.pth \
-i /root/GroundingDINO/data/HRSC2016_dataset/HRSC2016/Test/AllImages/100000661.bmp \
-o "out" \
-t "ship-in-top-down-view"
# --cpu-only