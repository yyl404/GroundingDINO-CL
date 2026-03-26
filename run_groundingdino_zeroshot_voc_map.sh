#!/usr/bin/env bash
set -euo pipefail

# Zero-shot GroundingDINO evaluation on VOC with mAP@0.5.
# This script does NOT load prompt-tuning weights.

ENV_NAME="${ENV_NAME:-dino}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
REALTIME_LOG="${REALTIME_LOG:-1}"

VOC_ROOT="${VOC_ROOT:-${REPO_ROOT}/data/VOC/VOCdevkit/VOC2007-tiny}"
SPLIT="${SPLIT:-test}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/weights/groundingdino_swint_ogc.pth}"
DEVICE="${DEVICE:-cuda}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"

RUN_TAG="${RUN_TAG:-zeroshot_voc_map_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/out/${RUN_TAG}}"
EVAL_OUT="${OUTPUT_ROOT}/eval"

echo "========== GroundingDINO Zero-shot VOC mAP =========="
echo "Environment   : ${ENV_NAME}"
echo "VOC root      : ${VOC_ROOT}"
echo "Split         : ${SPLIT}"
echo "Config        : ${CONFIG_FILE}"
echo "Checkpoint    : ${CHECKPOINT_PATH}"
echo "Device        : ${DEVICE}"
echo "Eval output   : ${EVAL_OUT}"
echo "======================================================"

if [[ ! -d "${VOC_ROOT}" ]]; then
  echo "ERROR: VOC root not found: ${VOC_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "${EVAL_OUT}"

CONDA_RUN_ARGS=(-n "${ENV_NAME}")
if [[ "${REALTIME_LOG}" == "1" ]]; then
  CONDA_RUN_ARGS=(--no-capture-output -n "${ENV_NAME}")
fi

echo
echo "[1/1] Running zero-shot VOC evaluation ..."
VOC_ROOT="${VOC_ROOT}" \
SPLIT="${SPLIT}" \
CONFIG_FILE="${CONFIG_FILE}" \
CHECKPOINT_PATH="${CHECKPOINT_PATH}" \
DEVICE="${DEVICE}" \
BOX_THRESHOLD="${BOX_THRESHOLD}" \
TEXT_THRESHOLD="${TEXT_THRESHOLD}" \
EVAL_OUT="${EVAL_OUT}" \
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python - <<'PY'
import json
import os

import numpy as np
import torch

from groundingdino.prompt_tuning.predictor import (
    decode_predictions,
    evaluate_voc_map,
    load_groundingdino_model,
    save_detection_txt,
    save_metrics,
)
from groundingdino.prompt_tuning.voc import (
    VOC_CLASSES,
    VOCDataset,
    build_caption,
    build_class_token_map,
    build_eval_transform,
    get_split_present_class_names,
)
from tools.pred_pt import to_absolute_xyxy

repo_root = os.getcwd()
voc_root = os.environ.get("VOC_ROOT", "")
split = os.environ.get("SPLIT", "test")
config_file = os.environ.get("CONFIG_FILE", "")
checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
device = os.environ.get("DEVICE", "cuda")
box_threshold = float(os.environ.get("BOX_THRESHOLD", "0.3"))
text_threshold = float(os.environ.get("TEXT_THRESHOLD", "0.25"))
eval_out = os.environ.get("EVAL_OUT", os.path.join(repo_root, "out", "zeroshot_eval"))

os.makedirs(eval_out, exist_ok=True)
txt_dir = os.path.join(eval_out, "txt")
os.makedirs(txt_dir, exist_ok=True)

model = load_groundingdino_model(config_file, checkpoint_path, device=device)
model.to(device)
model.eval()

caption = build_caption(VOC_CLASSES)
class_token_map = build_class_token_map(model.tokenizer, VOC_CLASSES)
dataset = VOCDataset(
    voc_root=voc_root,
    split=split,
    transforms=build_eval_transform(),
    classes=VOC_CLASSES,
    keep_difficult=True,
)

all_predictions = {}
all_ground_truths = {}
for i in range(len(dataset)):
    image_tensor, target = dataset[i]
    image_name = target["image_name"]
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor[None], captions=[caption])

    orig_h = int(target["orig_size"][0].item())
    orig_w = int(target["orig_size"][1].item())
    boxes_cxcywh, scores, class_ids = decode_predictions(
        outputs=outputs,
        class_token_map=class_token_map,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        num_classes=len(VOC_CLASSES),
    )
    boxes_xyxy = to_absolute_xyxy(boxes_cxcywh, orig_w, orig_h)
    save_detection_txt(os.path.join(txt_dir, f"{image_name}.txt"), boxes_xyxy, scores, class_ids, VOC_CLASSES)

    all_predictions[image_name] = [
        (float(score), int(class_id), box.astype(np.float32))
        for score, class_id, box in zip(scores, class_ids, boxes_xyxy)
    ]

    gt_boxes = target["boxes_abs"].numpy().astype(np.float32)
    gt_labels = target["labels"].numpy().astype(np.int64)
    gt_difficult = target["difficult"].numpy().astype(np.int64)
    all_ground_truths[image_name] = [
        (int(label), box.astype(np.float32), int(diff))
        for label, box, diff in zip(gt_labels, gt_boxes, gt_difficult)
    ]

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(dataset)} samples.")

dataset_classes = get_split_present_class_names(voc_root, split, classes=VOC_CLASSES)
eval_class_ids = sorted({VOC_CLASSES.index(c) for c in dataset_classes})
metrics = evaluate_voc_map(
    all_predictions,
    all_ground_truths,
    num_classes=len(VOC_CLASSES),
    class_names=VOC_CLASSES,
    eval_class_ids=eval_class_ids,
)

metrics_path = os.path.join(eval_out, f"metrics_{split}.json")
save_metrics(metrics_path, metrics)
print(f"Evaluation done. classes={len(eval_class_ids)} mAP@0.5={metrics['mAP@0.5']:.4f}")
print(f"Metrics saved to: {metrics_path}")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
PY

echo
echo "Done. Metrics file: ${EVAL_OUT}/metrics_${SPLIT}.json"
