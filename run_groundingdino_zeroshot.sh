#!/usr/bin/env bash
set -euo pipefail

# Zero-shot GroundingDINO detection/inference flow (no prompt-tuning checkpoint):
# 1) Single-image zero-shot detection + visualization
# 2) Batch zero-shot inference on split/input directory

ENV_NAME="${ENV_NAME:-dino}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
REALTIME_LOG="${REALTIME_LOG:-1}"

CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/weights/groundingdino_swint_ogc.pth}"

VOC_ROOT="${VOC_ROOT:-${REPO_ROOT}/data/VOC/VOCdevkit/VOC2007-tiny}"
SPLIT="${SPLIT:-test}"
MAX_IMAGES="${MAX_IMAGES:-20}"

INPUT_DIR="${INPUT_DIR:-${VOC_ROOT}/JPEGImages}"
SINGLE_IMAGE="${SINGLE_IMAGE:-${INPUT_DIR}/000007.jpg}"
TEXT_PROMPT="${TEXT_PROMPT:-aeroplane . bicycle . bird . boat . bottle . bus . car . cat . chair . cow . diningtable . dog . horse . motorbike . person . pottedplant . sheep . sofa . train . tvmonitor .}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
DEVICE="${DEVICE:-cuda}"

RUN_TAG="${RUN_TAG:-zeroshot_gdino_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/out/${RUN_TAG}}"
SINGLE_OUT="${OUTPUT_ROOT}/single"
BATCH_OUT="${OUTPUT_ROOT}/batch"

echo "========== GroundingDINO Zero-shot Detection & Inference =========="
echo "Environment      : ${ENV_NAME}"
echo "Config           : ${CONFIG_FILE}"
echo "Checkpoint       : ${CHECKPOINT_PATH}"
echo "VOC root         : ${VOC_ROOT}"
echo "Split            : ${SPLIT}"
echo "Input dir        : ${INPUT_DIR}"
echo "Single image     : ${SINGLE_IMAGE}"
echo "Max batch images : ${MAX_IMAGES}"
echo "Output root      : ${OUTPUT_ROOT}"
echo "===================================================================="

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: Input dir not found: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SINGLE_IMAGE}" ]]; then
  echo "ERROR: Single image not found: ${SINGLE_IMAGE}" >&2
  exit 1
fi

mkdir -p "${SINGLE_OUT}" "${BATCH_OUT}"

CONDA_RUN_ARGS=(-n "${ENV_NAME}")
if [[ "${REALTIME_LOG}" == "1" ]]; then
  CONDA_RUN_ARGS=(--no-capture-output -n "${ENV_NAME}")
fi

CPU_FLAG=()
if [[ "${DEVICE}" == "cpu" ]]; then
  CPU_FLAG=(--cpu-only)
fi

echo
echo "[1/2] Single-image zero-shot detection ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/demo/inference_on_a_image.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --image_path "${SINGLE_IMAGE}" \
  --text_prompt "${TEXT_PROMPT}" \
  --output_dir "${SINGLE_OUT}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  "${CPU_FLAG[@]}"

echo
echo "[2/2] Batch zero-shot inference ..."
mapfile -t IMAGE_LIST < <(VOC_ROOT="${VOC_ROOT}" SPLIT="${SPLIT}" INPUT_DIR="${INPUT_DIR}" MAX_IMAGES="${MAX_IMAGES}" python - <<'PY'
import os
import sys

voc_root = os.environ["VOC_ROOT"]
split = os.environ["SPLIT"]
input_dir = os.environ["INPUT_DIR"]
max_images = int(os.environ["MAX_IMAGES"])

split_file = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
paths = []
if os.path.isfile(split_file):
    with open(split_file, "r", encoding="utf-8") as f:
        ids = [x.strip() for x in f if x.strip()]
    for image_id in ids:
        p = os.path.join(input_dir, f"{image_id}.jpg")
        if os.path.isfile(p):
            paths.append(p)
else:
    for name in sorted(os.listdir(input_dir)):
        if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            paths.append(os.path.join(input_dir, name))

for p in paths[:max_images]:
    print(p)
PY
)

if [[ "${#IMAGE_LIST[@]}" -eq 0 ]]; then
  echo "ERROR: No images selected for batch inference." >&2
  exit 1
fi

for image_path in "${IMAGE_LIST[@]}"; do
  base="$(basename "${image_path}")"
  stem="${base%.*}"
  out_dir="${BATCH_OUT}/${stem}"
  mkdir -p "${out_dir}"
  PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/demo/inference_on_a_image.py" \
    --config_file "${CONFIG_FILE}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --image_path "${image_path}" \
    --text_prompt "${TEXT_PROMPT}" \
    --output_dir "${out_dir}" \
    --box_threshold "${BOX_THRESHOLD}" \
    --text_threshold "${TEXT_THRESHOLD}" \
    "${CPU_FLAG[@]}"
  echo "Processed: ${image_path}"
done

echo
echo "Done. Outputs:"
echo "- Single image result : ${SINGLE_OUT}/pred.jpg"
echo "- Batch result folder : ${BATCH_OUT}"
