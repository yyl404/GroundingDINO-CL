#!/usr/bin/env bash
set -euo pipefail

# End-to-end Prompt Tuning flow on VOC2007 test split:
# 1) Prompt tuning training (split=test)
# 2) Prompt-tuned evaluation (split=test, report mAP@0.5)

ENV_NAME="${ENV_NAME:-dino}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
REALTIME_LOG="${REALTIME_LOG:-1}"

VOC_ROOT="${VOC_ROOT:-${REPO_ROOT}/data/VOC/VOCdevkit/VOC2007-tiny}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/weights/groundingdino_swint_ogc.pth}"
SPLIT="${SPLIT:-train}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
LR="${LR:-1e-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-16}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"

RUN_TAG="${RUN_TAG:-pt_voc2007_${SPLIT}_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/out/${RUN_TAG}}"
TRAIN_OUT="${OUTPUT_ROOT}/train"
EVAL_OUT="${OUTPUT_ROOT}/eval"

echo "========== Prompt Tuning VOC2007 Flow =========="
echo "Environment     : ${ENV_NAME}"
echo "VOC root        : ${VOC_ROOT}"
echo "Config          : ${CONFIG_FILE}"
echo "Checkpoint      : ${CHECKPOINT_PATH}"
echo "Split           : ${SPLIT}"
echo "Train output    : ${TRAIN_OUT}"
echo "Eval output     : ${EVAL_OUT}"
echo "Device          : ${DEVICE}"
echo "Epochs          : ${EPOCHS}"
echo "Batch size      : ${BATCH_SIZE}"
echo "==============================================="

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

mkdir -p "${TRAIN_OUT}" "${EVAL_OUT}"

CONDA_RUN_ARGS=(-n "${ENV_NAME}")
if [[ "${REALTIME_LOG}" == "1" ]]; then
  CONDA_RUN_ARGS=(--no-capture-output -n "${ENV_NAME}")
fi

echo
echo "[1/2] Training prompt on split=${SPLIT} ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/train_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --voc_root "${VOC_ROOT}" \
  --output_dir "${TRAIN_OUT}" \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --prompt_length "${PROMPT_LENGTH}"

PROMPT_PATH="${TRAIN_OUT}/prompt_final.pth"
if [[ ! -f "${PROMPT_PATH}" ]]; then
  echo "ERROR: Prompt weight was not generated: ${PROMPT_PATH}" >&2
  exit 1
fi

echo
echo "[2/2] Evaluating prompt on split=${SPLIT} ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/pred_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt_path "${PROMPT_PATH}" \
  --input "${VOC_ROOT}" \
  --output_dir "${EVAL_OUT}" \
  --test \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --prompt_length "${PROMPT_LENGTH}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --save_mode txt

METRICS_PATH="${EVAL_OUT}/metrics_${SPLIT}.json"
if [[ -f "${METRICS_PATH}" ]]; then
  echo
  echo "Metrics:"
  PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python -c "import json;print(json.dumps(json.load(open('${METRICS_PATH}')), indent=2, ensure_ascii=False))"
fi

echo
echo "Done. Outputs saved under: ${OUTPUT_ROOT}"
