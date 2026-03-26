#!/usr/bin/env bash
set -euo pipefail

# Class-incremental Prompt Tuning flow for VOC2007-tiny-15+5
# Stage-1: train on task1_15
# Stage-2: incremental train on task2_5 (resume Stage-1 prompt)
# Final: evaluate final prompt on both task1_15 and task2_5

ENV_NAME="${ENV_NAME:-dino}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
REALTIME_LOG="${REALTIME_LOG:-1}"

TASK1_VOC_ROOT="${TASK1_VOC_ROOT:-${REPO_ROOT}/data/VOC/VOCdevkit/VOC2007-tiny-15+5/task1_15}"
TASK2_VOC_ROOT="${TASK2_VOC_ROOT:-${REPO_ROOT}/data/VOC/VOCdevkit/VOC2007-tiny-15+5/task2_5}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/weights/groundingdino_swint_ogc.pth}"

TRAIN_SPLIT="${TRAIN_SPLIT:-trainval}"
TEST_SPLIT="${TEST_SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
PROMPT_INIT_STD="${PROMPT_INIT_STD:-0.02}"
PROMPT_MODE="${PROMPT_MODE:-class_independent}"
DOMAIN_ID="${DOMAIN_ID:-voc2007_tiny_15plus5}"

EPOCHS_TASK1="${EPOCHS_TASK1:-1}"
EPOCHS_TASK2="${EPOCHS_TASK2:-1}"
BATCH_SIZE_TASK1="${BATCH_SIZE_TASK1:-8}"
BATCH_SIZE_TASK2="${BATCH_SIZE_TASK2:-8}"

BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"

RUN_TAG="${RUN_TAG:-pt_inc_voc2007_tiny_15plus5_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/out/${RUN_TAG}}"
STAGE1_OUT="${OUTPUT_ROOT}/stage1_task1_15"
STAGE2_OUT="${OUTPUT_ROOT}/stage2_task2_5"
EVAL_TASK1_OUT="${OUTPUT_ROOT}/eval_task1_15"
EVAL_TASK2_OUT="${OUTPUT_ROOT}/eval_task2_5"

echo "========== VOC2007-tiny-15+5 Incremental Prompt Tuning =========="
echo "Environment        : ${ENV_NAME}"
echo "Task1 VOC root     : ${TASK1_VOC_ROOT}"
echo "Task2 VOC root     : ${TASK2_VOC_ROOT}"
echo "Config             : ${CONFIG_FILE}"
echo "Checkpoint         : ${CHECKPOINT_PATH}"
echo "Train split        : ${TRAIN_SPLIT}"
echo "Test split         : ${TEST_SPLIT}"
echo "Prompt mode        : ${PROMPT_MODE}"
echo "Domain id          : ${DOMAIN_ID}"
echo "Stage1 output      : ${STAGE1_OUT}"
echo "Stage2 output      : ${STAGE2_OUT}"
echo "Eval task1 output  : ${EVAL_TASK1_OUT}"
echo "Eval task2 output  : ${EVAL_TASK2_OUT}"
echo "=================================================================="

for p in "${TASK1_VOC_ROOT}" "${TASK2_VOC_ROOT}"; do
  if [[ ! -d "${p}" ]]; then
    echo "ERROR: VOC root not found: ${p}" >&2
    exit 1
  fi
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

mkdir -p "${STAGE1_OUT}" "${STAGE2_OUT}" "${EVAL_TASK1_OUT}" "${EVAL_TASK2_OUT}"

CONDA_RUN_ARGS=(-n "${ENV_NAME}")
if [[ "${REALTIME_LOG}" == "1" ]]; then
  CONDA_RUN_ARGS=(--no-capture-output -n "${ENV_NAME}")
fi

echo
echo "[1/4] Stage-1 train on task1_15 ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/train_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --voc_root "${TASK1_VOC_ROOT}" \
  --output_dir "${STAGE1_OUT}" \
  --split "${TRAIN_SPLIT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS_TASK1}" \
  --batch_size "${BATCH_SIZE_TASK1}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --prompt_init_std "${PROMPT_INIT_STD}" \
  --prompt_mode "${PROMPT_MODE}" \
  --domain_id "${DOMAIN_ID}"

PROMPT_STAGE1="${STAGE1_OUT}/prompt_final.pth"
if [[ ! -f "${PROMPT_STAGE1}" ]]; then
  echo "ERROR: Stage-1 prompt weight not found: ${PROMPT_STAGE1}" >&2
  exit 1
fi

echo
echo "[2/4] Stage-2 incremental train on task2_5 (resume stage-1) ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/train_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --voc_root "${TASK2_VOC_ROOT}" \
  --output_dir "${STAGE2_OUT}" \
  --split "${TRAIN_SPLIT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS_TASK2}" \
  --batch_size "${BATCH_SIZE_TASK2}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --prompt_init_std "${PROMPT_INIT_STD}" \
  --prompt_mode "${PROMPT_MODE}" \
  --domain_id "${DOMAIN_ID}" \
  --resume_prompt "${PROMPT_STAGE1}"

PROMPT_FINAL="${STAGE2_OUT}/prompt_final.pth"
if [[ ! -f "${PROMPT_FINAL}" ]]; then
  echo "ERROR: Final prompt weight not found: ${PROMPT_FINAL}" >&2
  exit 1
fi

echo
echo "[3/4] Evaluate final prompt on task1_15 ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/pred_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt_path "${PROMPT_FINAL}" \
  --domain_id "${DOMAIN_ID}" \
  --input "${TASK1_VOC_ROOT}" \
  --output_dir "${EVAL_TASK1_OUT}" \
  --test \
  --split "${TEST_SPLIT}" \
  --device "${DEVICE}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --save_mode txt

echo
echo "[4/4] Evaluate final prompt on task2_5 ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/pred_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt_path "${PROMPT_FINAL}" \
  --domain_id "${DOMAIN_ID}" \
  --input "${TASK2_VOC_ROOT}" \
  --output_dir "${EVAL_TASK2_OUT}" \
  --test \
  --split "${TEST_SPLIT}" \
  --device "${DEVICE}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --save_mode txt

echo
echo "Done. Results:"
echo "- Stage1 prompt : ${PROMPT_STAGE1}"
echo "- Final prompt  : ${PROMPT_FINAL}"
echo "- Task1 metrics : ${EVAL_TASK1_OUT}/metrics_${TEST_SPLIT}.json"
echo "- Task2 metrics : ${EVAL_TASK2_OUT}/metrics_${TEST_SPLIT}.json"
