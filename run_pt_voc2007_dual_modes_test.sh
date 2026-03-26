#!/usr/bin/env bash
set -euo pipefail

# Three training modes for prompt tuning + evaluation after each mode:
# 1) baseline (main prompt only)
# 2) joint (main + aux prompt, joint tuning)
# 3) frozen_main (freeze main prompt, tune aux prompt only)
# Finally run one manual-aux dual-prompt visualization inference.

ENV_NAME="${ENV_NAME:-dino}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
REALTIME_LOG="${REALTIME_LOG:-1}"

VOC_ROOT="${VOC_ROOT:-${REPO_ROOT}/data/VOC/VOCdevkit/VOC2007-tiny}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/weights/groundingdino_swint_ogc.pth}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
TEST_SPLIT="${TEST_SPLIT:-test}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-3}"
PROMPT_LENGTH="${PROMPT_LENGTH:-16}"
PROMPT_MODE="${PROMPT_MODE:-class_independent}"
DOMAIN_ID="${DOMAIN_ID:-voc2007_dual_modes}"
PROMPT_INIT_STD="${PROMPT_INIT_STD:-0.02}"
AUX_PROMPT_INIT_STD="${AUX_PROMPT_INIT_STD:-0.02}"
JOINT_MAIN_ONLY_LOSS_WEIGHT="${JOINT_MAIN_ONLY_LOSS_WEIGHT:-1.0}"

EPOCHS_BASELINE="${EPOCHS_BASELINE:-1}"
EPOCHS_JOINT="${EPOCHS_JOINT:-1}"
EPOCHS_FROZEN="${EPOCHS_FROZEN:-1}"
BATCH_SIZE_BASELINE="${BATCH_SIZE_BASELINE:-8}"
BATCH_SIZE_JOINT="${BATCH_SIZE_JOINT:-8}"
BATCH_SIZE_FROZEN="${BATCH_SIZE_FROZEN:-8}"

BOX_THRESHOLD="${BOX_THRESHOLD:-0.3}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"

INFER_INPUT="${INFER_INPUT:-${VOC_ROOT}/JPEGImages}"
AUX_PROMPTS_STR="${AUX_PROMPTS_STR:-left right frontal rear}"
read -r -a AUX_PROMPTS <<< "${AUX_PROMPTS_STR}"

RUN_TAG="${RUN_TAG:-pt_dual_modes_voc2007_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/out/${RUN_TAG}}"
BASELINE_TRAIN_OUT="${OUTPUT_ROOT}/baseline_train"
BASELINE_EVAL_OUT="${OUTPUT_ROOT}/baseline_eval"
JOINT_TRAIN_OUT="${OUTPUT_ROOT}/joint_train"
JOINT_EVAL_OUT="${OUTPUT_ROOT}/joint_eval"
FROZEN_TRAIN_OUT="${OUTPUT_ROOT}/frozen_main_train"
FROZEN_EVAL_OUT="${OUTPUT_ROOT}/frozen_main_eval"
VIS_OUT="${OUTPUT_ROOT}/manual_aux_vis"

echo "========== VOC Prompt Tuning: Three Modes + Dual Inference =========="
echo "Environment        : ${ENV_NAME}"
echo "VOC root           : ${VOC_ROOT}"
echo "Config             : ${CONFIG_FILE}"
echo "Checkpoint         : ${CHECKPOINT_PATH}"
echo "Train split        : ${TRAIN_SPLIT}"
echo "Test split         : ${TEST_SPLIT}"
echo "Prompt mode        : ${PROMPT_MODE}"
echo "Domain id          : ${DOMAIN_ID}"
echo "Output root        : ${OUTPUT_ROOT}"
echo "Inference input    : ${INFER_INPUT}"
echo "Aux prompts        : ${AUX_PROMPTS_STR}"
echo "====================================================================="

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
if [[ ! -e "${INFER_INPUT}" ]]; then
  echo "ERROR: Inference input does not exist: ${INFER_INPUT}" >&2
  exit 1
fi

mkdir -p \
  "${BASELINE_TRAIN_OUT}" "${BASELINE_EVAL_OUT}" \
  "${JOINT_TRAIN_OUT}" "${JOINT_EVAL_OUT}" \
  "${FROZEN_TRAIN_OUT}" "${FROZEN_EVAL_OUT}" \
  "${VIS_OUT}"

CONDA_RUN_ARGS=(-n "${ENV_NAME}")
if [[ "${REALTIME_LOG}" == "1" ]]; then
  CONDA_RUN_ARGS=(--no-capture-output -n "${ENV_NAME}")
fi

echo
echo "[1/7] Train baseline (main prompt only) ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/train_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --voc_root "${VOC_ROOT}" \
  --output_dir "${BASELINE_TRAIN_OUT}" \
  --split "${TRAIN_SPLIT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS_BASELINE}" \
  --batch_size "${BATCH_SIZE_BASELINE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --prompt_length "${PROMPT_LENGTH}" \
  --prompt_init_std "${PROMPT_INIT_STD}" \
  --prompt_mode "${PROMPT_MODE}" \
  --domain_id "${DOMAIN_ID}" \
  --dual_train_mode baseline

PROMPT_BASELINE="${BASELINE_TRAIN_OUT}/prompt_final.pth"
if [[ ! -f "${PROMPT_BASELINE}" ]]; then
  echo "ERROR: Baseline prompt not found: ${PROMPT_BASELINE}" >&2
  exit 1
fi

echo
echo "[2/7] Evaluate baseline ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/pred_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt_path "${PROMPT_BASELINE}" \
  --domain_id "${DOMAIN_ID}" \
  --input "${VOC_ROOT}" \
  --output_dir "${BASELINE_EVAL_OUT}" \
  --test \
  --split "${TEST_SPLIT}" \
  --device "${DEVICE}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --save_mode txt

echo
echo "[3/7] Train joint dual-prompt (main + aux) ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/train_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --voc_root "${VOC_ROOT}" \
  --output_dir "${JOINT_TRAIN_OUT}" \
  --split "${TRAIN_SPLIT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS_JOINT}" \
  --batch_size "${BATCH_SIZE_JOINT}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --prompt_length "${PROMPT_LENGTH}" \
  --prompt_init_std "${PROMPT_INIT_STD}" \
  --aux_prompt_init_std "${AUX_PROMPT_INIT_STD}" \
  --prompt_mode "${PROMPT_MODE}" \
  --domain_id "${DOMAIN_ID}" \
  --dual_train_mode joint \
  --joint_main_only_loss_weight "${JOINT_MAIN_ONLY_LOSS_WEIGHT}" \
  --resume_prompt "${PROMPT_BASELINE}"

PROMPT_JOINT="${JOINT_TRAIN_OUT}/prompt_final.pth"
if [[ ! -f "${PROMPT_JOINT}" ]]; then
  echo "ERROR: Joint prompt not found: ${PROMPT_JOINT}" >&2
  exit 1
fi

echo
echo "[4/7] Evaluate joint dual-prompt ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/pred_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt_path "${PROMPT_JOINT}" \
  --domain_id "${DOMAIN_ID}" \
  --input "${VOC_ROOT}" \
  --output_dir "${JOINT_EVAL_OUT}" \
  --test \
  --split "${TEST_SPLIT}" \
  --device "${DEVICE}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --save_mode txt \
  --dual_prompt

echo
echo "[5/7] Train frozen_main dual-prompt (freeze main, tune aux) ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/train_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --voc_root "${VOC_ROOT}" \
  --output_dir "${FROZEN_TRAIN_OUT}" \
  --split "${TRAIN_SPLIT}" \
  --device "${DEVICE}" \
  --epochs "${EPOCHS_FROZEN}" \
  --batch_size "${BATCH_SIZE_FROZEN}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --prompt_length "${PROMPT_LENGTH}" \
  --prompt_init_std "${PROMPT_INIT_STD}" \
  --aux_prompt_init_std "${AUX_PROMPT_INIT_STD}" \
  --prompt_mode "${PROMPT_MODE}" \
  --domain_id "${DOMAIN_ID}" \
  --dual_train_mode frozen_main \
  --resume_prompt "${PROMPT_BASELINE}" \
  --frozen_main_prompt_path "${PROMPT_BASELINE}"

PROMPT_FROZEN="${FROZEN_TRAIN_OUT}/prompt_final.pth"
if [[ ! -f "${PROMPT_FROZEN}" ]]; then
  echo "ERROR: Frozen-main prompt not found: ${PROMPT_FROZEN}" >&2
  exit 1
fi

echo
echo "[6/7] Evaluate frozen_main dual-prompt ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/pred_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt_path "${PROMPT_FROZEN}" \
  --domain_id "${DOMAIN_ID}" \
  --input "${VOC_ROOT}" \
  --output_dir "${FROZEN_EVAL_OUT}" \
  --test \
  --split "${TEST_SPLIT}" \
  --device "${DEVICE}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --save_mode txt \
  --dual_prompt

echo
echo "[7/7] Manual auxiliary prompts visualization inference (using joint prompt) ..."
PYTHONUNBUFFERED=1 conda run "${CONDA_RUN_ARGS[@]}" python "${REPO_ROOT}/tools/pred_pt.py" \
  --config_file "${CONFIG_FILE}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt_path "${PROMPT_JOINT}" \
  --domain_id "${DOMAIN_ID}" \
  --input "${INFER_INPUT}" \
  --output_dir "${VIS_OUT}" \
  --device "${DEVICE}" \
  --box_threshold "${BOX_THRESHOLD}" \
  --text_threshold "${TEXT_THRESHOLD}" \
  --save_mode vis \
  --dual_prompt \
  --aux_prompts "${AUX_PROMPTS[@]}"

echo
echo "Done. Key outputs:"
echo "- Baseline prompt     : ${PROMPT_BASELINE}"
echo "- Baseline metrics    : ${BASELINE_EVAL_OUT}/metrics_${TEST_SPLIT}.json"
echo "- Joint prompt        : ${PROMPT_JOINT}"
echo "- Joint metrics       : ${JOINT_EVAL_OUT}/metrics_${TEST_SPLIT}.json"
echo "- Frozen-main prompt  : ${PROMPT_FROZEN}"
echo "- Frozen-main metrics : ${FROZEN_EVAL_OUT}/metrics_${TEST_SPLIT}.json"
echo "- Manual vis dir      : ${VIS_OUT}/vis"
