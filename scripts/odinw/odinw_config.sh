# Shared OdinW-13 incremental learning configuration.
# Dataset order: dictionary sort (13 datasets).
# NOTE: source this file after cd to the repo root.

export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

ODINW_DATASETS=(
  AerialMaritimeDrone
  Aquarium
  CottontailRabbits
  EgoHands
  NorthAmericaMushrooms
  Packages
  PascalVOC
  Raccoon
  ShellfishOpenImages
  VehiclesOpenImages
  pistols
  pothole
  thermalDogsAndPeople
)

CONFIG_FILE="${CONFIG_FILE:-groundingdino/config/GroundingDINO_SwinT_OGC.py}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-weights/groundingdino_swint_ogc.pth}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
FEWSHOT_SEED="${FEWSHOT_SEED:-30}"

# SHOT_MODE: 1 | 3 | 5 | 10 | full
SHOT_MODE="${SHOT_MODE:-full}"

# TEXT_MODE: prompt | fixed
TEXT_MODE="${TEXT_MODE:-fixed}"

# PARAM_TUNE: full | lora | delta | frozen
PARAM_TUNE="${PARAM_TUNE:-full}"

RUN_TAG="${TEXT_MODE}_${PARAM_TUNE}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-outputs/odinw_il/${SHOT_MODE}/${RUN_TAG}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-outputs/eval_odinw_il/${RUN_TAG}}"

# Evaluation defaults for OdinW scripts.
EVAL_METRIC="${EVAL_METRIC:-mAP50-95}"
VIS_BATCH="${VIS_BATCH:-4}"
