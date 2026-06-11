#!/usr/bin/env bash
export TEXT_MODE=prompt
export PARAM_TUNE=lora
exec "$(dirname "$0")/eval_odinw_il.sh"
