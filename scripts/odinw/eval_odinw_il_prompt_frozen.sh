#!/usr/bin/env bash
export TEXT_MODE=prompt
export PARAM_TUNE=frozen
exec "$(dirname "$0")/eval_odinw_il.sh"
