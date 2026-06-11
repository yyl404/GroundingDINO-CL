#!/usr/bin/env bash
export TEXT_MODE=fixed
export PARAM_TUNE=full
exec "$(dirname "$0")/eval_odinw_il.sh"
