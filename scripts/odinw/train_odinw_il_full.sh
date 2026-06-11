#!/usr/bin/env bash
export SHOT_MODE=full
exec "$(dirname "$0")/train_odinw_il.sh"
