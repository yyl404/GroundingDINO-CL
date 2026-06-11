#!/usr/bin/env bash
export SHOT_MODE=10
exec "$(dirname "$0")/train_odinw_il.sh"
