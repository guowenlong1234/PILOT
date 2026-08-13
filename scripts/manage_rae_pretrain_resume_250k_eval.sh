#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export ETPR1_PRETRAIN_TMUX_SESSION=etpr1-rae-pretrain-resume-250k
export ETPR1_PRETRAIN_OUTPUT_DIR=/home/a6000/gwl/ETP-R1/data/pretrain_resume_source_250000
export ETPR1_PRETRAIN_MASTER_PORT=29531
export ETPR1_PRETRAIN_TRAIN_BATCH_SIZE=16
export ETPR1_PRETRAIN_GRADIENT_ACCUMULATION_STEPS=4
export ETPR1_PRETRAIN_ALLOW_WORLD_SIZE_CHANGE=1
export ETPR1_PRETRAIN_ALLOW_MODEL_CONFIG_PATH_CHANGE=1

exec "${SCRIPT_DIR}/manage_rae_pretrain_host.sh" "${1:-status}"
