#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
PILOT_RUN_ID=${ETPR1_E24_PILOT_RUN_ID:-20260825T165434}
PILOT_ROOT=${ETPR1_E24_PILOT_ROOT:-${REPO_ROOT}/data/logs/active_lookahead/e24_joint_pilot/${PILOT_RUN_ID}}

export ETPR1_E24_EVAL_CKPT_DIR=${ETPR1_E24_EVAL_CKPT_DIR:-${PILOT_ROOT}/checkpoints/etpr1_e24_joint_pilot}
export ETPR1_E24_EVAL_OUTPUT_ROOT=${ETPR1_E24_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/active_lookahead/e24_joint_pilot_eval/${PILOT_RUN_ID}}
export ETPR1_E24_EVAL_EXP_NAME=${ETPR1_E24_EVAL_EXP_NAME:-etpr1_e24_joint_pilot_eval}
export ETPR1_E24_EVAL_EXPECTED_CHECKPOINTS=1
export ETPR1_E24_EVAL_EPISODE_COUNT=16
export ETPR1_E24_EVAL_SELECTION_ENABLED=False
export ETPR1_E24_EVAL_CHECKPOINT_ORDER=ascending
export ETPR1_E24_EVAL_NUM_ENVIRONMENTS=${ETPR1_E24_EVAL_NUM_ENVIRONMENTS:-8}

exec "${SCRIPT_DIR}/manage_e24_joint_eval_watch_host.sh" "$ACTION"
