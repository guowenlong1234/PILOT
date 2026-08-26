#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
RUN_ID=${ETPR1_NATIVE_CLS_EVAL_RUN_ID:-$(date +%Y%m%dT%H%M%S)}

export ETPR1_E24_EVAL_CONFIG_FILE=${REPO_ROOT}/run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml
export ETPR1_E24_EVAL_CKPT_DIR=${ETPR1_NATIVE_CLS_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/active_lookahead/native_cls_e24_joint_smoke/checkpoint_for_eval}
export ETPR1_E24_EVAL_OUTPUT_ROOT=${ETPR1_NATIVE_CLS_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/active_lookahead/native_cls_e24_single_episode_eval/${RUN_ID}}
export ETPR1_E24_EVAL_EXP_NAME=etpr1_native_cls_e24_single_episode
export ETPR1_E24_EVAL_EXPECTED_CHECKPOINTS=1
export ETPR1_E24_EVAL_EPISODE_COUNT=1
export ETPR1_E24_EVAL_RESULT_EPISODE_COUNT=1
export ETPR1_E24_EVAL_SELECTION_ENABLED=False
export ETPR1_E24_EVAL_NUM_ENVIRONMENTS=1
export ETPR1_E24_EVAL_CHECKPOINT_ORDER=ascending

exec "${SCRIPT_DIR}/manage_e24_joint_eval_watch_host.sh" "$ACTION"
