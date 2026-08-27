#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export ETPR1_E24_EVAL_CONFIG_FILE=${REPO_ROOT}/run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml
export ETPR1_E24_EVAL_CKPT_DIR=${ETPR1_NATIVE_CLS_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/active_lookahead/native_cls_e24_joint_sft/checkpoints/etpr1_native_cls_e24_joint_sft}
export ETPR1_E24_EVAL_OUTPUT_ROOT=${ETPR1_NATIVE_CLS_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/active_lookahead/native_cls_e24_joint_eval}
export ETPR1_E24_EVAL_EXP_NAME=${ETPR1_NATIVE_CLS_EVAL_EXP_NAME:-etpr1_native_cls_e24_joint_eval_watch}
export ETPR1_E24_EVAL_PRETRAIN_PATH=${ETPR1_NATIVE_CLS_EVAL_PRETRAIN_PATH:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
export ETPR1_E24_EVAL_NUM_ENVIRONMENTS=${ETPR1_NATIVE_CLS_EVAL_NUM_ENVIRONMENTS:-8}
export ETPR1_E24_EVAL_EXPECTED_CHECKPOINTS=${ETPR1_NATIVE_CLS_EXPECTED_CHECKPOINTS:-50}
export ETPR1_E24_EVAL_EPISODE_COUNT=${ETPR1_NATIVE_CLS_EVAL_EPISODE_COUNT:--1}
export ETPR1_E24_EVAL_SELECTION_ENABLED=True
export ETPR1_E24_EVAL_CHECKPOINT_ORDER=${ETPR1_NATIVE_CLS_EVAL_CHECKPOINT_ORDER:-ascending}
export ETPR1_E24_EVAL_READY_SHA_REQUIRED=False

exec "${SCRIPT_DIR}/manage_e24_joint_eval_watch_host.sh" "${1:-status}"
