#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export ETPR1_R2R_EVAL_CONFIG_FILE=${REPO_ROOT}/run_rxr/iter_train_rae_dino_sft.yaml
export ETPR1_R2R_EVAL_CKPT_DIR=${ETPR1_RXR_BASE_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/checkpoints/etpr1_rxr_rae_dino_sft}
export ETPR1_R2R_EVAL_OUTPUT_ROOT=${ETPR1_RXR_BASE_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/eval_watch_val_unseen}
export ETPR1_R2R_EVAL_EXP_NAME=${ETPR1_RXR_BASE_EVAL_EXP_NAME:-etpr1_rxr_rae_dino_sft_eval_watch}
export ETPR1_R2R_EVAL_PRETRAIN_PATH=${ETPR1_RXR_EVAL_PRETRAIN_PATH:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
export ETPR1_R2R_EVAL_NUM_ENVIRONMENTS=${ETPR1_RXR_EVAL_NUM_ENVIRONMENTS:-3}
export ETPR1_R2R_EVAL_EPISODE_COUNT=${ETPR1_RXR_EVAL_EPISODE_COUNT:--1}
export ETPR1_R2R_EVAL_CHECKPOINT_ORDER=${ETPR1_RXR_EVAL_CHECKPOINT_ORDER:-ascending}
export ETPR1_R2R_EVAL_BLOCKING_PROCESS_PATTERN=${ETPR1_RXR_EVAL_BLOCKING_PROCESS_PATTERN:-run_rxr/iter_train_rae_dino_native_cls_e24_joint.yaml}
export ETPR1_R2R_EVAL_READY_SHA_REQUIRED=True

exec "${SCRIPT_DIR}/manage_rae_r2r_eval_watch_host.sh" "${1:-status}"
