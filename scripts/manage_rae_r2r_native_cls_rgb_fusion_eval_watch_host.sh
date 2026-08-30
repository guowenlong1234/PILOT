#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export ETPR1_R2R_EVAL_CONFIG_FILE=${REPO_ROOT}/run_r2r/iter_train_rae_dino_native_cls_rgb_fusion.yaml
export ETPR1_R2R_EVAL_CKPT_DIR=${ETPR1_RGB_FUSION_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/raenwm_rgb_fusion/native_cls_sft/checkpoints/etpr1_native_cls_rgb_fusion_sft}
export ETPR1_R2R_EVAL_OUTPUT_ROOT=${ETPR1_RGB_FUSION_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/raenwm_rgb_fusion/native_cls_eval}
export ETPR1_R2R_EVAL_EXP_NAME=${ETPR1_RGB_FUSION_EVAL_EXP_NAME:-etpr1_native_cls_rgb_fusion_eval_watch}
export ETPR1_R2R_EVAL_PRETRAIN_PATH=${ETPR1_RGB_FUSION_EVAL_PRETRAIN_PATH:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
export ETPR1_R2R_EVAL_NUM_ENVIRONMENTS=${ETPR1_RGB_FUSION_EVAL_NUM_ENVIRONMENTS:-8}
export ETPR1_R2R_EVAL_EPISODE_COUNT=${ETPR1_RGB_FUSION_EVAL_EPISODE_COUNT:--1}
export ETPR1_R2R_EVAL_CHECKPOINT_ORDER=${ETPR1_RGB_FUSION_EVAL_CHECKPOINT_ORDER:-ascending}
# The training job publishes checkpoints by atomic rename but does not emit
# the optional .sha256 ready marker used by the standalone sync watcher.
export ETPR1_R2R_EVAL_READY_SHA_REQUIRED=False

exec "${SCRIPT_DIR}/manage_rae_r2r_eval_watch_host.sh" "${1:-status}"
