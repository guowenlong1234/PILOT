#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export ETPR1_R2R_EVAL_CKPT_DIR=${ETPR1_R2R_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_eval_best465000_20260816/checkpoints/rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft}
export ETPR1_R2R_EVAL_OUTPUT_ROOT=${ETPR1_R2R_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_eval_best465000_20260816/eval_watch_val_unseen}
export ETPR1_R2R_EVAL_EXP_NAME=${ETPR1_R2R_EVAL_EXP_NAME:-rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft_eval_watch}
export ETPR1_R2R_EVAL_PRETRAIN_PATH=${ETPR1_R2R_EVAL_PRETRAIN_PATH:-/home/a6000/gwl/ETP-R1/data/pretrain_resume_source_250000/best/model_best_step_465000.pt}
export ETPR1_R2R_EVAL_NUM_ENVIRONMENTS=${ETPR1_R2R_EVAL_NUM_ENVIRONMENTS:-8}
export ETPR1_R2R_EVAL_CHECKPOINT_ORDER=${ETPR1_R2R_EVAL_CHECKPOINT_ORDER:-ascending}

exec "${SCRIPT_DIR}/manage_rae_r2r_eval_watch_host.sh" "${1:-status}"
