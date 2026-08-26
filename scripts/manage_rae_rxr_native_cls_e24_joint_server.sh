#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-status}
RUN_ID=${ETPR1_RXR_JOINT_RUN_ID:-$(date +%Y%m%dT%H%M%S)}

export ETPR1_E24_JOINT_CONFIG_FILE=run_rxr/iter_train_rae_dino_native_cls_e24_joint.yaml
export ETPR1_E24_JOINT_EXP_NAME=${ETPR1_RXR_JOINT_EXP_NAME:-etpr1_rxr_native_cls_e24_joint_sft}
export ETPR1_E24_JOINT_OUTPUT_ROOT=${ETPR1_RXR_JOINT_OUTPUT_ROOT:-data/logs/active_lookahead/rxr_native_cls_e24_joint_sft}
export ETPR1_E24_JOINT_PRETRAIN_PATH=${ETPR1_RXR_JOINT_PRETRAIN_PATH:-/home/gwl/project/etpr1/ETP-R1/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
export ETPR1_E24_JOINT_SYNC_ENABLED=${ETPR1_RXR_JOINT_SYNC_ENABLED:-False}
export ETPR1_E24_JOINT_SYNC_DESTINATION=${ETPR1_RXR_JOINT_SYNC_DESTINATION:-a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/data/logs/active_lookahead/rxr_native_cls_e24_joint_sft/checkpoints/etpr1_rxr_native_cls_e24_joint_sft}
export ETPR1_E24_JOINT_ITERS=${ETPR1_RXR_JOINT_ITERS:-10000}
export ETPR1_E24_JOINT_LOG_EVERY=${ETPR1_RXR_JOINT_LOG_EVERY:-200}
export ETPR1_E24_JOINT_NPROC_PER_NODE=${ETPR1_RXR_JOINT_NPROC_PER_NODE:-2}
export ETPR1_E24_JOINT_NUM_ENVIRONMENTS=${ETPR1_RXR_JOINT_NUM_ENVIRONMENTS:-2}
export ETPR1_E24_JOINT_BATCH_SIZE=${ETPR1_RXR_JOINT_BATCH_SIZE:-2}
export ETPR1_E24_JOINT_CUDA_VISIBLE_DEVICES=${ETPR1_RXR_JOINT_CUDA_VISIBLE_DEVICES:-0,1}
export ETPR1_E24_JOINT_MASTER_PORT=${ETPR1_RXR_JOINT_MASTER_PORT:-24734}

case "$ACTION" in
    smoke)
        export ETPR1_E24_JOINT_EXP_NAME=etpr1_rxr_native_cls_e24_joint_smoke
        export ETPR1_E24_JOINT_OUTPUT_ROOT="data/logs/active_lookahead/rxr_native_cls_e24_joint_smoke/${RUN_ID}"
        export ETPR1_E24_JOINT_ITERS=2
        export ETPR1_E24_JOINT_LOG_EVERY=2
        export ETPR1_E24_JOINT_SYNC_ENABLED=False
        export ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK=True
        export ETPR1_E24_JOINT_NPROC_PER_NODE=1
        export ETPR1_E24_JOINT_NUM_ENVIRONMENTS=1
        export ETPR1_E24_JOINT_BATCH_SIZE=1
        export ETPR1_E24_JOINT_CUDA_VISIBLE_DEVICES=0
        export ETPR1_RXR_ALLOW_SMOKE_BASE=True
        exec "${SCRIPT_DIR}/manage_rae_r2r_e24_joint_server.sh" start
        ;;
    smoke-resume)
        [ -n "${ETPR1_RXR_JOINT_RUN_ID:-}" ] || {
            echo "ETPR1_RXR_JOINT_RUN_ID is required for smoke-resume" >&2
            exit 2
        }
        export ETPR1_E24_JOINT_EXP_NAME=etpr1_rxr_native_cls_e24_joint_smoke
        export ETPR1_E24_JOINT_OUTPUT_ROOT="data/logs/active_lookahead/rxr_native_cls_e24_joint_smoke/${RUN_ID}"
        export ETPR1_E24_JOINT_ITERS=3
        export ETPR1_E24_JOINT_LOG_EVERY=1
        export ETPR1_E24_JOINT_SYNC_ENABLED=False
        export ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK=True
        export ETPR1_E24_JOINT_NPROC_PER_NODE=1
        export ETPR1_E24_JOINT_NUM_ENVIRONMENTS=1
        export ETPR1_E24_JOINT_BATCH_SIZE=1
        export ETPR1_E24_JOINT_CUDA_VISIBLE_DEVICES=0
        export ETPR1_RXR_ALLOW_SMOKE_BASE=True
        exec "${SCRIPT_DIR}/manage_rae_r2r_e24_joint_server.sh" resume
        ;;
    smoke-ddp)
        export ETPR1_E24_JOINT_EXP_NAME=etpr1_rxr_native_cls_e24_joint_ddp_smoke
        export ETPR1_E24_JOINT_OUTPUT_ROOT="data/logs/active_lookahead/rxr_native_cls_e24_joint_ddp_smoke/${RUN_ID}"
        export ETPR1_E24_JOINT_ITERS=2
        export ETPR1_E24_JOINT_LOG_EVERY=2
        export ETPR1_E24_JOINT_SYNC_ENABLED=False
        export ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK=True
        export ETPR1_E24_JOINT_NPROC_PER_NODE=2
        export ETPR1_E24_JOINT_NUM_ENVIRONMENTS=1
        export ETPR1_E24_JOINT_BATCH_SIZE=1
        export ETPR1_E24_JOINT_CUDA_VISIBLE_DEVICES=0,1
        export ETPR1_RXR_ALLOW_SMOKE_BASE=True
        exec "${SCRIPT_DIR}/manage_rae_r2r_e24_joint_server.sh" start
        ;;
    start|resume|status|logs|tail|stop)
        exec "${SCRIPT_DIR}/manage_rae_r2r_e24_joint_server.sh" "$ACTION"
        ;;
    *)
        echo "Usage: $0 {smoke|smoke-resume|smoke-ddp|start|resume|status|logs|tail|stop}" >&2
        exit 2
        ;;
esac
