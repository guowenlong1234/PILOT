#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-status}
RUN_ID=${ETPR1_NATIVE_CLS_RUN_ID:-$(date +%Y%m%dT%H%M%S)}

export ETPR1_E24_JOINT_CONFIG_FILE=run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml
export ETPR1_E24_JOINT_EXP_NAME=${ETPR1_NATIVE_CLS_EXP_NAME:-etpr1_native_cls_e24_joint_sft}
export ETPR1_E24_JOINT_OUTPUT_ROOT=${ETPR1_NATIVE_CLS_OUTPUT_ROOT:-data/logs/active_lookahead/native_cls_e24_joint_sft}
export ETPR1_E24_JOINT_SYNC_DESTINATION=${ETPR1_NATIVE_CLS_SYNC_DESTINATION:-a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/data/logs/active_lookahead/native_cls_e24_joint_sft/checkpoints/${ETPR1_E24_JOINT_EXP_NAME}}

case "$ACTION" in
    smoke-single)
        export ETPR1_E24_JOINT_EXP_NAME=etpr1_native_cls_e24_joint_single_smoke
        export ETPR1_E24_JOINT_OUTPUT_ROOT="data/logs/active_lookahead/native_cls_e24_joint_single_smoke/${RUN_ID}"
        export ETPR1_E24_JOINT_ITERS=2
        export ETPR1_E24_JOINT_LOG_EVERY=2
        export ETPR1_E24_JOINT_SYNC_ENABLED=False
        export ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK=True
        export ETPR1_E24_JOINT_NPROC_PER_NODE=1
        export ETPR1_E24_JOINT_NUM_ENVIRONMENTS=1
        export ETPR1_E24_JOINT_BATCH_SIZE=1
        export ETPR1_E24_JOINT_CUDA_VISIBLE_DEVICES=0
        exec "${SCRIPT_DIR}/manage_rae_r2r_e24_joint_server.sh" start
        ;;
    smoke)
        export ETPR1_E24_JOINT_EXP_NAME=etpr1_native_cls_e24_joint_smoke
        export ETPR1_E24_JOINT_OUTPUT_ROOT="data/logs/active_lookahead/native_cls_e24_joint_smoke/${RUN_ID}"
        export ETPR1_E24_JOINT_ITERS=2
        export ETPR1_E24_JOINT_LOG_EVERY=2
        export ETPR1_E24_JOINT_SYNC_ENABLED=False
        export ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK=True
        exec "${SCRIPT_DIR}/manage_rae_r2r_e24_joint_server.sh" start
        ;;
    start|resume|status|logs|tail|stop)
        exec "${SCRIPT_DIR}/manage_rae_r2r_e24_joint_server.sh" "$ACTION"
        ;;
    *)
        echo "Usage: $0 {smoke-single|smoke|start|resume|status|logs|tail|stop}" >&2
        exit 2
        ;;
esac
