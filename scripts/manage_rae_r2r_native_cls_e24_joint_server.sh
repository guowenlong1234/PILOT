#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-status}
RUN_ID=${ETPR1_NATIVE_CLS_RUN_ID:-$(date +%Y%m%dT%H%M%S)}

export ETPR1_E24_JOINT_CONFIG_FILE=run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml
export ETPR1_E24_JOINT_EXP_NAME=${ETPR1_NATIVE_CLS_EXP_NAME:-etpr1_native_cls_e24_joint_sft}
export ETPR1_E24_JOINT_OUTPUT_ROOT=${ETPR1_NATIVE_CLS_OUTPUT_ROOT:-data/logs/active_lookahead/native_cls_e24_joint_sft}

case "$ACTION" in
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
        echo "Usage: $0 {smoke|start|resume|status|logs|tail|stop}" >&2
        exit 2
        ;;
esac
