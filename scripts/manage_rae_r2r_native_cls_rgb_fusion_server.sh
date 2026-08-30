#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-status}
RUN_ID=${ETPR1_RGB_FUSION_RUN_ID:-$(date +%Y%m%dT%H%M%S)}
EXP_NAME=${ETPR1_RGB_FUSION_EXP_NAME:-etpr1_native_cls_rgb_fusion_sft}
OUTPUT_ROOT=${ETPR1_RGB_FUSION_OUTPUT_ROOT:-data/logs/raenwm_rgb_fusion/native_cls_sft}
DELEGATED_ACTION=$ACTION

case "$ACTION" in
    smoke-single)
        EXP_NAME=etpr1_native_cls_rgb_fusion_single_smoke
        OUTPUT_ROOT="data/logs/raenwm_rgb_fusion/native_cls_single_smoke/${RUN_ID}"
        export ETPR1_RGB_FUSION_ITERS=2
        export ETPR1_RGB_FUSION_LOG_EVERY=2
        export ETPR1_RGB_FUSION_SYNC_ENABLED=False
        export ETPR1_RGB_FUSION_NPROC_PER_NODE=1
        export ETPR1_RGB_FUSION_NUM_ENVIRONMENTS=1
        export ETPR1_RGB_FUSION_BATCH_SIZE=1
        export ETPR1_RGB_FUSION_CUDA_VISIBLE_DEVICES=0
        DELEGATED_ACTION=start
        ;;
    smoke)
        EXP_NAME=etpr1_native_cls_rgb_fusion_smoke
        OUTPUT_ROOT="data/logs/raenwm_rgb_fusion/native_cls_smoke/${RUN_ID}"
        export ETPR1_RGB_FUSION_ITERS=2
        export ETPR1_RGB_FUSION_LOG_EVERY=2
        export ETPR1_RGB_FUSION_SYNC_ENABLED=False
        DELEGATED_ACTION=start
        ;;
    start|resume|status|logs|tail|stop) ;;
    *)
        echo "Usage: $0 {smoke-single|smoke|start|resume|status|logs|tail|stop}" >&2
        exit 2
        ;;
esac

export ETPR1_RGB_FUSION_EXP_NAME="$EXP_NAME"
export ETPR1_RGB_FUSION_OUTPUT_ROOT="$OUTPUT_ROOT"
export ETPR1_R2R_SFT_EXP_NAME="$EXP_NAME"
export ETPR1_R2R_SFT_OUTPUT_ROOT="$OUTPUT_ROOT"
export ETPR1_R2R_SFT_JOB_SCRIPT="${SCRIPT_DIR}/run_rae_r2r_native_cls_rgb_fusion_server_job.sh"
exec "${SCRIPT_DIR}/manage_rae_r2r_sft_server.sh" "$DELEGATED_ACTION"
