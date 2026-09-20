#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACTION=${1:-status}
RUN_ID=${ETPR1_RXR_SFT_RUN_ID:-$(date +%Y%m%dT%H%M%S)}

export ETPR1_R2R_SFT_CONFIG_FILE=${ETPR1_RXR_SFT_CONFIG_FILE:-run_rxr/iter_train_rae_dino_sft.yaml}
export ETPR1_R2R_SFT_EXP_NAME=${ETPR1_RXR_SFT_EXP_NAME:-etpr1_rxr_rae_dino_sft}
export ETPR1_R2R_SFT_OUTPUT_ROOT=${ETPR1_RXR_SFT_OUTPUT_ROOT:-data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal}
export ETPR1_R2R_SFT_ITERS=${ETPR1_RXR_SFT_ITERS:-30000}
export ETPR1_R2R_SFT_LOG_EVERY=${ETPR1_RXR_SFT_LOG_EVERY:-200}
export ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS=${ETPR1_RXR_SFT_GRADIENT_ACCUMULATION_STEPS:-2}
export ETPR1_R2R_SFT_PRETRAINED_PATH=${ETPR1_RXR_SFT_PRETRAINED_PATH:-/home/gwl/project/etpr1/ETP-R1/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
export ETPR1_R2R_SFT_CHECKPOINT_SYNC_ENABLED=${ETPR1_RXR_SFT_CHECKPOINT_SYNC_ENABLED:-False}
export ETPR1_R2R_SFT_CHECKPOINT_SYNC_DESTINATION=${ETPR1_RXR_SFT_CHECKPOINT_SYNC_DESTINATION:-a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/checkpoints/etpr1_rxr_rae_dino_sft}
export ETPR1_R2R_SFT_NPROC_PER_NODE=${ETPR1_RXR_SFT_NPROC_PER_NODE:-2}
export ETPR1_R2R_SFT_NUM_ENVIRONMENTS=${ETPR1_RXR_SFT_NUM_ENVIRONMENTS:-6}
export ETPR1_R2R_SFT_BATCH_SIZE=${ETPR1_RXR_SFT_BATCH_SIZE:-6}
export ETPR1_R2R_SFT_CUDA_VISIBLE_DEVICES=${ETPR1_RXR_SFT_CUDA_VISIBLE_DEVICES:-0,1}
export ETPR1_R2R_SFT_MASTER_PORT=${ETPR1_RXR_SFT_MASTER_PORT:-24683}
export ETPR1_R2R_SFT_LR=1.5e-5
export ETPR1_R2R_SFT_SAMPLE_RATIO=0.75
export ETPR1_R2R_SFT_DECAY_INTERVAL=5000
export ETPR1_R2R_SFT_WARMUP_ITERS=1000
export ETPR1_R2R_SFT_MIN_LR_RATIO=0.6
export ETPR1_R2R_SFT_WAYPOINT_AUG=True

case "$ACTION" in
    smoke)
        export ETPR1_R2R_SFT_EXP_NAME=etpr1_rxr_rae_dino_sft_smoke
        export ETPR1_R2R_SFT_OUTPUT_ROOT="data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_smoke/${RUN_ID}"
        export ETPR1_R2R_SFT_ITERS=2
        export ETPR1_R2R_SFT_LOG_EVERY=2
        export ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS=1
        export ETPR1_R2R_SFT_CHECKPOINT_SYNC_ENABLED=False
        export ETPR1_R2R_SFT_NPROC_PER_NODE=1
        export ETPR1_R2R_SFT_NUM_ENVIRONMENTS=1
        export ETPR1_R2R_SFT_BATCH_SIZE=1
        export ETPR1_R2R_SFT_CUDA_VISIBLE_DEVICES=0
        exec "${SCRIPT_DIR}/manage_rae_r2r_sft_server.sh" start
        ;;
    smoke-resume)
        [ -n "${ETPR1_RXR_SFT_RUN_ID:-}" ] || {
            echo "ETPR1_RXR_SFT_RUN_ID is required for smoke-resume" >&2
            exit 2
        }
        export ETPR1_R2R_SFT_EXP_NAME=etpr1_rxr_rae_dino_sft_smoke
        export ETPR1_R2R_SFT_OUTPUT_ROOT="data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_smoke/${RUN_ID}"
        export ETPR1_R2R_SFT_ITERS=3
        export ETPR1_R2R_SFT_LOG_EVERY=1
        export ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS=1
        export ETPR1_R2R_SFT_CHECKPOINT_SYNC_ENABLED=False
        export ETPR1_R2R_SFT_NPROC_PER_NODE=1
        export ETPR1_R2R_SFT_NUM_ENVIRONMENTS=1
        export ETPR1_R2R_SFT_BATCH_SIZE=1
        export ETPR1_R2R_SFT_CUDA_VISIBLE_DEVICES=0
        exec "${SCRIPT_DIR}/manage_rae_r2r_sft_server.sh" resume
        ;;
    smoke-ddp)
        export ETPR1_R2R_SFT_EXP_NAME=etpr1_rxr_rae_dino_sft_ddp_smoke
        export ETPR1_R2R_SFT_OUTPUT_ROOT="data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_ddp_smoke/${RUN_ID}"
        export ETPR1_R2R_SFT_ITERS=2
        export ETPR1_R2R_SFT_LOG_EVERY=2
        export ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS=2
        export ETPR1_R2R_SFT_CHECKPOINT_SYNC_ENABLED=False
        export ETPR1_R2R_SFT_NPROC_PER_NODE=2
        export ETPR1_R2R_SFT_NUM_ENVIRONMENTS=1
        export ETPR1_R2R_SFT_BATCH_SIZE=1
        export ETPR1_R2R_SFT_CUDA_VISIBLE_DEVICES=0,1
        exec "${SCRIPT_DIR}/manage_rae_r2r_sft_server.sh" start
        ;;
    start|resume|status|logs|tail|stop)
        exec "${SCRIPT_DIR}/manage_rae_r2r_sft_server.sh" "$ACTION"
        ;;
    *)
        echo "Usage: $0 {smoke|smoke-resume|smoke-ddp|start|resume|status|logs|tail|stop}" >&2
        exit 2
        ;;
esac
