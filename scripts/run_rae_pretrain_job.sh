#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: run_rae_pretrain_job.sh <start|resume> <log_file>}
LOG_FILE=${2:?Usage: run_rae_pretrain_job.sh <start|resume> <log_file>}
MASTER_PORT=${ETPR1_PRETRAIN_MASTER_PORT:-29531}
OUTPUT_DIR=${ETPR1_PRETRAIN_OUTPUT_DIR:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768}

case "$MODE" in
    start|resume) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname -- "$LOG_FILE")"
cd "$REPO_ROOT"

identity_file=${LOG_FILE%.log}_source_identity.json
manifest_file=${LOG_FILE%.log}_source_manifest.sha256
identity_args=(
    --project-root "$REPO_ROOT"
    --manifest-output "$manifest_file"
    --identity-output "$identity_file"
)
if [ -n "${ETPR1_SOURCE_COMMIT:-}" ]; then
    identity_args+=(--requested-commit "$ETPR1_SOURCE_COMMIT")
fi
python scripts/rae_smoke_source_identity.py "${identity_args[@]}" >>"$LOG_FILE"
source_identity=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["identity"])' \
    "$identity_file")
source_kind=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["kind"])' \
    "$identity_file")

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "mode=$MODE"
    echo "repo_root=$REPO_ROOT"
    echo "output_dir=$OUTPUT_DIR"
    echo "master_port=$MASTER_PORT"
    echo "source_identity=$source_identity"
    echo "source_kind=$source_kind"
    echo "source_manifest=$manifest_file"
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
} >>"$LOG_FILE"

resume_args=()
if [ "$MODE" = resume ]; then
    resume_args=(--resume_checkpoint latest)
fi

pretrain_args=()
if [ -n "${ETPR1_PRETRAIN_TRAIN_BATCH_SIZE:-}" ]; then
    pretrain_args+=(--train_batch_size "$ETPR1_PRETRAIN_TRAIN_BATCH_SIZE")
fi
if [ -n "${ETPR1_PRETRAIN_GRADIENT_ACCUMULATION_STEPS:-}" ]; then
    pretrain_args+=(
        --gradient_accumulation_steps
        "$ETPR1_PRETRAIN_GRADIENT_ACCUMULATION_STEPS"
    )
fi
if [ -n "${ETPR1_PRETRAIN_N_WORKERS:-}" ]; then
    pretrain_args+=(--n_workers "$ETPR1_PRETRAIN_N_WORKERS")
fi
if [ -n "${ETPR1_PRETRAIN_VAL_N_WORKERS:-}" ]; then
    pretrain_args+=(--val_n_workers "$ETPR1_PRETRAIN_VAL_N_WORKERS")
fi
if [ -n "${ETPR1_PRETRAIN_DATALOADER_START_METHOD:-}" ]; then
    pretrain_args+=(
        --dataloader_start_method
        "$ETPR1_PRETRAIN_DATALOADER_START_METHOD"
    )
fi
if [ -n "${ETPR1_PRETRAIN_PREFETCH_FACTOR:-}" ]; then
    pretrain_args+=(--prefetch_factor "$ETPR1_PRETRAIN_PREFETCH_FACTOR")
fi
if [ -n "${ETPR1_PRETRAIN_FEATURE_CACHE_SIZE_MB:-}" ]; then
    pretrain_args+=(
        --feature_cache_size_mb
        "$ETPR1_PRETRAIN_FEATURE_CACHE_SIZE_MB"
    )
fi
if [ -n "${ETPR1_PRETRAIN_VAL_FEATURE_CACHE_SIZE_MB:-}" ]; then
    pretrain_args+=(
        --val_feature_cache_size_mb
        "$ETPR1_PRETRAIN_VAL_FEATURE_CACHE_SIZE_MB"
    )
fi
if [ "${ETPR1_PRETRAIN_DISABLE_THREAD_PREFETCH:-0}" = 1 ]; then
    pretrain_args+=(--no_thread_prefetch)
fi
if [ "${ETPR1_PRETRAIN_PIN_MEM:-0}" = 1 ] && \
    [ "${ETPR1_PRETRAIN_DISABLE_PIN_MEM:-0}" = 1 ]; then
    echo "Cannot enable and disable pretrain pin memory together" >&2
    exit 2
elif [ "${ETPR1_PRETRAIN_PIN_MEM:-0}" = 1 ]; then
    pretrain_args+=(--pin_mem)
elif [ "${ETPR1_PRETRAIN_DISABLE_PIN_MEM:-0}" = 1 ]; then
    pretrain_args+=(--no_pin_mem)
fi
if [ "${ETPR1_PRETRAIN_ALLOW_WORLD_SIZE_CHANGE:-0}" = 1 ]; then
    pretrain_args+=(--allow_world_size_change)
fi
if [ "${ETPR1_PRETRAIN_ALLOW_MODEL_CONFIG_PATH_CHANGE:-0}" = 1 ]; then
    pretrain_args+=(--allow_model_config_path_change)
fi

{
    echo "train_batch_size_override=${ETPR1_PRETRAIN_TRAIN_BATCH_SIZE:-default}"
    echo "gradient_accumulation_override=${ETPR1_PRETRAIN_GRADIENT_ACCUMULATION_STEPS:-default}"
    echo "n_workers_override=${ETPR1_PRETRAIN_N_WORKERS:-default}"
    echo "val_n_workers_override=${ETPR1_PRETRAIN_VAL_N_WORKERS:-default}"
    echo "dataloader_start_method=${ETPR1_PRETRAIN_DATALOADER_START_METHOD:-default}"
    echo "prefetch_factor=${ETPR1_PRETRAIN_PREFETCH_FACTOR:-default}"
    echo "feature_cache_size_mb=${ETPR1_PRETRAIN_FEATURE_CACHE_SIZE_MB:-default}"
    echo "val_feature_cache_size_mb=${ETPR1_PRETRAIN_VAL_FEATURE_CACHE_SIZE_MB:-default}"
    echo "disable_thread_prefetch=${ETPR1_PRETRAIN_DISABLE_THREAD_PREFETCH:-0}"
    echo "pin_mem=${ETPR1_PRETRAIN_PIN_MEM:-0}"
    echo "disable_pin_mem=${ETPR1_PRETRAIN_DISABLE_PIN_MEM:-0}"
    echo "allow_world_size_change=${ETPR1_PRETRAIN_ALLOW_WORLD_SIZE_CHANGE:-0}"
    echo "allow_model_config_path_change=${ETPR1_PRETRAIN_ALLOW_MODEL_CONFIG_PATH_CHANGE:-0}"
} >>"$LOG_FILE"

set +e
ETPR1_PRETRAIN_OUTPUT_DIR="$OUTPUT_DIR" \
    bash pretrain_src/run_pt/run_mix_rae_dino.bash \
    "$MASTER_PORT" "${resume_args[@]}" "${pretrain_args[@]}" \
    2>&1 | tee -a "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e

{
    echo "run_finished_at=$(date --iso-8601=seconds)"
    echo "exit_code=$exit_code"
} >>"$LOG_FILE"
exit "$exit_code"
