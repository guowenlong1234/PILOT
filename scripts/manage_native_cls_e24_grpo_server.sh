#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATASET=${1:-}
ACTION=${2:-status}
case "$DATASET" in r2r|rxr) ;; *) echo "Usage: $0 <r2r|rxr> {start|resume|status|logs|tail|stop}" >&2; exit 2 ;; esac

if [ "$DATASET" = r2r ]; then
    EXP_NAME=${ETPR1_R2R_ACTIVE_GRPO_EXP_NAME:-r2r_native_cls_e24_frozen_grpo}
    OUTPUT_ROOT=${ETPR1_R2R_ACTIVE_GRPO_OUTPUT_ROOT:-data/logs/active_lookahead/r2r_native_cls_e24_grpo}
else
    EXP_NAME=${ETPR1_RXR_ACTIVE_GRPO_EXP_NAME:-rxr_native_cls_e24_frozen_grpo}
    OUTPUT_ROOT=${ETPR1_RXR_ACTIVE_GRPO_OUTPUT_ROOT:-data/logs/active_lookahead/rxr_native_cls_e24_grpo}
fi
[[ "$OUTPUT_ROOT" = /* ]] && OUTPUT_PATH=$OUTPUT_ROOT || OUTPUT_PATH=${REPO_ROOT}/${OUTPUT_ROOT}
CHECKPOINT_DIR=${OUTPUT_PATH}/checkpoints/${EXP_NAME}
TRAIN_STATE_DIR=${CHECKPOINT_DIR}/train_states
SUPERVISOR_DIR=${OUTPUT_PATH}/supervisor_server/${EXP_NAME}
LATEST_LOG=${SUPERVISOR_DIR}/latest.log
PID_FILE=${SUPERVISOR_DIR}/trainer.pid
JOB_SCRIPT=${REPO_ROOT}/scripts/run_native_cls_e24_grpo_server_job.sh

read_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid=$(tr -d '[:space:]' <"$PID_FILE")
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$pid"
}

is_running() { local pid; pid=$(read_pid) || return 1; kill -0 "$pid" 2>/dev/null; }

check_gpus_idle() {
    local rows used
    mapfile -t rows < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    [ "${#rows[@]}" -eq 2 ] || { echo "Expected exactly two training GPUs." >&2; exit 1; }
    for row in "${rows[@]}"; do
        used=${row##*,}; used=${used// /}
        [ "$used" -le 1024 ] || { echo "GPU already uses ${used} MiB." >&2; exit 1; }
    done
}

require_pair() {
    local model iteration
    for model in "$CHECKPOINT_DIR"/ckpt.iter*.pth; do
        [ -e "$model" ] || continue
        iteration=${model##*ckpt.iter}; iteration=${iteration%.pth}
        [ -f "$TRAIN_STATE_DIR/train_state.iter${iteration}.pth" ] && return 0
    done
    echo "No complete active-GRPO checkpoint pair found." >&2
    exit 1
}

launch() {
    local mode=$1 timestamp log_file supervisor_log pid
    is_running && { echo "Active GRPO is already running: $(read_pid)" >&2; exit 1; }
    check_gpus_idle
    if [ "$mode" = start ]; then
        compgen -G "$CHECKPOINT_DIR/ckpt.iter*.pth" >/dev/null && { echo "Output already contains checkpoints; use resume." >&2; exit 1; }
    else
        require_pair
    fi
    mkdir -p "$SUPERVISOR_DIR"
    timestamp=$(date +%Y%m%dT%H%M%S)
    log_file=${SUPERVISOR_DIR}/${mode}_${timestamp}.log
    supervisor_log=${SUPERVISOR_DIR}/${mode}_${timestamp}.supervisor.log
    ln -sfn "$(basename -- "$log_file")" "$LATEST_LOG"
    nohup setsid "$JOB_SCRIPT" "$DATASET" "$mode" "$log_file" </dev/null >"$supervisor_log" 2>&1 &
    pid=$!; printf '%s\n' "$pid" >"$PID_FILE"
    sleep 2
    kill -0 "$pid" 2>/dev/null || { tail -n 80 "$log_file" >&2 || true; exit 1; }
    echo "Started $DATASET active GRPO: pid=$pid log=$log_file"
}

status() {
    is_running && echo "process=running pid=$(read_pid)" || echo "process=stopped"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    [ -e "$LATEST_LOG" ] && { echo "latest_log=$(readlink -f -- "$LATEST_LOG")"; tail -n 25 "$LATEST_LOG"; }
}

case "$ACTION" in
    start) launch start ;;
    resume) launch resume ;;
    status) status ;;
    logs) [ -e "$LATEST_LOG" ] && tail -n 100 "$LATEST_LOG" || { echo "No log found" >&2; exit 1; } ;;
    tail) [ -e "$LATEST_LOG" ] && exec tail -f "$LATEST_LOG" || { echo "No log found" >&2; exit 1; } ;;
    stop) is_running || { echo "Active GRPO is not running."; exit 0; }; pid=$(read_pid); kill -TERM -- "-$pid"; echo "Sent TERM to process group $pid." ;;
    *) echo "Usage: $0 <r2r|rxr> {start|resume|status|logs|tail|stop}" >&2; exit 2 ;;
esac
