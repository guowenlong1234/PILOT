#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
EXP_NAME=${ETPR1_R2R_SFT_EXP_NAME:-rae_dinov2_r2r_sft}
OUTPUT_ROOT=${ETPR1_R2R_SFT_OUTPUT_ROOT:-data/logs/rae_dinov2/r2r_sft_formal}
if [[ "$OUTPUT_ROOT" = /* ]]; then
    OUTPUT_PATH=$OUTPUT_ROOT
else
    OUTPUT_PATH=${REPO_ROOT}/${OUTPUT_ROOT}
fi
CHECKPOINT_DIR=${OUTPUT_PATH}/checkpoints/${EXP_NAME}
TRAIN_STATE_DIR=${CHECKPOINT_DIR}/train_states
SUPERVISOR_DIR=${OUTPUT_PATH}/supervisor_server
LATEST_LOG=${SUPERVISOR_DIR}/latest.log
PID_FILE=${SUPERVISOR_DIR}/trainer.pid
JOB_SCRIPT=${REPO_ROOT}/scripts/run_rae_r2r_sft_server_job.sh

read_pid() {
    [ -f "$PID_FILE" ] || return 1
    local pid
    pid=$(tr -d '[:space:]' <"$PID_FILE")
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$pid"
}

is_running() {
    local pid
    pid=$(read_pid) || return 1
    kill -0 "$pid" 2>/dev/null
}

check_gpus_idle() {
    local used
    while read -r used; do
        used=${used// /}
        if [ "$used" -gt 1024 ]; then
            echo "A GPU already uses ${used} MiB; refusing to launch." >&2
            exit 1
        fi
    done < <(
        nvidia-smi --query-gpu=memory.used \
            --format=csv,noheader,nounits
    )
}

launch() {
    local mode=$1
    if is_running; then
        echo "SFT process is already running: PID $(read_pid)" >&2
        exit 1
    fi
    check_gpus_idle
    mkdir -p "$SUPERVISOR_DIR"

    if [ "$mode" = resume ]; then
        if [ ! -f "${CHECKPOINT_DIR}/ckpt.iter200.pth" ] \
            || [ ! -f "${TRAIN_STATE_DIR}/train_state.iter200.pth" ]; then
            echo "Complete iter200 checkpoint pair is missing." >&2
            exit 1
        fi
    fi

    local timestamp log_file supervisor_log pid
    timestamp=$(date +%Y%m%dT%H%M%S)
    log_file=${SUPERVISOR_DIR}/${mode}_${timestamp}.log
    supervisor_log=${SUPERVISOR_DIR}/${mode}_${timestamp}.supervisor.log
    ln -sfn "$(basename -- "$log_file")" "$LATEST_LOG"

    nohup setsid "$JOB_SCRIPT" "$mode" "$log_file" \
        </dev/null >"$supervisor_log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "SFT process exited during startup. Inspect $log_file" >&2
        tail -n 80 "$log_file" >&2 || true
        exit 1
    fi
    echo "Started SFT process: PID $pid"
    echo "Log: $log_file"
}

show_status() {
    if is_running; then
        echo "process=running pid=$(read_pid)"
    else
        echo "process=stopped"
    fi
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
    df -h "$OUTPUT_PATH" 2>/dev/null || df -h "$REPO_ROOT"
    if [ -e "$LATEST_LOG" ]; then
        echo "latest_log=$(readlink -f -- "$LATEST_LOG")"
        tail -n 25 "$LATEST_LOG"
    fi
}

case "$ACTION" in
    start)
        launch start
        ;;
    resume)
        launch resume
        ;;
    status)
        show_status
        ;;
    logs)
        [ -e "$LATEST_LOG" ] || {
            echo "No server SFT log found" >&2
            exit 1
        }
        tail -n 100 "$LATEST_LOG"
        ;;
    tail)
        [ -e "$LATEST_LOG" ] || {
            echo "No server SFT log found" >&2
            exit 1
        }
        exec tail -f "$LATEST_LOG"
        ;;
    stop)
        if ! is_running; then
            echo "SFT process is not running."
            exit 0
        fi
        pid=$(read_pid)
        kill -TERM -- "-$pid"
        echo "Sent TERM to SFT process group $pid."
        ;;
    *)
        echo "Usage: $0 {start|resume|status|logs|tail|stop}" >&2
        exit 2
        ;;
esac
