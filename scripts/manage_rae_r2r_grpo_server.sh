#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
EXP_NAME=${ETPR1_R2R_GRPO_EXP_NAME:-rae_dinov2_etpnav_cls_768_legacy452500_sft8000_r2r_grpo}
OUTPUT_ROOT=${ETPR1_R2R_GRPO_OUTPUT_ROOT:-data/logs/rae_dinov2_etpnav_cls_768/r2r_grpo_legacy452500_sft8000}
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
JOB_SCRIPT=${REPO_ROOT}/scripts/run_rae_r2r_grpo_server_job.sh

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

check_two_gpus_idle() {
    local gpu_rows used
    mapfile -t gpu_rows < <(
        nvidia-smi --query-gpu=index,memory.used \
            --format=csv,noheader,nounits
    )
    if [ "${#gpu_rows[@]}" -ne 2 ]; then
        echo "Expected exactly two training GPUs, found ${#gpu_rows[@]}." >&2
        exit 1
    fi
    for row in "${gpu_rows[@]}"; do
        used=${row##*,}
        used=${used// /}
        if [ "$used" -gt 1024 ]; then
            echo "A GPU already uses ${used} MiB; refusing to launch GRPO." >&2
            exit 1
        fi
    done
}

require_complete_pair() {
    local checkpoint iteration found=false
    for checkpoint in "${CHECKPOINT_DIR}"/ckpt.iter*.pth; do
        [ -e "$checkpoint" ] || continue
        iteration=${checkpoint##*ckpt.iter}
        iteration=${iteration%.pth}
        if [ -f "${TRAIN_STATE_DIR}/train_state.iter${iteration}.pth" ]; then
            found=true
        fi
    done
    if [ "$found" != true ]; then
        echo "No complete GRPO model/training-state checkpoint pair found." >&2
        exit 1
    fi
}

launch() {
    local mode=$1
    if is_running; then
        echo "GRPO process is already running: PID $(read_pid)" >&2
        exit 1
    fi
    check_two_gpus_idle
    if [ "$mode" = start ]; then
        if compgen -G "${CHECKPOINT_DIR}/ckpt.iter*.pth" >/dev/null; then
            echo "GRPO output already contains checkpoints; use resume or a new output root." >&2
            exit 1
        fi
    else
        require_complete_pair
    fi
    mkdir -p "$SUPERVISOR_DIR"

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
        echo "GRPO process exited during startup. Inspect $log_file" >&2
        tail -n 80 "$log_file" >&2 || true
        exit 1
    fi
    echo "Started GRPO process: PID $pid"
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
    if [ -d "$CHECKPOINT_DIR" ]; then
        printf 'model_checkpoints='
        find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' | wc -l
        printf 'training_states='
        find "$TRAIN_STATE_DIR" -maxdepth 1 -type f -name 'train_state.iter*.pth' 2>/dev/null | wc -l
    fi
    if [ -e "$LATEST_LOG" ]; then
        echo "latest_log=$(readlink -f -- "$LATEST_LOG")"
        tail -n 25 "$LATEST_LOG"
    fi
}

case "$ACTION" in
    start) launch start ;;
    resume) launch resume ;;
    status) show_status ;;
    logs)
        [ -e "$LATEST_LOG" ] || { echo "No GRPO log found" >&2; exit 1; }
        tail -n 100 "$LATEST_LOG"
        ;;
    tail)
        [ -e "$LATEST_LOG" ] || { echo "No GRPO log found" >&2; exit 1; }
        exec tail -f "$LATEST_LOG"
        ;;
    stop)
        if ! is_running; then
            echo "GRPO process is not running."
            exit 0
        fi
        pid=$(read_pid)
        kill -TERM -- "-$pid"
        echo "Sent TERM to GRPO process group $pid."
        ;;
    *)
        echo "Usage: $0 {start|resume|status|logs|tail|stop}" >&2
        exit 2
        ;;
esac
