#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
EXP_NAME=${ETPR1_E24_JOINT_EXP_NAME:-etpr1_e24_joint_sft}
OUTPUT_ROOT=${ETPR1_E24_JOINT_OUTPUT_ROOT:-data/logs/active_lookahead/e24_joint_sft}
OUTPUT_PATH=${REPO_ROOT}/${OUTPUT_ROOT}
CHECKPOINT_DIR=${OUTPUT_PATH}/checkpoints/${EXP_NAME}
TRAIN_STATE_DIR=${CHECKPOINT_DIR}/train_states
SUPERVISOR_DIR=${OUTPUT_PATH}/supervisor_server
LATEST_LOG=${SUPERVISOR_DIR}/latest.log
PID_FILE=${SUPERVISOR_DIR}/trainer.pid
JOB_SCRIPT=${REPO_ROOT}/scripts/run_rae_r2r_e24_joint_server_job.sh

read_pid() {
    [ -s "$PID_FILE" ] || return 1
    local pid
    read -r pid <"$PID_FILE"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '%s\n' "$pid"
}

is_running() {
    local pid
    pid=$(read_pid) || return 1
    kill -0 "$pid" 2>/dev/null
}

gpus_are_idle() {
    local used
    while read -r used; do
        used=${used// /}
        [ "$used" -le 1024 ] || return 1
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
}

launch() {
    local mode=$1
    is_running && { echo "E24 joint SFT training is already running: PID $(read_pid)" >&2; exit 1; }
    gpus_are_idle || { echo "At least one training GPU is busy; refusing to launch." >&2; exit 1; }
    if [ "$mode" = resume ]; then
        local checkpoint iteration found=false
        for checkpoint in "$CHECKPOINT_DIR"/ckpt.iter*.pth; do
            [ -e "$checkpoint" ] || continue
            iteration=${checkpoint##*ckpt.iter}
            iteration=${iteration%.pth}
            if [ -s "$TRAIN_STATE_DIR/train_state.iter${iteration}.pth" ]; then
                found=true
                break
            fi
        done
        [ "$found" = true ] || { echo "No complete model/training-state pair found." >&2; exit 1; }
    fi
    mkdir -p "$SUPERVISOR_DIR"
    local timestamp log_file supervisor_log pid
    timestamp=$(date +%Y%m%dT%H%M%S)
    log_file=${SUPERVISOR_DIR}/${mode}_${timestamp}.log
    supervisor_log=${SUPERVISOR_DIR}/${mode}_${timestamp}.supervisor.log
    ln -sfn "$(basename -- "$log_file")" "$LATEST_LOG"
    nohup setsid "$JOB_SCRIPT" "$mode" "$log_file" </dev/null >"$supervisor_log" 2>&1 &
    pid=$!
    printf '%s\n' "$pid" >"$PID_FILE"
    sleep 2
    kill -0 "$pid" 2>/dev/null || { tail -n 100 "$log_file" >&2 || true; exit 1; }
    echo "Started E24 joint SFT training: PID $pid"
    echo "Log: $log_file"
}

show_status() {
    if is_running; then echo "process=running pid=$(read_pid)"; else echo "process=stopped"; fi
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
    echo "model_checkpoints=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' 2>/dev/null | wc -l)"
    echo "training_states=$(find "$TRAIN_STATE_DIR" -maxdepth 1 -type f -name 'train_state.iter*.pth' 2>/dev/null | wc -l)"
    [ -e "$LATEST_LOG" ] && { echo "latest_log=$(readlink -f -- "$LATEST_LOG")"; tail -n 30 "$LATEST_LOG"; }
    return 0
}

case "$ACTION" in
    start) launch start ;;
    smoke)
        run_id=${ETPR1_E24_JOINT_RUN_ID:-$(date +%Y%m%dT%H%M%S)}
        export ETPR1_E24_JOINT_EXP_NAME=etpr1_e24_joint_smoke
        export ETPR1_E24_JOINT_OUTPUT_ROOT="data/logs/active_lookahead/e24_joint_smoke/${run_id}"
        export ETPR1_E24_JOINT_ITERS=2
        export ETPR1_E24_JOINT_LOG_EVERY=2
        export ETPR1_E24_JOINT_SYNC_ENABLED=False
        export ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK=True
        export ETPR1_E24_JOINT_CONFIG_FILE=run_r2r/iter_train_rae_dino_e24_joint.yaml
        exec "$0" start
        ;;
    pilot)
        run_id=${ETPR1_E24_JOINT_RUN_ID:-$(date +%Y%m%dT%H%M%S)}
        export ETPR1_E24_JOINT_EXP_NAME=etpr1_e24_joint_pilot
        export ETPR1_E24_JOINT_OUTPUT_ROOT="data/logs/active_lookahead/e24_joint_pilot/${run_id}"
        export ETPR1_E24_JOINT_ITERS=400
        export ETPR1_E24_JOINT_LOG_EVERY=200
        export ETPR1_E24_JOINT_SYNC_ENABLED=False
        exec "$0" start
        ;;
    resume) launch resume ;;
    status) show_status ;;
    logs) [ -e "$LATEST_LOG" ] && tail -n 120 "$LATEST_LOG" ;;
    tail) exec tail -f "$LATEST_LOG" ;;
    stop)
        if is_running; then
            pid=$(read_pid)
            kill -TERM -- "-$pid"
            echo "Sent TERM to E24 joint SFT process group $pid."
        else
            echo "process=stopped"
        fi
        ;;
    *) echo "Usage: $0 {smoke|pilot|start|resume|status|logs|tail|stop}" >&2; exit 2 ;;
esac
