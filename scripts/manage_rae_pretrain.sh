#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
SESSION=${ETPR1_PRETRAIN_TMUX_SESSION:-etpr1-rae-pretrain}
OUTPUT_DIR=${ETPR1_PRETRAIN_OUTPUT_DIR:-pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp}
if [[ "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_PATH=$OUTPUT_DIR
else
    OUTPUT_PATH=${REPO_ROOT}/${OUTPUT_DIR}
fi
SUPERVISOR_DIR=${OUTPUT_PATH}/supervisor
LATEST_LOG=${SUPERVISOR_DIR}/latest.log
JOB_SCRIPT=${REPO_ROOT}/scripts/run_rae_pretrain_job.sh

require_tmux() {
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux is not installed in gwl-etpr1-rae" >&2
        exit 1
    }
}

session_exists() {
    tmux has-session -t "$SESSION" 2>/dev/null
}

check_gpu_idle() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' ')
    if [ "$used" -gt 1024 ]; then
        echo "GPU already uses ${used} MiB; refusing to launch another long job." >&2
        exit 1
    fi
}

launch() {
    local mode=$1
    require_tmux
    if session_exists; then
        echo "tmux session already exists: $SESSION" >&2
        exit 1
    fi
    check_gpu_idle
    mkdir -p "$SUPERVISOR_DIR"
    local timestamp log_file
    timestamp=$(date +%Y%m%dT%H%M%S)
    log_file=${SUPERVISOR_DIR}/${mode}_${timestamp}.log
    ln -sfn "$(basename -- "$log_file")" "$LATEST_LOG"

    if [ "$mode" = start ]; then
        if compgen -G "${OUTPUT_PATH}/ckpts/model_step_*.pt" >/dev/null; then
            echo "Existing checkpoints found; use resume instead of start." >&2
            exit 1
        fi
    else
        if ! compgen -G "${OUTPUT_PATH}/ckpts/train_state_*.pt" >/dev/null; then
            echo "No resumable train_state checkpoint found in ${OUTPUT_PATH}/ckpts" >&2
            exit 1
        fi
    fi

    local job_command
    printf -v job_command '%q %q %q' "$JOB_SCRIPT" "$mode" "$log_file"
    tmux new-session -d -s "$SESSION" "$job_command"
    sleep 1
    if ! session_exists; then
        echo "Training session exited during startup. Inspect ${log_file}" >&2
        tail -n 80 "$log_file" >&2 || true
        exit 1
    fi
    echo "Started tmux session: $SESSION"
    echo "Log: $log_file"
}

show_status() {
    require_tmux
    if session_exists; then
        echo "session=running name=$SESSION"
        tmux list-panes -t "$SESSION" \
            -F 'pane_pid=#{pane_pid} pane_dead=#{pane_dead} command=#{pane_current_command}'
    else
        echo "session=stopped name=$SESSION"
    fi
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
    df -h "$OUTPUT_PATH" 2>/dev/null || df -h "$REPO_ROOT"
    find "${OUTPUT_PATH}/ckpts" -maxdepth 1 \
        \( -name 'model_step_*.pt' -o -name 'train_state_*.pt' \) \
        -printf '%T@ %f %s bytes\n' 2>/dev/null | sort -nr | head -n 8 || true
    if [ -e "$LATEST_LOG" ]; then
        echo "latest_log=$(readlink -f -- "$LATEST_LOG")"
        tail -n 20 "$LATEST_LOG"
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
        [ -e "$LATEST_LOG" ] || { echo "No supervisor log found" >&2; exit 1; }
        tail -n 100 "$LATEST_LOG"
        ;;
    tail)
        [ -e "$LATEST_LOG" ] || { echo "No supervisor log found" >&2; exit 1; }
        exec tail -f "$LATEST_LOG"
        ;;
    attach)
        require_tmux
        exec tmux attach-session -t "$SESSION"
        ;;
    stop)
        require_tmux
        session_exists || { echo "Session is not running: $SESSION"; exit 0; }
        tmux send-keys -t "$SESSION" C-c
        for _ in $(seq 1 30); do
            session_exists || { echo "Training stopped cleanly."; exit 0; }
            sleep 1
        done
        echo "Training did not stop within 30 seconds; no force kill was sent." >&2
        exit 1
        ;;
    *)
        echo "Usage: $0 {start|resume|status|logs|tail|attach|stop}" >&2
        exit 2
        ;;
esac
