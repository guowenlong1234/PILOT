#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
CONTAINER=${ETPR1_SECOND_EVAL_HANDOFF_CONTAINER:-gwl-etpr1-rae}
POLL_SECONDS=${ETPR1_SECOND_EVAL_HANDOFF_POLL_SECONDS:-30}
GPU_IDLE_LIMIT_MIB=${ETPR1_SECOND_EVAL_HANDOFF_GPU_IDLE_LIMIT_MIB:-1024}
FIRST_EXPECTED_RESULTS=${ETPR1_SECOND_EVAL_HANDOFF_FIRST_EXPECTED_RESULTS:-75}
FIRST_RESULT_DIR=${ETPR1_SECOND_EVAL_HANDOFF_FIRST_RESULT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815/eval_watch_val_unseen/results/rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft_eval_watch/eval_results}
SECOND_WATCH=${SCRIPT_DIR}/manage_eval_best465000_r2r_sft_eval_watch_host.sh
MONITOR_ROOT=${ETPR1_SECOND_EVAL_HANDOFF_ROOT:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/second_sft_eval_handoff}
PID_FILE=${MONITOR_ROOT}/watch.pid
LOG_FILE=${MONITOR_ROOT}/watch.log
LAUNCHED_FILE=${MONITOR_ROOT}/launched.env

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

valid_first_result_count() {
    docker exec -e FIRST_RESULT_DIR="$FIRST_RESULT_DIR" \
        "$CONTAINER" bash -lc '
            source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh
            conda activate etpr1_rae
            python - <<"PY"
import glob
import json
import os

paths = glob.glob(os.path.join(
    os.environ["FIRST_RESULT_DIR"],
    "stats_ckpt_*_val_unseen.json",
))
for path in paths:
    with open(path, encoding="utf-8") as stream:
        json.load(stream)
print(len(paths))
PY
        '
}

gpu_is_idle() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used \
        --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    [ "$used" -le "$GPU_IDLE_LIMIT_MIB" ]
}

run_worker() {
    local result_count
    echo "worker_started_at=$(date --iso-8601=seconds) expected_first_results=$FIRST_EXPECTED_RESULTS"
    while true; do
        if [ -s "$LAUNCHED_FILE" ]; then
            echo "second_eval_already_launched_at=$(date --iso-8601=seconds)"
            return 0
        fi
        if ! result_count=$(valid_first_result_count); then
            echo "waiting_at=$(date --iso-8601=seconds) reason=first_results_invalid_or_unavailable"
            sleep "$POLL_SECONDS"
            continue
        fi
        if [ "$result_count" -lt "$FIRST_EXPECTED_RESULTS" ]; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=first_eval_incomplete results=$result_count/$FIRST_EXPECTED_RESULTS"
            sleep "$POLL_SECONDS"
            continue
        fi
        if ! gpu_is_idle; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=gpu_busy_after_first_eval results=$result_count/$FIRST_EXPECTED_RESULTS"
            sleep "$POLL_SECONDS"
            continue
        fi
        if "$SECOND_WATCH" start; then
            printf '%s\n' \
                "launched_at=$(date --iso-8601=seconds)" \
                "first_results=$result_count" \
                "second_watch=$SECOND_WATCH" \
                >"$LAUNCHED_FILE"
            echo "second_eval_launched_at=$(date --iso-8601=seconds) first_results=$result_count"
            return 0
        fi
        echo "waiting_at=$(date --iso-8601=seconds) reason=second_eval_launch_failed"
        sleep "$POLL_SECONDS"
    done
}

start_worker() {
    if is_running; then
        echo "watch=running pid=$(read_pid)"
        return 0
    fi
    mkdir -p "$MONITOR_ROOT"
    nohup setsid "$0" worker </dev/null >>"$LOG_FILE" 2>&1 &
    printf '%s\n' "$!" >"$PID_FILE"
    sleep 2
    is_running || {
        echo "Second SFT evaluation handoff failed during startup" >&2
        tail -n 100 "$LOG_FILE" >&2 || true
        exit 1
    }
    echo "watch=started pid=$(read_pid) log=$LOG_FILE"
}

show_status() {
    local result_count
    result_count=$(find "$FIRST_RESULT_DIR" -maxdepth 1 -type f \
        -name 'stats_ckpt_*_val_unseen.json' 2>/dev/null | wc -l)
    if is_running; then
        echo "watch=running pid=$(read_pid)"
    else
        echo "watch=stopped"
    fi
    echo "first_results=$result_count/$FIRST_EXPECTED_RESULTS"
    if [ -s "$LAUNCHED_FILE" ]; then
        echo "second_eval=launched"
        cat "$LAUNCHED_FILE"
    else
        echo "second_eval=waiting"
    fi
    [ -f "$LOG_FILE" ] && tail -n 20 "$LOG_FILE"
    return 0
}

case "$ACTION" in
    start) start_worker ;;
    worker) run_worker ;;
    status) show_status ;;
    tail) exec tail -f "$LOG_FILE" ;;
    stop)
        if is_running; then
            pid=$(read_pid)
            kill -TERM -- "-$pid"
            echo "Sent TERM to second SFT evaluation handoff process group $pid"
        else
            echo "watch=stopped"
        fi
        ;;
    *) echo "Usage: $0 {start|status|tail|stop}" >&2; exit 2 ;;
esac
