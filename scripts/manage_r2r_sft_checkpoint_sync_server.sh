#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
SOURCE_DIR=${ETPR1_R2R_SFT_SYNC_SOURCE_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_formal/checkpoints/rae_dinov2_etpnav_cls_768_r2r_sft}
DEST_HOST=${ETPR1_R2R_SFT_SYNC_DEST_HOST:-a6000@10.10.10.2}
DEST_DIR=${ETPR1_R2R_SFT_SYNC_DEST_DIR:-/home/a6000/gwl/ETP-R1/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_formal/checkpoints/rae_dinov2_etpnav_cls_768_r2r_sft}
POLL_SECONDS=${ETPR1_R2R_SFT_SYNC_POLL_SECONDS:-30}
STATE_DIR=${ETPR1_R2R_SFT_SYNC_STATE_DIR:-${SOURCE_DIR}/sync_to_eval}
PID_FILE=${STATE_DIR}/sync.pid
LOG_FILE=${STATE_DIR}/sync.log

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

validate_settings() {
    [[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
        echo "POLL_SECONDS must be a positive integer" >&2
        exit 2
    }
    [ -d "$SOURCE_DIR" ] || {
        echo "Source checkpoint directory does not exist: $SOURCE_DIR" >&2
        exit 1
    }
    command -v rsync >/dev/null
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$DEST_HOST" \
        "mkdir -p '$DEST_DIR/.incoming'"
}

sync_checkpoint() {
    local checkpoint=$1 base iteration training_state source_size remote_size
    local incoming remote_checkpoint
    base=${checkpoint##*/}
    iteration=${base#ckpt.iter}
    iteration=${iteration%.pth}
    training_state=${SOURCE_DIR}/train_states/train_state.iter${iteration}.pth
    [ -s "$training_state" ] || return 0

    remote_checkpoint=${DEST_DIR}/${base}
    if remote_size=$(ssh -o BatchMode=yes "$DEST_HOST" \
        "test -s '$remote_checkpoint' && stat -c %s '$remote_checkpoint'" \
        2>/dev/null); then
        source_size=$(stat -c %s "$checkpoint")
        if [ "$remote_size" = "$source_size" ]; then
            return 0
        fi
        echo "remote_size_mismatch_at=$(date --iso-8601=seconds) iter=$iteration local=$source_size remote=$remote_size"
        return 1
    fi

    incoming=${DEST_DIR}/.incoming/${base}.part.$$
    echo "sync_started_at=$(date --iso-8601=seconds) iter=$iteration source=$checkpoint"
    rsync -a --partial --protect-args "$checkpoint" \
        "${DEST_HOST}:${incoming}"
    source_size=$(stat -c %s "$checkpoint")
    remote_size=$(ssh -o BatchMode=yes "$DEST_HOST" "stat -c %s '$incoming'")
    if [ "$remote_size" != "$source_size" ]; then
        echo "sync_failed_at=$(date --iso-8601=seconds) iter=$iteration reason=size_mismatch local=$source_size remote=$remote_size"
        return 1
    fi
    ssh -o BatchMode=yes "$DEST_HOST" \
        "chmod 0644 '$incoming' && mv '$incoming' '$remote_checkpoint'"
    echo "sync_finished_at=$(date --iso-8601=seconds) iter=$iteration bytes=$source_size destination=$remote_checkpoint"
}

run_worker() {
    validate_settings
    mkdir -p "$STATE_DIR"
    echo "worker_started_at=$(date --iso-8601=seconds) source=$SOURCE_DIR destination=${DEST_HOST}:${DEST_DIR} poll_seconds=$POLL_SECONDS"
    while true; do
        while IFS= read -r checkpoint; do
            sync_checkpoint "$checkpoint" || true
        done < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' | sort -V)
        sleep "$POLL_SECONDS"
    done
}

start_worker() {
    if is_running; then
        echo "sync=running pid=$(read_pid)"
        return 0
    fi
    validate_settings
    mkdir -p "$STATE_DIR"
    nohup setsid "$0" worker </dev/null >>"$LOG_FILE" 2>&1 &
    printf '%s\n' "$!" >"$PID_FILE"
    sleep 2
    is_running || {
        echo "Checkpoint sync failed during startup" >&2
        tail -n 80 "$LOG_FILE" >&2 || true
        exit 1
    }
    echo "sync=started pid=$(read_pid) log=$LOG_FILE"
}

show_status() {
    if is_running; then
        echo "sync=running pid=$(read_pid)"
    else
        echo "sync=stopped"
    fi
    echo "complete_local=$(find "$SOURCE_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' 2>/dev/null | wc -l)"
    if [ -f "$LOG_FILE" ]; then
        tail -n 20 "$LOG_FILE"
    fi
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
            echo "Sent TERM to checkpoint sync process group $pid"
        else
            echo "sync=stopped"
        fi
        ;;
    *) echo "Usage: $0 {start|status|tail|stop}" >&2; exit 2 ;;
esac
