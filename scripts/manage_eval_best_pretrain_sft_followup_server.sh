#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}
POLL_SECONDS=${ETPR1_SFT_FOLLOWUP_POLL_SECONDS:-60}
PYTHON_BIN=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}

CURRENT_EXP=${ETPR1_SFT_FOLLOWUP_CURRENT_EXP:-rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft}
CURRENT_OUTPUT_ROOT=${ETPR1_SFT_FOLLOWUP_CURRENT_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815}
CURRENT_ITERS=${ETPR1_SFT_FOLLOWUP_CURRENT_ITERS:-15000}
CURRENT_CKPT_DIR=${CURRENT_OUTPUT_ROOT}/checkpoints/${CURRENT_EXP}
CURRENT_FINAL_CKPT=${CURRENT_CKPT_DIR}/ckpt.iter${CURRENT_ITERS}.pth
CURRENT_FINAL_STATE=${CURRENT_CKPT_DIR}/train_states/train_state.iter${CURRENT_ITERS}.pth
CURRENT_PID_FILE=${CURRENT_OUTPUT_ROOT}/supervisor_server/trainer.pid
CURRENT_LATEST_LOG=${CURRENT_OUTPUT_ROOT}/supervisor_server/latest.log

REMOTE_HOST=${ETPR1_SFT_FOLLOWUP_REMOTE_HOST:-a6000@10.10.10.2}
REMOTE_PRETRAIN_ROOT=${ETPR1_SFT_FOLLOWUP_REMOTE_PRETRAIN_ROOT:-/home/a6000/gwl/ETP-R1/data/pretrain_resume_source_250000}
REMOTE_BEST_METRICS=${REMOTE_PRETRAIN_ROOT}/best/best_metrics.json
LOCAL_PRETRAIN_ROOT=${ETPR1_SFT_FOLLOWUP_LOCAL_PRETRAIN_ROOT:-${REPO_ROOT}/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best}

MONITOR_ROOT=${ETPR1_SFT_FOLLOWUP_MONITOR_ROOT:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/eval_best_sft_followup_monitor}
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

current_sft_running() {
    [ -s "$CURRENT_PID_FILE" ] || return 1
    local pid
    read -r pid <"$CURRENT_PID_FILE"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

current_sft_completed() {
    local current_log
    [ -s "$CURRENT_FINAL_CKPT" ] || return 1
    [ -s "$CURRENT_FINAL_STATE" ] || return 1
    current_log=$(readlink -f -- "$CURRENT_LATEST_LOG" 2>/dev/null) || return 1
    [ -s "$current_log" ] || return 1
    grep -q '^exit_code=0$' "$current_log"
}

remote_pretrain_running() {
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
        "ps -eo args | grep -F -- '$REMOTE_PRETRAIN_ROOT' | grep -v grep" \
        >/dev/null 2>&1
}

remote_pretrain_completed() {
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
        "test -s '$REMOTE_PRETRAIN_ROOT/ckpts/model_step_500000.pt' && \
         test -s '$REMOTE_PRETRAIN_ROOT/ckpts/train_state_500000.pt' && \
         grep -q '^exit_code=0$' '$REMOTE_PRETRAIN_ROOT/supervisor/latest.log'"
}

read_final_best() {
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" \
        "cat '$REMOTE_BEST_METRICS'" \
        | "$PYTHON_BIN" -c '
import json
import re
import sys

metrics = json.load(sys.stdin)
model = metrics["model_checkpoint"]
step = metrics["step"]
score = metrics["score"]
if not isinstance(step, int) or step <= 0:
    raise SystemExit("invalid best step")
if model != f"model_best_step_{step}.pt":
    raise SystemExit("best checkpoint and step do not match")
if re.fullmatch(r"model_best_step_[0-9]+[.]pt", model) is None:
    raise SystemExit("unsafe best checkpoint name")
print(f"{model}\t{step}\t{score}")
'
}

copy_final_best() {
    local best_model=$1 best_step=$2 best_score=$3
    local remote_checkpoint local_checkpoint part remote_size local_size
    local remote_sha local_sha
    remote_checkpoint=${REMOTE_PRETRAIN_ROOT}/best/${best_model}
    local_checkpoint=${LOCAL_PRETRAIN_ROOT}/${best_model}
    part=${local_checkpoint}.part.$$
    mkdir -p "$LOCAL_PRETRAIN_ROOT"

    remote_size=$(ssh -n -o BatchMode=yes "$REMOTE_HOST" \
        "stat -c %s '$remote_checkpoint'")
    remote_sha=$(ssh -n -o BatchMode=yes "$REMOTE_HOST" \
        "sha256sum '$remote_checkpoint'" | awk '{print $1}')

    if [ -s "$local_checkpoint" ]; then
        local_size=$(stat -c %s "$local_checkpoint")
        local_sha=$(sha256sum "$local_checkpoint" | awk '{print $1}')
    else
        rm -f -- "$part"
        scp -p "${REMOTE_HOST}:${remote_checkpoint}" "$part"
        local_size=$(stat -c %s "$part")
        local_sha=$(sha256sum "$part" | awk '{print $1}')
        if [ "$local_size" != "$remote_size" ] || [ "$local_sha" != "$remote_sha" ]; then
            rm -f -- "$part"
            echo "Copied checkpoint failed size or SHA-256 verification" >&2
            return 1
        fi
        mv -- "$part" "$local_checkpoint"
    fi

    if [ "$local_size" != "$remote_size" ] || [ "$local_sha" != "$remote_sha" ]; then
        echo "Existing local checkpoint does not match final remote best" >&2
        return 1
    fi
    printf '%s\n' \
        "selected_at=$(date --iso-8601=seconds)" \
        "remote_checkpoint=$remote_checkpoint" \
        "local_checkpoint=$local_checkpoint" \
        "best_step=$best_step" \
        "best_score=$best_score" \
        "bytes=$local_size" \
        "sha256=$local_sha" \
        >"${LOCAL_PRETRAIN_ROOT}/selection.env"
    printf '%s\n' "$local_checkpoint"
}

launch_followup_sft() {
    local best_model=$1 best_step=$2 best_score=$3 local_checkpoint=$4
    local launch_date followup_exp followup_output sync_destination
    launch_date=$(date +%Y%m%d)
    followup_exp=rae_dinov2_etpnav_cls_768_eval_best${best_step}_r2r_sft
    followup_output=data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_eval_best${best_step}_${launch_date}
    sync_destination=a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/${followup_output}/checkpoints/${followup_exp}

    if [ -e "${REPO_ROOT}/${followup_output}" ]; then
        echo "Refusing to overwrite existing follow-up output: ${REPO_ROOT}/${followup_output}" >&2
        return 1
    fi

    ETPR1_R2R_SFT_EXP_NAME="$followup_exp" \
    ETPR1_R2R_SFT_OUTPUT_ROOT="$followup_output" \
    ETPR1_R2R_SFT_ITERS=15000 \
    ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS=1 \
    ETPR1_R2R_SFT_PRETRAINED_PATH="$local_checkpoint" \
    ETPR1_R2R_SFT_CHECKPOINT_SYNC_ENABLED=True \
    ETPR1_R2R_SFT_CHECKPOINT_SYNC_DESTINATION="$sync_destination" \
        "${SCRIPT_DIR}/manage_rae_r2r_sft_server.sh" start

    printf '%s\n' \
        "launched_at=$(date --iso-8601=seconds)" \
        "best_model=$best_model" \
        "best_step=$best_step" \
        "best_score=$best_score" \
        "pretrained_path=$local_checkpoint" \
        "exp_name=$followup_exp" \
        "output_root=$followup_output" \
        "checkpoint_sync_destination=$sync_destination" \
        >"$LAUNCHED_FILE"
}

run_worker() {
    local selection best_model best_step best_score local_checkpoint
    echo "worker_started_at=$(date --iso-8601=seconds) current_exp=$CURRENT_EXP current_iters=$CURRENT_ITERS"
    while true; do
        if [ -s "$LAUNCHED_FILE" ]; then
            echo "followup_already_launched_at=$(date --iso-8601=seconds)"
            return 0
        fi
        if current_sft_running; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=current_sft_running"
            sleep "$POLL_SECONDS"
            continue
        fi
        if ! current_sft_completed; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=current_sft_not_successfully_complete"
            sleep "$POLL_SECONDS"
            continue
        fi
        if remote_pretrain_running; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=remote_pretrain_running"
            sleep "$POLL_SECONDS"
            continue
        fi
        if ! remote_pretrain_completed; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=remote_pretrain_not_successfully_complete"
            sleep "$POLL_SECONDS"
            continue
        fi

        if ! selection=$(read_final_best); then
            echo "waiting_at=$(date --iso-8601=seconds) reason=final_best_metrics_unavailable"
            sleep "$POLL_SECONDS"
            continue
        fi
        IFS=$'\t' read -r best_model best_step best_score <<<"$selection"
        echo "final_best_selected_at=$(date --iso-8601=seconds) model=$best_model step=$best_step score=$best_score"
        if ! local_checkpoint=$(copy_final_best "$best_model" "$best_step" "$best_score"); then
            echo "waiting_at=$(date --iso-8601=seconds) reason=final_best_copy_failed"
            sleep "$POLL_SECONDS"
            continue
        fi
        echo "final_best_copied_at=$(date --iso-8601=seconds) path=$local_checkpoint"
        if launch_followup_sft "$best_model" "$best_step" "$best_score" "$local_checkpoint"; then
            echo "followup_launched_at=$(date --iso-8601=seconds)"
            return 0
        fi
        echo "waiting_at=$(date --iso-8601=seconds) reason=followup_launch_failed"
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
        echo "Follow-up SFT watch failed during startup" >&2
        tail -n 100 "$LOG_FILE" >&2 || true
        exit 1
    }
    echo "watch=started pid=$(read_pid) log=$LOG_FILE"
}

show_status() {
    if is_running; then
        echo "watch=running pid=$(read_pid)"
    else
        echo "watch=stopped"
    fi
    if current_sft_running; then
        echo "current_sft=running"
    elif current_sft_completed; then
        echo "current_sft=complete"
    else
        echo "current_sft=incomplete_or_failed"
    fi
    if [ -s "$LAUNCHED_FILE" ]; then
        echo "followup=launched"
        cat "$LAUNCHED_FILE"
    else
        echo "followup=waiting"
    fi
    [ -f "$LOG_FILE" ] && tail -n 20 "$LOG_FILE"
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
            echo "Sent TERM to follow-up SFT watch process group $pid"
        else
            echo "watch=stopped"
        fi
        ;;
    *) echo "Usage: $0 {start|status|tail|stop}" >&2; exit 2 ;;
esac
