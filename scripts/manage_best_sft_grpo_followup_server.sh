#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ACTION=${1:-status}

REMOTE_HOST=${ETPR1_BEST_SFT_GRPO_REMOTE_HOST:-a6000@10.10.10.2}
REMOTE_REPO=${ETPR1_BEST_SFT_GRPO_REMOTE_REPO:-/home/a6000/gwl/ETP-R1}
REMOTE_PYTHON=${ETPR1_BEST_SFT_GRPO_REMOTE_PYTHON:-/home/a6000/gwl/miniconda3/envs/etpr1_rae/bin/python}
LOCAL_PYTHON=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}
POLL_SECONDS=${ETPR1_BEST_SFT_GRPO_POLL_SECONDS:-60}
SFT_ITERS=${ETPR1_BEST_SFT_GRPO_SFT_ITERS:-15000}
SFT_LOG_EVERY=${ETPR1_BEST_SFT_GRPO_SFT_LOG_EVERY:-200}

FIRST_LOCAL_CKPT_DIR=${ETPR1_BEST_SFT_GRPO_FIRST_LOCAL_CKPT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815/checkpoints/rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft}
SECOND_LOCAL_CKPT_DIR=${ETPR1_BEST_SFT_GRPO_SECOND_LOCAL_CKPT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_eval_best465000_20260816/checkpoints/rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft}
# The first-round evaluation is complete and its remote checkpoint copies were
# intentionally reclaimed. Its 75 result files remain authoritative; the
# selected checkpoint is validated against the training-machine original.
FIRST_REMOTE_CKPT_DIR=${ETPR1_BEST_SFT_GRPO_FIRST_REMOTE_CKPT_DIR:--}
SECOND_REMOTE_CKPT_DIR=${ETPR1_BEST_SFT_GRPO_SECOND_REMOTE_CKPT_DIR:-${REMOTE_REPO}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_eval_best465000_20260816/checkpoints/rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft}
FIRST_REMOTE_RESULT_DIR=${ETPR1_BEST_SFT_GRPO_FIRST_REMOTE_RESULT_DIR:-${REMOTE_REPO}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815/eval_watch_val_unseen/results/rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft_eval_watch/eval_results}
SECOND_REMOTE_RESULT_DIR=${ETPR1_BEST_SFT_GRPO_SECOND_REMOTE_RESULT_DIR:-${REMOTE_REPO}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_eval_best465000_20260816/eval_watch_val_unseen/results/rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft_eval_watch/eval_results}

FIRST_LOCAL_PRETRAIN=${ETPR1_BEST_SFT_GRPO_FIRST_LOCAL_PRETRAIN:-${REPO_ROOT}/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_legacy_base_transfer/model_step_452500_nonvisual_transfer.pt}
SECOND_LOCAL_PRETRAIN=${ETPR1_BEST_SFT_GRPO_SECOND_LOCAL_PRETRAIN:-${REPO_ROOT}/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
FIRST_REMOTE_PRETRAIN=${ETPR1_BEST_SFT_GRPO_FIRST_REMOTE_PRETRAIN:-${REMOTE_REPO}/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_raw_cls_20260810/best/model_best_step_220000.pt}
SECOND_REMOTE_PRETRAIN=${ETPR1_BEST_SFT_GRPO_SECOND_REMOTE_PRETRAIN:-${REMOTE_REPO}/data/pretrain_resume_source_250000/best/model_best_step_465000.pt}

GRPO_EXP_NAME=${ETPR1_BEST_SFT_GRPO_EXP_NAME:-rae_dinov2_etpnav_cls_768_best_of_two_sft_r2r_grpo}
GRPO_OUTPUT_ROOT=${ETPR1_BEST_SFT_GRPO_OUTPUT_ROOT:-data/logs/rae_dinov2_etpnav_cls_768/r2r_grpo_best_of_two_sft_20260816}
GRPO_ITERS=${ETPR1_BEST_SFT_GRPO_ITERS:-1000}
GRPO_LOG_EVERY=${ETPR1_BEST_SFT_GRPO_LOG_EVERY:-10}
GRPO_KEEP_LAST_STATES=${ETPR1_BEST_SFT_GRPO_KEEP_LAST_STATES:-3}
GRPO_KEEP_STATE_EVERY=${ETPR1_BEST_SFT_GRPO_KEEP_STATE_EVERY:-250}
EVAL_MIN_FREE_GIB=${ETPR1_BEST_SFT_GRPO_EVAL_MIN_FREE_GIB:-8}
TRAIN_MIN_FREE_GIB=${ETPR1_BEST_SFT_GRPO_TRAIN_MIN_FREE_GIB:-30}

if [[ "$GRPO_OUTPUT_ROOT" = /* ]]; then
    GRPO_OUTPUT_PATH=$GRPO_OUTPUT_ROOT
    : "${ETPR1_BEST_SFT_GRPO_REMOTE_GRPO_CKPT_DIR:?Set ETPR1_BEST_SFT_GRPO_REMOTE_GRPO_CKPT_DIR with an absolute GRPO output root}"
    REMOTE_GRPO_CKPT_DIR=$ETPR1_BEST_SFT_GRPO_REMOTE_GRPO_CKPT_DIR
else
    GRPO_OUTPUT_PATH=${REPO_ROOT}/${GRPO_OUTPUT_ROOT}
    REMOTE_GRPO_CKPT_DIR=${ETPR1_BEST_SFT_GRPO_REMOTE_GRPO_CKPT_DIR:-${REMOTE_REPO}/${GRPO_OUTPUT_ROOT}/checkpoints/${GRPO_EXP_NAME}}
fi
LOCAL_GRPO_CKPT_DIR=${GRPO_OUTPUT_PATH}/checkpoints/${GRPO_EXP_NAME}
LOCAL_GRPO_STATE_DIR=${LOCAL_GRPO_CKPT_DIR}/train_states
GRPO_SUPERVISOR_DIR=${GRPO_OUTPUT_PATH}/supervisor_server
GRPO_PID_FILE=${GRPO_SUPERVISOR_DIR}/trainer.pid
GRPO_LATEST_LOG=${GRPO_SUPERVISOR_DIR}/latest.log
REMOTE_EVAL_ROOT=${ETPR1_BEST_SFT_GRPO_REMOTE_EVAL_ROOT:-${REMOTE_REPO}/${GRPO_OUTPUT_ROOT}/eval_watch_val_unseen}
REMOTE_EVAL_EXP=${ETPR1_BEST_SFT_GRPO_REMOTE_EVAL_EXP:-${GRPO_EXP_NAME}_eval_watch}
REMOTE_EVAL_RESULT_DIR=${REMOTE_EVAL_ROOT}/results/${REMOTE_EVAL_EXP}/eval_results

MONITOR_ROOT=${ETPR1_BEST_SFT_GRPO_MONITOR_ROOT:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/best_sft_grpo_followup_monitor}
PID_FILE=${MONITOR_ROOT}/watch.pid
LOG_FILE=${MONITOR_ROOT}/watch.log
SELECTION_FILE=${MONITOR_ROOT}/selection.json
LAUNCHED_FILE=${MONITOR_ROOT}/launched.env
COMPLETED_FILE=${MONITOR_ROOT}/completed.env
GRPO_MANAGER=${SCRIPT_DIR}/manage_rae_r2r_grpo_server.sh
REMOTE_EVAL_MANAGER=${REMOTE_REPO}/scripts/manage_rae_r2r_eval_watch_host.sh

for value in "$POLL_SECONDS" "$SFT_ITERS" "$SFT_LOG_EVERY" "$GRPO_ITERS" "$GRPO_LOG_EVERY" "$GRPO_KEEP_LAST_STATES" "$EVAL_MIN_FREE_GIB" "$TRAIN_MIN_FREE_GIB"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Expected a positive integer: $value" >&2; exit 2; }
done
[[ "$GRPO_KEEP_STATE_EVERY" =~ ^[0-9]+$ ]] || { echo "Expected a non-negative keep-state interval" >&2; exit 2; }

EXPECTED_GRPO_CHECKPOINTS=$(( (GRPO_ITERS + GRPO_LOG_EVERY - 1) / GRPO_LOG_EVERY ))

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

grpo_running() {
    [ -s "$GRPO_PID_FILE" ] || return 1
    local pid
    read -r pid <"$GRPO_PID_FILE"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

remote_select() {
    local first second command
    first="round1=${FIRST_REMOTE_RESULT_DIR}=${FIRST_REMOTE_CKPT_DIR}"
    second="round2=${SECOND_REMOTE_RESULT_DIR}=${SECOND_REMOTE_CKPT_DIR}"
    printf -v command "cd %q && %q scripts/select_best_r2r_sft_checkpoint.py --total-iterations %q --checkpoint-interval %q --candidate %q --candidate %q" \
        "$REMOTE_REPO" "$REMOTE_PYTHON" "$SFT_ITERS" "$SFT_LOG_EVERY" "$first" "$second"
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "$command"
}

save_selection() {
    local selection=$1 temporary
    mkdir -p "$MONITOR_ROOT"
    temporary=${SELECTION_FILE}.tmp.$$
    printf '%s\n' "$selection" >"$temporary"
    mv -- "$temporary" "$SELECTION_FILE"
}

read_selection() {
    "$LOCAL_PYTHON" - "$SELECTION_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    item = json.load(stream)
print(
    item["run"], item["iteration"], item["success"], item["spl"],
    item["score"], item["checkpoint_bytes"], sep="\t"
)
PY
}

resolve_selected_resources() {
    local selection_values
    selection_values=$(read_selection)
    IFS=$'\t' read -r SELECTED_RUN SELECTED_ITER SELECTED_SR SELECTED_SPL SELECTED_SCORE SELECTED_BYTES <<<"$selection_values"
    case "$SELECTED_RUN" in
        round1)
            SELECTED_LOCAL_CKPT=${FIRST_LOCAL_CKPT_DIR}/ckpt.iter${SELECTED_ITER}.pth
            SELECTED_LOCAL_PRETRAIN=$FIRST_LOCAL_PRETRAIN
            SELECTED_REMOTE_PRETRAIN=$FIRST_REMOTE_PRETRAIN
            ;;
        round2)
            SELECTED_LOCAL_CKPT=${SECOND_LOCAL_CKPT_DIR}/ckpt.iter${SELECTED_ITER}.pth
            SELECTED_LOCAL_PRETRAIN=$SECOND_LOCAL_PRETRAIN
            SELECTED_REMOTE_PRETRAIN=$SECOND_REMOTE_PRETRAIN
            ;;
        *) echo "Unknown selected SFT run: $SELECTED_RUN" >&2; return 1 ;;
    esac
    [ -s "$SELECTED_LOCAL_CKPT" ] || { echo "Selected local checkpoint is missing: $SELECTED_LOCAL_CKPT" >&2; return 1; }
    [ -s "$SELECTED_LOCAL_PRETRAIN" ] || { echo "Selected local pretrain is missing: $SELECTED_LOCAL_PRETRAIN" >&2; return 1; }
    # Remote SFT copies may be intentionally removed after their evaluations.
    # The training-machine original is authoritative for GRPO and disk sizing.
    SELECTED_BYTES=$(stat -c %s -- "$SELECTED_LOCAL_CKPT")
}

available_bytes_local() {
    df -PB1 "$REPO_ROOT" | awk 'NR == 2 {print $4}'
}

available_bytes_remote() {
    local command
    printf -v command "df --output=avail -B1 %q | tail -n 1 | tr -d ' '" "$REMOTE_REPO"
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "$command"
}

storage_ready() {
    local local_available remote_available projected eval_required train_required
    local_available=$(available_bytes_local) || return 1
    remote_available=$(available_bytes_remote) || return 1
    projected=$(( SELECTED_BYTES * EXPECTED_GRPO_CHECKPOINTS ))
    eval_required=$(( projected + EVAL_MIN_FREE_GIB * 1024 * 1024 * 1024 ))
    train_required=$(( projected + TRAIN_MIN_FREE_GIB * 1024 * 1024 * 1024 ))
    if [ "$remote_available" -lt "$eval_required" ]; then
        echo "waiting_at=$(date --iso-8601=seconds) reason=eval_disk_space remote_available=$remote_available required=$eval_required projected_checkpoints=$projected"
        return 1
    fi
    if [ "$local_available" -lt "$train_required" ]; then
        echo "waiting_at=$(date --iso-8601=seconds) reason=train_disk_space local_available=$local_available required=$train_required projected_checkpoints=$projected"
        return 1
    fi
}

grpo_env() {
    printf '%s\0' \
        "ETPR1_R2R_GRPO_EXP_NAME=$GRPO_EXP_NAME" \
        "ETPR1_R2R_GRPO_OUTPUT_ROOT=$GRPO_OUTPUT_ROOT" \
        "ETPR1_R2R_GRPO_ITERS=$GRPO_ITERS" \
        "ETPR1_R2R_GRPO_LOG_EVERY=$GRPO_LOG_EVERY" \
        "ETPR1_R2R_GRPO_KEEP_LAST_STATES=$GRPO_KEEP_LAST_STATES" \
        "ETPR1_R2R_GRPO_KEEP_STATE_EVERY=$GRPO_KEEP_STATE_EVERY" \
        "ETPR1_R2R_GRPO_SFT_CHECKPOINT=$SELECTED_LOCAL_CKPT" \
        "ETPR1_R2R_GRPO_PRETRAINED_PATH=$SELECTED_LOCAL_PRETRAIN" \
        "ETPR1_R2R_GRPO_CHECKPOINT_SYNC_ENABLED=True" \
        "ETPR1_R2R_GRPO_CHECKPOINT_SYNC_DESTINATION=${REMOTE_HOST}:${REMOTE_GRPO_CKPT_DIR}"
}

start_remote_eval() {
    local command
    printf -v command "cd %q && env ETPR1_R2R_EVAL_CKPT_DIR=%q ETPR1_R2R_EVAL_OUTPUT_ROOT=%q ETPR1_R2R_EVAL_EXP_NAME=%q ETPR1_R2R_EVAL_PRETRAIN_PATH=%q ETPR1_R2R_EVAL_CHECKPOINT_ORDER=ascending %q start" \
        "$REMOTE_REPO" "$REMOTE_GRPO_CKPT_DIR" "$REMOTE_EVAL_ROOT" "$REMOTE_EVAL_EXP" "$SELECTED_REMOTE_PRETRAIN" "$REMOTE_EVAL_MANAGER"
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "$command"
}

stop_remote_eval() {
    local command
    printf -v command "cd %q && env ETPR1_R2R_EVAL_CKPT_DIR=%q ETPR1_R2R_EVAL_OUTPUT_ROOT=%q ETPR1_R2R_EVAL_EXP_NAME=%q ETPR1_R2R_EVAL_PRETRAIN_PATH=%q %q stop" \
        "$REMOTE_REPO" "$REMOTE_GRPO_CKPT_DIR" "$REMOTE_EVAL_ROOT" "$REMOTE_EVAL_EXP" "$SELECTED_REMOTE_PRETRAIN" "$REMOTE_EVAL_MANAGER"
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "$command"
}

mark_launched() {
    mkdir -p "$MONITOR_ROOT"
    printf '%s\n' \
        "launched_at=$(date --iso-8601=seconds)" \
        "selected_run=$SELECTED_RUN" \
        "selected_iteration=$SELECTED_ITER" \
        "selected_sr=$SELECTED_SR" \
        "selected_spl=$SELECTED_SPL" \
        "selected_score=$SELECTED_SCORE" \
        "selected_checkpoint=$SELECTED_LOCAL_CKPT" \
        "selected_pretrain=$SELECTED_LOCAL_PRETRAIN" \
        "grpo_exp_name=$GRPO_EXP_NAME" \
        "grpo_output_root=$GRPO_OUTPUT_ROOT" \
        "grpo_iters=$GRPO_ITERS" \
        "grpo_log_every=$GRPO_LOG_EVERY" \
        "expected_grpo_checkpoints=$EXPECTED_GRPO_CHECKPOINTS" \
        "remote_checkpoint_dir=$REMOTE_GRPO_CKPT_DIR" \
        "remote_eval_result_dir=$REMOTE_EVAL_RESULT_DIR" \
        >"$LAUNCHED_FILE"
}

launch_grpo_once() {
    local -a environment
    if [ -s "$LAUNCHED_FILE" ]; then
        return 0
    fi
    if grpo_running || [ -e "$GRPO_LATEST_LOG" ] || compgen -G "${LOCAL_GRPO_CKPT_DIR}/ckpt.iter*.pth" >/dev/null; then
        echo "grpo_existing_run_detected_at=$(date --iso-8601=seconds)"
        mark_launched
        return 0
    fi
    start_remote_eval || return 1
    mapfile -d '' -t environment < <(grpo_env)
    if env "${environment[@]}" "$GRPO_MANAGER" start; then
        mark_launched
        echo "grpo_launched_at=$(date --iso-8601=seconds) selected_run=$SELECTED_RUN selected_iter=$SELECTED_ITER score=$SELECTED_SCORE"
        return 0
    fi
    return 1
}

grpo_complete() {
    local latest
    [ -s "${LOCAL_GRPO_CKPT_DIR}/ckpt.iter${GRPO_ITERS}.pth" ] || return 1
    [ -s "${LOCAL_GRPO_STATE_DIR}/train_state.iter${GRPO_ITERS}.pth" ] || return 1
    latest=$(readlink -f -- "$GRPO_LATEST_LOG" 2>/dev/null) || return 1
    [ -s "$latest" ] && grep -q '^exit_code=0$' "$latest"
}

remote_grpo_results_complete() {
    local candidate command
    candidate="grpo=${REMOTE_EVAL_RESULT_DIR}=${REMOTE_GRPO_CKPT_DIR}"
    printf -v command "cd %q && %q scripts/select_best_r2r_sft_checkpoint.py --total-iterations %q --checkpoint-interval %q --candidate %q" \
        "$REMOTE_REPO" "$REMOTE_PYTHON" "$GRPO_ITERS" "$GRPO_LOG_EVERY" "$candidate"
    ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "$command" >/dev/null
}

run_worker() {
    local selection
    echo "worker_started_at=$(date --iso-8601=seconds) selection_metric=success+spl grpo_iters=$GRPO_ITERS grpo_log_every=$GRPO_LOG_EVERY"
    while true; do
        if [ -s "$COMPLETED_FILE" ]; then
            echo "workflow_already_complete_at=$(date --iso-8601=seconds)"
            return 0
        fi
        if [ ! -s "$SELECTION_FILE" ]; then
            if ! selection=$(remote_select 2>/dev/null); then
                echo "waiting_at=$(date --iso-8601=seconds) reason=sft_evaluations_incomplete_or_invalid expected_per_run=$(( (SFT_ITERS + SFT_LOG_EVERY - 1) / SFT_LOG_EVERY ))"
                sleep "$POLL_SECONDS"
                continue
            fi
            save_selection "$selection"
            echo "sft_best_selected_at=$(date --iso-8601=seconds) selection=$selection"
        fi
        if ! resolve_selected_resources; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=selected_training_resource_unavailable"
            sleep "$POLL_SECONDS"
            continue
        fi
        if [ ! -s "$LAUNCHED_FILE" ]; then
            if ! storage_ready; then
                sleep "$POLL_SECONDS"
                continue
            fi
            if ! launch_grpo_once; then
                echo "waiting_at=$(date --iso-8601=seconds) reason=grpo_or_eval_launch_failed"
                sleep "$POLL_SECONDS"
                continue
            fi
        fi
        if grpo_running; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=grpo_running"
            sleep "$POLL_SECONDS"
            continue
        fi
        if ! grpo_complete; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=grpo_not_successfully_complete action=resume_with_standard_grpo_manager"
            sleep "$POLL_SECONDS"
            continue
        fi
        if ! remote_grpo_results_complete; then
            echo "waiting_at=$(date --iso-8601=seconds) reason=grpo_evaluation_or_sync_incomplete expected=$EXPECTED_GRPO_CHECKPOINTS"
            sleep "$POLL_SECONDS"
            continue
        fi
        stop_remote_eval || true
        printf '%s\n' \
            "completed_at=$(date --iso-8601=seconds)" \
            "selected_run=$SELECTED_RUN" \
            "selected_iteration=$SELECTED_ITER" \
            "selected_score=$SELECTED_SCORE" \
            "grpo_final_iteration=$GRPO_ITERS" \
            "evaluated_checkpoints=$EXPECTED_GRPO_CHECKPOINTS" \
            >"$COMPLETED_FILE"
        echo "workflow_completed_at=$(date --iso-8601=seconds) evaluated=$EXPECTED_GRPO_CHECKPOINTS"
        return 0
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
        if [ -s "$COMPLETED_FILE" ]; then
            echo "watch=finished workflow=complete"
            return 0
        fi
        echo "Best-SFT GRPO follow-up failed during startup" >&2
        tail -n 100 "$LOG_FILE" >&2 || true
        exit 1
    }
    echo "watch=started pid=$(read_pid) log=$LOG_FILE"
}

resume_grpo() {
    local -a environment
    [ -s "$SELECTION_FILE" ] || { echo "No persisted SFT selection is available yet." >&2; exit 1; }
    [ -s "$LAUNCHED_FILE" ] || { echo "GRPO has not been launched by this workflow." >&2; exit 1; }
    resolve_selected_resources
    if grpo_running; then
        echo "GRPO is already running." >&2
        exit 1
    fi
    storage_ready || { echo "Insufficient space to resume GRPO." >&2; exit 1; }
    start_remote_eval
    mapfile -d '' -t environment < <(grpo_env)
    env "${environment[@]}" "$GRPO_MANAGER" resume
    echo "GRPO resume requested with the persisted selection and current workflow settings."
}

show_status() {
    if is_running; then echo "watch=running pid=$(read_pid)"; else echo "watch=stopped"; fi
    echo "selection_metric=success+spl"
    echo "grpo_checkpoints=$(find "$LOCAL_GRPO_CKPT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' 2>/dev/null | wc -l)/$EXPECTED_GRPO_CHECKPOINTS"
    if grpo_running; then echo "grpo=running"; elif grpo_complete; then echo "grpo=complete"; else echo "grpo=waiting_or_stopped"; fi
    [ -s "$SELECTION_FILE" ] && { echo "selection:"; "$LOCAL_PYTHON" -m json.tool "$SELECTION_FILE"; }
    [ -s "$LAUNCHED_FILE" ] && { echo "launch:"; cat "$LAUNCHED_FILE"; }
    [ -s "$COMPLETED_FILE" ] && { echo "workflow=complete"; cat "$COMPLETED_FILE"; }
    [ -f "$LOG_FILE" ] && tail -n 25 "$LOG_FILE"
    return 0
}

case "$ACTION" in
    start) start_worker ;;
    resume) resume_grpo ;;
    worker) run_worker ;;
    status) show_status ;;
    tail) exec tail -f "$LOG_FILE" ;;
    stop)
        if is_running; then
            pid=$(read_pid)
            kill -TERM -- "-$pid"
            echo "Sent TERM to follow-up monitor process group $pid; GRPO and evaluation jobs were not stopped."
        else
            echo "watch=stopped"
        fi
        ;;
    *) echo "Usage: $0 {start|resume|status|tail|stop}" >&2; exit 2 ;;
esac
