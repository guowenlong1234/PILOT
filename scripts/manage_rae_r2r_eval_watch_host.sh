#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/checkpoint_order.sh"
ACTION=${1:-status}
CONFIG_FILE=${ETPR1_R2R_EVAL_CONFIG_FILE:-${REPO_ROOT}/run_r2r/iter_train_rae_dino.yaml}
CONTAINER=${ETPR1_R2R_EVAL_CONTAINER:-gwl-etpr1-rae}
PROTECTED_CONTAINER=${ETPR1_R2R_EVAL_PROTECTED_CONTAINER:-gwl-etpnav}
CONFIG_CKPT_DIR=$(checkpoint_watch_dir_from_config "$CONFIG_FILE")
CKPT_DIR=${ETPR1_R2R_EVAL_CKPT_DIR:-${CONFIG_CKPT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_formal/checkpoints/rae_dinov2_etpnav_cls_768_r2r_sft}}
TRAIN_OUTPUT_ROOT=$(dirname -- "$(dirname -- "$CKPT_DIR")")
EVAL_ROOT=${ETPR1_R2R_EVAL_OUTPUT_ROOT:-${TRAIN_OUTPUT_ROOT}/eval_watch_val_unseen}
EXP_NAME=${ETPR1_R2R_EVAL_EXP_NAME:-$(basename -- "$CKPT_DIR")_eval_watch}
RESULT_DIR=${EVAL_ROOT}/results/${EXP_NAME}/eval_results
PRETRAIN_PATH=${ETPR1_R2R_EVAL_PRETRAIN_PATH:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_raw_cls_20260810/best/model_best_step_220000.pt}
NUM_ENVIRONMENTS=${ETPR1_R2R_EVAL_NUM_ENVIRONMENTS:-8}
EPISODE_COUNT=${ETPR1_R2R_EVAL_EPISODE_COUNT:--1}
POLL_SECONDS=${ETPR1_R2R_EVAL_POLL_SECONDS:-30}
RETRY_SECONDS=${ETPR1_R2R_EVAL_RETRY_SECONDS:-60}
GPU_IDLE_LIMIT_MIB=${ETPR1_R2R_EVAL_GPU_IDLE_LIMIT_MIB:-1024}
GPU_LOCK_FILE=${ETPR1_EVAL_GPU_LOCK_FILE:-/tmp/etpr1-eval-gpu.lock}
BLOCKING_PROCESS_PATTERN=${ETPR1_R2R_EVAL_BLOCKING_PROCESS_PATTERN:-}
READY_SHA_REQUIRED=${ETPR1_R2R_EVAL_READY_SHA_REQUIRED:-False}
CHECKPOINT_ORDER=${ETPR1_R2R_EVAL_CHECKPOINT_ORDER:-$(checkpoint_order_from_config "$CONFIG_FILE")}
PID_FILE=${EVAL_ROOT}/watch.pid
LOG_FILE=${EVAL_ROOT}/watch.log

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

ensure_container() {
    docker inspect "$CONTAINER" >/dev/null
    if [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != true ]; then
        docker start "$CONTAINER" >/dev/null
        echo "container_started_at=$(date --iso-8601=seconds) container=$CONTAINER"
    fi
}

protected_task_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$PROTECTED_CONTAINER" 2>/dev/null || true)" = true ] || return 1
    docker exec "$PROTECTED_CONTAINER" bash -lc \
        "ps -eo cmd | grep -E 'torchrun|run.py|train.py' | grep -v grep" \
        >/dev/null 2>&1
}

blocking_project_task_running() {
    [ -n "$BLOCKING_PROCESS_PATTERN" ] || return 1
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null || true)" = true ] || return 1
    docker exec -e BLOCKING_PROCESS_PATTERN="$BLOCKING_PROCESS_PATTERN" \
        "$CONTAINER" bash -lc \
        'ps -eo args | grep -F -- "$BLOCKING_PROCESS_PATTERN" | grep -v grep' \
        >/dev/null 2>&1
}

gpu_is_idle() {
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
        | head -n 1 | tr -d ' ')
    [ "$used" -le "$GPU_IDLE_LIMIT_MIB" ]
}

valid_result() {
    local result=$1
    [ -s "$result" ] || return 1
    docker exec -e RESULT_PATH="$result" "$CONTAINER" bash -lc \
        'source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && python -m json.tool "$RESULT_PATH" >/dev/null' \
        >/dev/null 2>&1
}

evaluate_checkpoint() {
    local checkpoint=$1 base iteration result exit_code
    base=${checkpoint##*/}
    iteration=${base#ckpt.iter}
    iteration=${iteration%.pth}
    result=${RESULT_DIR}/stats_ckpt_${iteration}_val_unseen.json
    valid_result "$result" && return 0
    if [ "$READY_SHA_REQUIRED" = True ]; then
        ready=${checkpoint}.sha256
        [ -s "$ready" ] || {
            echo "checkpoint_waiting_at=$(date --iso-8601=seconds) iter=$iteration reason=missing_sha_ready_marker"
            return 2
        }
        expected_sha=$(awk 'NR==1 {print $1}' "$ready")
        actual_sha=$(sha256sum -- "$checkpoint" | awk '{print $1}')
        [ "$actual_sha" = "$expected_sha" ] || {
            echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration reason=sha256_mismatch expected=$expected_sha actual=$actual_sha"
            return 1
        }
    fi

    if protected_task_running; then
        echo "checkpoint_waiting_at=$(date --iso-8601=seconds) iter=$iteration reason=protected_etpnav_task"
        return 2
    fi
    if blocking_project_task_running; then
        echo "checkpoint_waiting_at=$(date --iso-8601=seconds) iter=$iteration reason=blocking_project_task pattern=$BLOCKING_PROCESS_PATTERN"
        return 2
    fi
    if ! gpu_is_idle; then
        echo "checkpoint_waiting_at=$(date --iso-8601=seconds) iter=$iteration reason=gpu_busy"
        return 2
    fi

    exec {gpu_lock_fd}>"$GPU_LOCK_FILE"
    if ! flock -n "$gpu_lock_fd"; then
        echo "checkpoint_waiting_at=$(date --iso-8601=seconds) iter=$iteration reason=eval_gpu_lock_busy"
        exec {gpu_lock_fd}>&-
        return 2
    fi
    if protected_task_running || blocking_project_task_running || ! gpu_is_idle; then
        echo "checkpoint_waiting_at=$(date --iso-8601=seconds) iter=$iteration reason=resource_changed_after_lock"
        flock -u "$gpu_lock_fd"
        exec {gpu_lock_fd}>&-
        return 2
    fi

    echo "checkpoint_started_at=$(date --iso-8601=seconds) iter=$iteration path=$checkpoint"
    set +e
    docker exec \
        -e CKPT_PATH="$checkpoint" \
        -e EVAL_ROOT="$EVAL_ROOT" \
        -e EXP_NAME="$EXP_NAME" \
        -e PRETRAIN_PATH="$PRETRAIN_PATH" \
        -e NUM_ENVIRONMENTS="$NUM_ENVIRONMENTS" \
        -e CONFIG_FILE="$CONFIG_FILE" \
        -e EPISODE_COUNT="$EPISODE_COUNT" \
        "$CONTAINER" bash -lc '
            source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh
            conda activate etpr1_rae
            cd /home/a6000/gwl/ETP-R1
            export MPLCONFIGDIR=/tmp/matplotlib-etpr1-eval-watch
            export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
            scripts/etpr1_rae_runtime_exec.sh python run.py \
                --exp_name "$EXP_NAME" \
                --run-type eval \
                --exp-config "$CONFIG_FILE" \
                SIMULATOR_GPU_IDS "[0]" \
                TORCH_GPU_IDS "[0]" \
                TORCH_GPU_ID 0 \
                GPU_NUMBERS 1 \
                NUM_ENVIRONMENTS "$NUM_ENVIRONMENTS" \
                EVAL.CKPT_PATH_DIR "$CKPT_PATH" \
                EVAL.EPISODE_COUNT "$EPISODE_COUNT" \
                EVAL.SAVE_RESULTS True \
                TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
                CHECKPOINT_FOLDER "$EVAL_ROOT/checkpoints/" \
                TENSORBOARD_DIR "$EVAL_ROOT/tensorboard/" \
                RESULTS_DIR "$EVAL_ROOT/results/" \
                MODEL.pretrained_path "$PRETRAIN_PATH" \
                MODEL.RGB_ENCODER.precision ambient \
                2>&1 | python -u scripts/filter_habitat_startup_noise.py
            exit ${PIPESTATUS[0]}
        '
    exit_code=$?
    set -e
    flock -u "$gpu_lock_fd"
    exec {gpu_lock_fd}>&-
    if [ "$exit_code" -ne 0 ]; then
        echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration exit_code=$exit_code"
        return 1
    fi
    if ! valid_result "$result"; then
        echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration reason=invalid_result result=$result"
        return 1
    fi
    echo "checkpoint_finished_at=$(date --iso-8601=seconds) iter=$iteration result=$result"
}

run_worker() {
    ensure_container
    mkdir -p "$CKPT_DIR" "$RESULT_DIR"
    validate_checkpoint_order "$CHECKPOINT_ORDER"
    echo "worker_started_at=$(date --iso-8601=seconds) checkpoints=$CKPT_DIR results=$RESULT_DIR checkpoint_order=$CHECKPOINT_ORDER"
    while true; do
        found=false
        while IFS= read -r checkpoint; do
            found=true
            evaluate_checkpoint "$checkpoint" || {
                exit_code=$?
                if [ "$exit_code" -eq 2 ]; then
                    sleep "$RETRY_SECONDS"
                else
                    sleep "$RETRY_SECONDS"
                fi
            }
        done < <(list_ordered_checkpoints "$CKPT_DIR" "$CHECKPOINT_ORDER")
        if [ "$found" = false ]; then
            echo "queue_waiting_at=$(date --iso-8601=seconds) reason=no_checkpoints"
        fi
        sleep "$POLL_SECONDS"
    done
}

start_worker() {
    if is_running; then
        echo "watch=running pid=$(read_pid)"
        return 0
    fi
    ensure_container
    mkdir -p "$EVAL_ROOT" "$RESULT_DIR"
    nohup setsid "$0" worker </dev/null >>"$LOG_FILE" 2>&1 &
    printf '%s\n' "$!" >"$PID_FILE"
    sleep 2
    is_running || {
        echo "Evaluation watch failed during startup" >&2
        tail -n 100 "$LOG_FILE" >&2 || true
        exit 1
    }
    echo "watch=started pid=$(read_pid) log=$LOG_FILE"
}

show_status() {
    ensure_container
    if is_running; then
        echo "watch=running pid=$(read_pid)"
    else
        echo "watch=stopped"
    fi
    echo "checkpoints=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' 2>/dev/null | wc -l)"
    echo "results=$(find "$RESULT_DIR" -maxdepth 1 -type f -name 'stats_ckpt_*_val_unseen.json' 2>/dev/null | wc -l)"
    echo "checkpoint_order=$CHECKPOINT_ORDER config=$CONFIG_FILE"
    echo "blocking_process_pattern=${BLOCKING_PROCESS_PATTERN:-none}"
    echo "episode_count=$EPISODE_COUNT config=$CONFIG_FILE"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
    [ -f "$LOG_FILE" ] && tail -n 25 "$LOG_FILE"
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
            echo "Sent TERM to evaluation watch process group $pid"
        else
            echo "watch=stopped"
        fi
        ;;
    *) echo "Usage: $0 {start|status|tail|stop}" >&2; exit 2 ;;
esac
