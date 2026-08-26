#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/checkpoint_order.sh"
ACTION=${1:-status}
CONFIG_FILE=${ETPR1_E24_EVAL_CONFIG_FILE:-${REPO_ROOT}/run_r2r/iter_train_rae_dino_e24_joint.yaml}
CONTAINER=${ETPR1_E24_EVAL_CONTAINER:-gwl-etpr1-rae}
PROTECTED_CONTAINER=${ETPR1_E24_EVAL_PROTECTED_CONTAINER:-gwl-etpnav}
CONFIG_CKPT_DIR=$(checkpoint_watch_dir_from_config "$CONFIG_FILE")
CKPT_DIR=${ETPR1_E24_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/active_lookahead/e24_joint_sft/checkpoints/etpr1_e24_joint_sft}
TRAIN_OUTPUT_ROOT=$(dirname -- "$(dirname -- "$CKPT_DIR")")
EVAL_ROOT=${ETPR1_E24_EVAL_OUTPUT_ROOT:-${REPO_ROOT}/data/logs/active_lookahead/e24_joint_eval}
EXP_NAME=${ETPR1_E24_EVAL_EXP_NAME:-etpr1_e24_joint_eval_watch}
RESULT_DIR=${EVAL_ROOT}/results/${EXP_NAME}/eval_results
PRETRAIN_PATH=${ETPR1_E24_EVAL_PRETRAIN_PATH:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
NUM_ENVIRONMENTS=${ETPR1_E24_EVAL_NUM_ENVIRONMENTS:-8}
POLL_SECONDS=${ETPR1_E24_EVAL_POLL_SECONDS:-30}
RETRY_SECONDS=${ETPR1_E24_EVAL_RETRY_SECONDS:-60}
GPU_IDLE_LIMIT_MIB=${ETPR1_E24_EVAL_GPU_IDLE_LIMIT_MIB:-1024}
BLOCKING_PROCESS_PATTERN=${ETPR1_E24_EVAL_BLOCKING_PROCESS_PATTERN:-}
CHECKPOINT_ORDER=${ETPR1_E24_EVAL_CHECKPOINT_ORDER:-$(checkpoint_order_from_config "$CONFIG_FILE")}
PID_FILE=${EVAL_ROOT}/watch.pid
LOG_FILE=${EVAL_ROOT}/watch.log
SPACE_CHECK_FILE=${EVAL_ROOT}/space_check.ok
EXPECTED_CHECKPOINTS=${ETPR1_E24_EVAL_EXPECTED_CHECKPOINTS:-50}
MIN_RESERVE_GIB=${ETPR1_E24_EVAL_MIN_RESERVE_GIB:-40}
EPISODE_COUNT=${ETPR1_E24_EVAL_EPISODE_COUNT:--1}
SELECTION_ENABLED=${ETPR1_E24_EVAL_SELECTION_ENABLED:-True}
RESULT_EPISODE_COUNT=${ETPR1_E24_EVAL_RESULT_EPISODE_COUNT:-}

if [ "$EPISODE_COUNT" != -1 ] \
    && ! [[ "$EPISODE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Episode count must be -1 or a positive integer: $EPISODE_COUNT" >&2
    exit 2
fi
case "$SELECTION_ENABLED" in
    True|False) ;;
    *)
        echo "Selection enabled must be True or False: $SELECTION_ENABLED" >&2
        exit 2
        ;;
esac
if [ -z "$RESULT_EPISODE_COUNT" ]; then
    if [ "$EPISODE_COUNT" = -1 ]; then
        RESULT_EPISODE_COUNT=1839
    else
        RESULT_EPISODE_COUNT=$EPISODE_COUNT
    fi
fi
if ! [[ "$RESULT_EPISODE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Result episode count must be a positive integer: $RESULT_EPISODE_COUNT" >&2
    exit 2
fi

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
    local checkpoint=$1 base iteration result diagnostic exit_code
    base=${checkpoint##*/}
    iteration=${base#ckpt.iter}
    iteration=${iteration%.pth}
    result=${RESULT_DIR}/stats_ckpt_${iteration}_val_unseen.json
    diagnostic=${RESULT_DIR}/lookahead_ckpt_${iteration}_val_unseen.json
    valid_result "$result" && valid_result "$diagnostic" && return 0

    if [ ! -s "$SPACE_CHECK_FILE" ]; then
        check_checkpoint_space || return 1
        printf 'checked_at=%s\n' "$(date --iso-8601=seconds)" >"$SPACE_CHECK_FILE"
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
    if [ "$exit_code" -ne 0 ]; then
        echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration exit_code=$exit_code"
        return 1
    fi
    if ! valid_result "$result"; then
        echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration reason=invalid_result result=$result"
        return 1
    fi
    if ! valid_result "$diagnostic"; then
        echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration reason=invalid_diagnostic diagnostic=$diagnostic"
        return 1
    fi
    echo "checkpoint_finished_at=$(date --iso-8601=seconds) iter=$iteration result=$result diagnostic=$diagnostic"
}

check_checkpoint_space() {
    local first size current_count remaining_bytes reserve_bytes free_bytes
    first=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' -print -quit)
    [ -n "$first" ] || return 0
    size=$(stat -c %s -- "$first")
    current_count=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' | wc -l)
    remaining_bytes=$((size * (EXPECTED_CHECKPOINTS - current_count)))
    reserve_bytes=$((MIN_RESERVE_GIB * 1024 * 1024 * 1024))
    free_bytes=$(df -B1 --output=avail "$CKPT_DIR" | tail -n 1 | tr -d ' ')
    if [ "$free_bytes" -lt $((remaining_bytes + reserve_bytes)) ]; then
        echo "space_check_failed_at=$(date --iso-8601=seconds) checkpoint_size=$size current_count=$current_count remaining_bytes=$remaining_bytes reserve_bytes=$reserve_bytes free_bytes=$free_bytes"
        return 1
    fi
    echo "space_check_ok checkpoint_size=$size current_count=$current_count remaining_bytes=$remaining_bytes reserve_bytes=$reserve_bytes free_bytes=$free_bytes"
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
        checkpoint_count=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' | wc -l)
        result_count=$(find "$RESULT_DIR" -maxdepth 1 -type f -name 'stats_ckpt_*_val_unseen.json' | wc -l)
        diagnostic_count=$(find "$RESULT_DIR" -maxdepth 1 -type f -name 'lookahead_ckpt_*_val_unseen.json' | wc -l)
        if [ "$checkpoint_count" -eq "$EXPECTED_CHECKPOINTS" ] \
            && [ "$result_count" -eq "$EXPECTED_CHECKPOINTS" ] \
            && [ "$diagnostic_count" -eq "$EXPECTED_CHECKPOINTS" ]; then
            if [ "$SELECTION_ENABLED" = True ]; then
                docker exec \
                    -e CKPT_DIR="$CKPT_DIR" \
                    -e RESULT_DIR="$RESULT_DIR" \
                    -e EVAL_ROOT="$EVAL_ROOT" \
                    -e EXPECTED_CHECKPOINTS="$EXPECTED_CHECKPOINTS" \
                    -e RESULT_EPISODE_COUNT="$RESULT_EPISODE_COUNT" \
                    "$CONTAINER" bash -lc '
                        source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh
                        conda activate etpr1_rae
                        cd /home/a6000/gwl/ETP-R1
                        python scripts/select_best_e24_joint_checkpoint.py \
                            --checkpoint-dir "$CKPT_DIR" \
                            --result-dir "$RESULT_DIR" \
                            --output-dir "$EVAL_ROOT/selection" \
                            --expected "$EXPECTED_CHECKPOINTS" \
                            --episodes "$RESULT_EPISODE_COUNT"
                    '
            fi
            echo "worker_completed_at=$(date --iso-8601=seconds) checkpoints=$checkpoint_count results=$result_count diagnostics=$diagnostic_count"
            return 0
        fi
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
    echo "diagnostics=$(find "$RESULT_DIR" -maxdepth 1 -type f -name 'lookahead_ckpt_*_val_unseen.json' 2>/dev/null | wc -l)"
    echo "checkpoint_order=$CHECKPOINT_ORDER config=$CONFIG_FILE"
    echo "episode_count=$EPISODE_COUNT result_episode_count=$RESULT_EPISODE_COUNT selection_enabled=$SELECTION_ENABLED"
    echo "blocking_process_pattern=${BLOCKING_PROCESS_PATTERN:-none}"
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
