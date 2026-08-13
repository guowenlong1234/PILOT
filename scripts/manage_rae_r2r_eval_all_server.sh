#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/checkpoint_order.sh"

CONFIG_FILE=${ETPR1_R2R_EVAL_CONFIG_FILE:-${REPO_ROOT}/run_r2r/iter_train_rae_dino.yaml}
EXP_NAME=${ETPR1_R2R_EVAL_EXP_NAME:-rae_dinov2_etpnav_cls_768_r2r_sft_all_ckpts_val_unseen}
TRAIN_ROOT=${ETPR1_R2R_EVAL_TRAIN_ROOT:-data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_formal}
EVAL_ROOT=${ETPR1_R2R_EVAL_OUTPUT_ROOT:-${TRAIN_ROOT}/eval_all_checkpoints_val_unseen}
CONFIG_CKPT_DIR=$(checkpoint_watch_dir_from_config "$CONFIG_FILE")
CKPT_DIR=${ETPR1_R2R_EVAL_CKPT_DIR:-${CONFIG_CKPT_DIR:-${TRAIN_ROOT}/checkpoints/rae_dinov2_etpnav_cls_768_r2r_sft}}
RESULT_DIR=${EVAL_ROOT}/results/${EXP_NAME}/eval_results
PID_DIR=${EVAL_ROOT}/pids
PYTHON_BIN=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}
RUNTIME_ROOT=${ETPR1_SERVER_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/server_sft}
PRETRAIN_PATH=${ETPR1_R2R_EVAL_PRETRAIN_PATH:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768/best/model_best_step_452500.pt}
NUM_WORKERS=${ETPR1_R2R_EVAL_WORKERS:-2}
NUM_ENVIRONMENTS=${ETPR1_R2R_EVAL_NUM_ENVIRONMENTS:-8}
CHECKPOINT_ORDER=${ETPR1_R2R_EVAL_CHECKPOINT_ORDER:-$(checkpoint_order_from_config "$CONFIG_FILE")}

usage() {
    echo "Usage: $0 <start|status|worker> [worker_index]" >&2
}

worker_pid_file() {
    printf '%s/worker_gpu%s.pid\n' "$PID_DIR" "$1"
}

worker_is_running() {
    local pid_file pid stat
    pid_file=$(worker_pid_file "$1")
    [ -s "$pid_file" ] || return 1
    read -r pid < "$pid_file"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    stat=$(ps -o stat= -p "$pid" 2>/dev/null) || return 1
    [[ "$stat" != Z* ]]
}

prepare_runtime() {
    local path
    for path in \
        "$PYTHON_BIN" \
        "$RUNTIME_ROOT/habitat-lab/habitat/__init__.py" \
        "$RUNTIME_ROOT/habitat-baselines/habitat_baselines/common/baseline_registry.py" \
        "$RUNTIME_ROOT/python/dtw/__init__.py" \
        "$PRETRAIN_PATH"; do
        if [ ! -e "$path" ]; then
            echo "Missing evaluation dependency: $path" >&2
            return 1
        fi
    done
    if [ ! -d "$CKPT_DIR" ]; then
        echo "Checkpoint directory does not exist: $CKPT_DIR" >&2
        return 1
    fi
    mkdir -p "$EVAL_ROOT" "$RESULT_DIR" "$PID_DIR"
}

run_worker() {
    local worker_index=${1:?worker index is required}
    local log_file completed=0 skipped=0 failed=0 record=0
    local ckpt base iteration result exit_code

    if ! [[ "$worker_index" =~ ^[0-9]+$ ]] || [ "$worker_index" -ge "$NUM_WORKERS" ]; then
        echo "Invalid worker index: $worker_index" >&2
        exit 2
    fi
    prepare_runtime
    validate_checkpoint_order "$CHECKPOINT_ORDER"
    cd "$REPO_ROOT"

    log_file=${EVAL_ROOT}/eval_gpu${worker_index}.log
    exec >>"$log_file" 2>&1

    export MPLCONFIGDIR=/tmp/matplotlib-etpr1-server-eval-gpu${worker_index}
    export GLOG_minloglevel=${GLOG_minloglevel:-2}
    export MAGNUM_LOG=${MAGNUM_LOG:-quiet}
    export HABITAT_SIM_LOG=${HABITAT_SIM_LOG:-quiet}
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
    export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$RUNTIME_ROOT/habitat-baselines/habitat_baselines"
    export PYTHONPATH="${REPO_ROOT}/scripts/benchmark_shims:${REPO_ROOT}:${REPO_ROOT}/vendor/legacy_clip:${RUNTIME_ROOT}/habitat-lab:${RUNTIME_ROOT}/python"
    export CUDA_VISIBLE_DEVICES=$worker_index
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

    echo "worker_started_at=$(date --iso-8601=seconds) physical_gpu=$worker_index checkpoint_order=$CHECKPOINT_ORDER"
    echo "source_commit=$(git rev-parse HEAD)"
    "$PYTHON_BIN" -c \
        'import sys, torch, transformers, habitat, habitat_sim; print(f"versions=python:{sys.version.split()[0]} torch:{torch.__version__} transformers:{transformers.__version__} cuda:{torch.version.cuda} habitat:{habitat.__version__} habitat_sim:{habitat_sim.__version__}")'

    while IFS= read -r ckpt; do
        if [ $((record % NUM_WORKERS)) -ne "$worker_index" ]; then
            record=$((record + 1))
            continue
        fi
        record=$((record + 1))
        base=${ckpt##*/}
        iteration=${base#ckpt.iter}
        iteration=${iteration%.pth}
        result=${RESULT_DIR}/stats_ckpt_${iteration}_val_unseen.json

        if [ -s "$result" ]; then
            echo "checkpoint_skipped_at=$(date --iso-8601=seconds) iter=$iteration result=$result"
            skipped=$((skipped + 1))
            continue
        fi

        echo "checkpoint_started_at=$(date --iso-8601=seconds) iter=$iteration path=$ckpt"
        "$PYTHON_BIN" run.py \
            --exp_name "$EXP_NAME" \
            --run-type eval \
            --exp-config run_r2r/iter_train_rae_dino.yaml \
            SIMULATOR_GPU_IDS "[0]" \
            TORCH_GPU_IDS "[0]" \
            TORCH_GPU_ID 0 \
            GPU_NUMBERS 1 \
            NUM_ENVIRONMENTS "$NUM_ENVIRONMENTS" \
            EVAL.CKPT_PATH_DIR "$ckpt" \
            EVAL.EPISODE_COUNT -1 \
            EVAL.SAVE_RESULTS True \
            TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
            CHECKPOINT_FOLDER "$EVAL_ROOT/checkpoints/" \
            TENSORBOARD_DIR "$EVAL_ROOT/tensorboard/" \
            RESULTS_DIR "$EVAL_ROOT/results/" \
            MODEL.pretrained_path "$PRETRAIN_PATH" \
            MODEL.RGB_ENCODER.precision ambient \
            2>&1 | "$PYTHON_BIN" -u scripts/filter_habitat_startup_noise.py
        exit_code=${PIPESTATUS[0]}

        if [ "$exit_code" -ne 0 ]; then
            failed=$((failed + 1))
            echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration exit_code=$exit_code"
            echo "worker_stopped completed=$completed skipped=$skipped failed=$failed"
            exit "$exit_code"
        fi
        if [ ! -s "$result" ] || ! "$PYTHON_BIN" -m json.tool "$result" >/dev/null; then
            failed=$((failed + 1))
            echo "checkpoint_failed_at=$(date --iso-8601=seconds) iter=$iteration reason=invalid_result result=$result"
            echo "worker_stopped completed=$completed skipped=$skipped failed=$failed"
            exit 3
        fi

        completed=$((completed + 1))
        echo "checkpoint_finished_at=$(date --iso-8601=seconds) iter=$iteration result=$result completed=$completed"
    done < <(list_ordered_checkpoints "$CKPT_DIR" "$CHECKPOINT_ORDER")

    echo "worker_finished_at=$(date --iso-8601=seconds) completed=$completed skipped=$skipped failed=$failed"
}

start_workers() {
    local checkpoint_count worker_index pid_file
    prepare_runtime
    checkpoint_count=$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' | wc -l)
    if [ "$checkpoint_count" -eq 0 ]; then
        echo "No checkpoints found in $CKPT_DIR" >&2
        exit 1
    fi
    echo "checkpoint_count=$checkpoint_count"

    for ((worker_index=0; worker_index<NUM_WORKERS; worker_index++)); do
        pid_file=$(worker_pid_file "$worker_index")
        if worker_is_running "$worker_index"; then
            echo "worker_already_running gpu=$worker_index pid=$(<"$pid_file")"
            continue
        fi
        nohup "$0" worker "$worker_index" </dev/null >/dev/null 2>&1 &
        printf '%s\n' "$!" > "$pid_file"
        echo "worker_started gpu=$worker_index pid=$! log=${EVAL_ROOT}/eval_gpu${worker_index}.log"
    done
}

show_status() {
    local worker_index pid_file state=stopped pid=-
    mkdir -p "$PID_DIR"
    for ((worker_index=0; worker_index<NUM_WORKERS; worker_index++)); do
        pid_file=$(worker_pid_file "$worker_index")
        if worker_is_running "$worker_index"; then
            state=running
            pid=$(<"$pid_file")
        else
            state=stopped
            [ -s "$pid_file" ] && pid=$(<"$pid_file") || pid=-
        fi
        echo "worker=$worker_index state=$state pid=$pid log=${EVAL_ROOT}/eval_gpu${worker_index}.log"
    done
    echo "results=$(find "$RESULT_DIR" -maxdepth 1 -type f -name 'stats_ckpt_*_val_unseen.json' 2>/dev/null | wc -l)/$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'ckpt.iter*.pth' 2>/dev/null | wc -l)"
    echo "checkpoint_order=$CHECKPOINT_ORDER config=$CONFIG_FILE"
}

case "${1:-}" in
    start) start_workers ;;
    status) show_status ;;
    worker) run_worker "${2:-}" ;;
    *) usage; exit 2 ;;
esac
