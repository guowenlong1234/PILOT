#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: run_rae_r2r_sft_server_job.sh <start|resume> <log_file>}
LOG_FILE=${2:?Usage: run_rae_r2r_sft_server_job.sh <start|resume> <log_file>}
EXP_NAME=${ETPR1_R2R_SFT_EXP_NAME:-rae_dinov2_etpnav_cls_768_r2r_sft}
OUTPUT_ROOT=${ETPR1_R2R_SFT_OUTPUT_ROOT:-data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_formal}
SFT_ITERS=${ETPR1_R2R_SFT_ITERS:-30000}
GRADIENT_ACCUMULATION_STEPS=${ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS:-2}
PRETRAINED_PATH=${ETPR1_R2R_SFT_PRETRAINED_PATH:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768/best/model_best_step_452500.pt}
RUNTIME_ROOT=${ETPR1_SERVER_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/server_sft}
PYTHON_BIN=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}
TORCHRUN_BIN=${ETPR1_SERVER_TORCHRUN:-/home/gwl/miniconda3/envs/etpnav_unified/bin/torchrun}
HABITAT_LAB_ROOT=${RUNTIME_ROOT}/habitat-lab
HABITAT_BASELINES_ROOT=${RUNTIME_ROOT}/habitat-baselines/habitat_baselines
RUNTIME_PYTHON=${RUNTIME_ROOT}/python

case "$MODE" in
    start|resume) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

for integer_setting in "$SFT_ITERS" "$GRADIENT_ACCUMULATION_STEPS"; do
    if ! [[ "$integer_setting" =~ ^[1-9][0-9]*$ ]]; then
        echo "SFT iteration and gradient accumulation settings must be positive integers: $integer_setting" >&2
        exit 2
    fi
done

for path in \
    "$PYTHON_BIN" \
    "$TORCHRUN_BIN" \
    "$HABITAT_LAB_ROOT/habitat/__init__.py" \
    "$HABITAT_BASELINES_ROOT/common/baseline_registry.py" \
    "$RUNTIME_PYTHON/dtw/__init__.py"; do
    if [ ! -e "$path" ]; then
        echo "Missing training-server runtime dependency: $path" >&2
        exit 1
    fi
done

mkdir -p "$(dirname -- "$LOG_FILE")"
cd "$REPO_ROOT"

if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "Missing SFT pretrained checkpoint: $PRETRAINED_PATH" >&2
    exit 1
fi

export MPLCONFIGDIR=/tmp/matplotlib-etpr1-server
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export MAGNUM_LOG=${MAGNUM_LOG:-quiet}
export HABITAT_SIM_LOG=${HABITAT_SIM_LOG:-quiet}
export PYTHONPATH="${REPO_ROOT}/scripts/benchmark_shims:${REPO_ROOT}:${REPO_ROOT}/vendor/legacy_clip:${HABITAT_LAB_ROOT}:${RUNTIME_PYTHON}"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$HABITAT_BASELINES_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

resume_args=(IL.load_from_ckpt False IL.is_requeue False)
if [ "$MODE" = resume ]; then
    resume_args=(IL.load_from_ckpt True IL.is_requeue True)
fi

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "mode=$MODE"
    echo "repo_root=$REPO_ROOT"
    echo "source_commit=$(git rev-parse HEAD)"
    echo "exp_name=$EXP_NAME"
    echo "output_root=$OUTPUT_ROOT"
    echo "sft_iters=$SFT_ITERS"
    echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
    echo "pretrained_path=$PRETRAINED_PATH"
    echo "python=$PYTHON_BIN"
    "$PYTHON_BIN" -c \
        'import sys, torch, transformers; print(f"versions=python:{sys.version.split()[0]} torch:{torch.__version__} cuda:{torch.version.cuda} transformers:{transformers.__version__}")'
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
} >>"$LOG_FILE"

set +e
"$TORCHRUN_BIN" \
    --nproc_per_node=2 \
    --master_port="${ETPR1_R2R_SFT_MASTER_PORT:-24673}" \
    run.py \
    --exp_name "$EXP_NAME" \
    --run-type dagger \
    --exp-config run_r2r/iter_train_rae_dino.yaml \
    SIMULATOR_GPU_IDS "[0,1]" \
    TORCH_GPU_IDS "[0,1]" \
    GPU_NUMBERS 2 \
    NUM_ENVIRONMENTS 8 \
    IL.batch_size 8 \
    IL.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    IL.iters "$SFT_ITERS" \
    IL.log_every 200 \
    IL.lr 1e-5 \
    IL.ml_weight 1.0 \
    IL.sample_ratio 0.75 \
    IL.decay_interval 2000 \
    IL.warmup_iters 500 \
    IL.min_lr_ratio 1.0 \
    IL.waypoint_aug True \
    IL.amp_init_scale 16384.0 \
    IL.use_fused_adamw True \
    IL.cudnn_benchmark True \
    IL.log_cuda_memory True \
    IL.resumable_checkpoints True \
    IL.keep_last_train_states 3 \
    IL.keep_train_state_every_n_iters 5000 \
    "${resume_args[@]}" \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    TASK_CONFIG.DATASET.SUFFIX _90 \
    ONLY_LAST_SAVEALL True \
    CHECKPOINT_FOLDER "$OUTPUT_ROOT/checkpoints/" \
    TENSORBOARD_DIR "$OUTPUT_ROOT/tensorboard/" \
    RESULTS_DIR "$OUTPUT_ROOT/results/" \
    MODEL.pretrained_path "$PRETRAINED_PATH" \
    MODEL.RGB_ENCODER.precision ambient \
    2>&1 \
    | "$PYTHON_BIN" -u scripts/filter_habitat_startup_noise.py \
    | tee -a "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e

{
    echo "run_finished_at=$(date --iso-8601=seconds)"
    echo "exit_code=$exit_code"
} >>"$LOG_FILE"
exit "$exit_code"
