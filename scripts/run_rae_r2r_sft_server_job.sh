#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: run_rae_r2r_sft_server_job.sh <start|resume> <log_file>}
LOG_FILE=${2:?Usage: run_rae_r2r_sft_server_job.sh <start|resume> <log_file>}
EXP_NAME=${ETPR1_R2R_SFT_EXP_NAME:-rae_dinov2_etpnav_cls_768_r2r_sft_panorama_order}
OUTPUT_ROOT=${ETPR1_R2R_SFT_OUTPUT_ROOT:-data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_panorama_order}
CONFIG_FILE=${ETPR1_R2R_SFT_CONFIG_FILE:-run_r2r/iter_train_rae_dino.yaml}
SFT_ITERS=${ETPR1_R2R_SFT_ITERS:-2000}
LOG_EVERY=${ETPR1_R2R_SFT_LOG_EVERY:-200}
GRADIENT_ACCUMULATION_STEPS=${ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS:-1}
PRETRAINED_PATH=${ETPR1_R2R_SFT_PRETRAINED_PATH:-/mnt/data2tb/ETP-R1_data/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_raw_cls_20260810/best/model_best_step_220000.pt}
CHECKPOINT_SYNC_ENABLED=${ETPR1_R2R_SFT_CHECKPOINT_SYNC_ENABLED:-True}
CHECKPOINT_SYNC_DESTINATION=${ETPR1_R2R_SFT_CHECKPOINT_SYNC_DESTINATION:-a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_panorama_order/checkpoints/rae_dinov2_etpnav_cls_768_r2r_sft_panorama_order}
NPROC_PER_NODE=${ETPR1_R2R_SFT_NPROC_PER_NODE:-2}
NUM_ENVIRONMENTS=${ETPR1_R2R_SFT_NUM_ENVIRONMENTS:-8}
BATCH_SIZE=${ETPR1_R2R_SFT_BATCH_SIZE:-8}
CUDA_DEVICES=${ETPR1_R2R_SFT_CUDA_VISIBLE_DEVICES:-0,1}
SFT_LR=${ETPR1_R2R_SFT_LR:-1e-5}
SAMPLE_RATIO=${ETPR1_R2R_SFT_SAMPLE_RATIO:-0.75}
DECAY_INTERVAL=${ETPR1_R2R_SFT_DECAY_INTERVAL:-3000}
WARMUP_ITERS=${ETPR1_R2R_SFT_WARMUP_ITERS:-500}
MIN_LR_RATIO=${ETPR1_R2R_SFT_MIN_LR_RATIO:-1.0}
WAYPOINT_AUG=${ETPR1_R2R_SFT_WAYPOINT_AUG:-True}
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

for integer_setting in "$SFT_ITERS" "$LOG_EVERY" "$GRADIENT_ACCUMULATION_STEPS" \
    "$NPROC_PER_NODE" "$NUM_ENVIRONMENTS" "$BATCH_SIZE"; do
    if ! [[ "$integer_setting" =~ ^[1-9][0-9]*$ ]]; then
        echo "SFT iteration and gradient accumulation settings must be positive integers: $integer_setting" >&2
        exit 2
    fi
done
case "$NPROC_PER_NODE" in
    1) CONFIG_GPU_IDS='[0]' ;;
    2) CONFIG_GPU_IDS='[0,1]' ;;
    *) echo "SFT workflow supports one or two ranks, got $NPROC_PER_NODE" >&2; exit 2 ;;
esac

for path in \
    "$PYTHON_BIN" \
    "$TORCHRUN_BIN" \
    "$HABITAT_LAB_ROOT/habitat/__init__.py" \
    "$HABITAT_BASELINES_ROOT/common/baseline_registry.py" \
    "$RUNTIME_PYTHON/dtw/__init__.py" \
    "$REPO_ROOT/$CONFIG_FILE"; do
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
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

resume_args=(IL.load_from_ckpt False IL.is_requeue False)
if [ "$MODE" = resume ]; then
    resume_args=(IL.load_from_ckpt True IL.is_requeue True)
fi
task_dataset_args=()
case "$CONFIG_FILE" in
    run_rxr/*) task_dataset_args=(
        TASK_CONFIG.DATASET.ROLES "['guide']"
        TASK_CONFIG.DATASET.LANGUAGES "['en-US','en-IN','hi-IN','te-IN']"
    ) ;;
esac

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "mode=$MODE"
    echo "repo_root=$REPO_ROOT"
    echo "source_commit=$(git rev-parse HEAD)"
    echo "exp_name=$EXP_NAME"
    echo "output_root=$OUTPUT_ROOT"
    echo "config_file=$CONFIG_FILE"
    echo "sft_iters=$SFT_ITERS"
    echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
    echo "nproc_per_node=$NPROC_PER_NODE"
    echo "num_environments=$NUM_ENVIRONMENTS"
    echo "batch_size=$BATCH_SIZE"
    echo "pretrained_path=$PRETRAINED_PATH"
    echo "checkpoint_sync_enabled=$CHECKPOINT_SYNC_ENABLED"
    echo "checkpoint_sync_destination=$CHECKPOINT_SYNC_DESTINATION"
    echo "python=$PYTHON_BIN"
    "$PYTHON_BIN" -c \
        'import sys, torch, transformers, habitat, habitat_sim; print(f"versions=python:{sys.version.split()[0]} torch:{torch.__version__} cuda:{torch.version.cuda} transformers:{transformers.__version__} habitat:{habitat.__version__} habitat_sim:{habitat_sim.__version__}")'
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
} >>"$LOG_FILE"

set +e
"$TORCHRUN_BIN" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="${ETPR1_R2R_SFT_MASTER_PORT:-24673}" \
    run.py \
    --exp_name "$EXP_NAME" \
    --run-type dagger \
    --exp-config "$CONFIG_FILE" \
    SIMULATOR_GPU_IDS "$CONFIG_GPU_IDS" \
    TORCH_GPU_IDS "$CONFIG_GPU_IDS" \
    GPU_NUMBERS "$NPROC_PER_NODE" \
    NUM_ENVIRONMENTS "$NUM_ENVIRONMENTS" \
    IL.batch_size "$BATCH_SIZE" \
    IL.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    IL.iters "$SFT_ITERS" \
    IL.log_every "$LOG_EVERY" \
    IL.lr "$SFT_LR" \
    IL.ml_weight 1.0 \
    IL.sample_ratio "$SAMPLE_RATIO" \
    IL.decay_interval "$DECAY_INTERVAL" \
    IL.warmup_iters "$WARMUP_ITERS" \
    IL.min_lr_ratio "$MIN_LR_RATIO" \
    IL.waypoint_aug "$WAYPOINT_AUG" \
    IL.amp_init_scale 16384.0 \
    IL.use_fused_adamw True \
    IL.cudnn_benchmark True \
    IL.log_cuda_memory True \
    IL.resumable_checkpoints True \
    IL.keep_last_train_states 3 \
    IL.keep_train_state_every_n_iters 5000 \
    IL.checkpoint_sync_enabled "$CHECKPOINT_SYNC_ENABLED" \
    IL.checkpoint_sync_destination "$CHECKPOINT_SYNC_DESTINATION" \
    "${resume_args[@]}" \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    TASK_CONFIG.DATASET.SUFFIX _90 \
    "${task_dataset_args[@]}" \
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
