#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: run_rae_r2r_native_cls_rgb_fusion_server_job.sh <start|resume> <log_file>}
LOG_FILE=${2:?Usage: run_rae_r2r_native_cls_rgb_fusion_server_job.sh <start|resume> <log_file>}
CONFIG_FILE=${ETPR1_RGB_FUSION_CONFIG_FILE:-run_r2r/iter_train_rae_dino_native_cls_rgb_fusion.yaml}
EXP_NAME=${ETPR1_RGB_FUSION_EXP_NAME:-etpr1_native_cls_rgb_fusion_sft}
OUTPUT_ROOT=${ETPR1_RGB_FUSION_OUTPUT_ROOT:-data/logs/raenwm_rgb_fusion/native_cls_sft}
START_CKPT=${ETPR1_RGB_FUSION_START_CKPT:-${REPO_ROOT}/pretrained/active_lookahead/base_iter14200.pth}
START_CKPT_SHA=${ETPR1_RGB_FUSION_START_CKPT_SHA:-1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61}
BASE_ITERATION=${ETPR1_RGB_FUSION_BASE_ITERATION:-14200}
PRETRAIN_PATH=${ETPR1_RGB_FUSION_PRETRAIN_PATH:-${REPO_ROOT}/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
NWM_CHECKPOINT=${ETPR1_RGB_FUSION_NWM_CHECKPOINT:-${REPO_ROOT}/pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar}
NWM_CHECKPOINT_SHA=${ETPR1_RGB_FUSION_NWM_CHECKPOINT_SHA:-38b24af13b76ba8faef367559244c3a0401e0557e7c299870c273cbee8a07064}
NWM_STAT=${ETPR1_RGB_FUSION_NWM_STAT:-${REPO_ROOT}/pretrained/raenwm_stage0/stat.pt}
NWM_STAT_SHA=${ETPR1_RGB_FUSION_NWM_STAT_SHA:-84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59}
CHECKPOINT_SYNC_DESTINATION=${ETPR1_RGB_FUSION_SYNC_DESTINATION:-a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/data/logs/raenwm_rgb_fusion/native_cls_sft/checkpoints/etpr1_native_cls_rgb_fusion_sft}
CHECKPOINT_SYNC_ENABLED=${ETPR1_RGB_FUSION_SYNC_ENABLED:-True}
TRAIN_ITERS=${ETPR1_RGB_FUSION_ITERS:-10000}
LOG_EVERY=${ETPR1_RGB_FUSION_LOG_EVERY:-200}
GRADIENT_ACCUMULATION_STEPS=${ETPR1_RGB_FUSION_GRADIENT_ACCUMULATION_STEPS:-1}
NPROC_PER_NODE=${ETPR1_RGB_FUSION_NPROC_PER_NODE:-2}
NUM_ENVIRONMENTS=${ETPR1_RGB_FUSION_NUM_ENVIRONMENTS:-4}
BATCH_SIZE=${ETPR1_RGB_FUSION_BATCH_SIZE:-4}
CUDA_DEVICES=${ETPR1_RGB_FUSION_CUDA_VISIBLE_DEVICES:-0,1}
TASK_SEED=${ETPR1_RGB_FUSION_TASK_SEED:-}
RUNTIME_ROOT=${ETPR1_SERVER_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/server_sft}
PYTHON_BIN=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}
TORCHRUN_BIN=${ETPR1_SERVER_TORCHRUN:-/home/gwl/miniconda3/envs/etpnav_unified/bin/torchrun}
HABITAT_LAB_ROOT=${RUNTIME_ROOT}/habitat-lab
HABITAT_BASELINES_ROOT=${RUNTIME_ROOT}/habitat-baselines/habitat_baselines
RUNTIME_PYTHON=${RUNTIME_ROOT}/python

case "$MODE" in
    start|resume) ;;
    *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac

for integer_setting in \
    "$TRAIN_ITERS" "$LOG_EVERY" "$GRADIENT_ACCUMULATION_STEPS" \
    "$NPROC_PER_NODE" "$NUM_ENVIRONMENTS" "$BATCH_SIZE" "$BASE_ITERATION"; do
    [[ "$integer_setting" =~ ^[1-9][0-9]*$ ]] || {
        echo "RGB-fusion training settings must be positive integers: $integer_setting" >&2
        exit 2
    }
done
case "$NPROC_PER_NODE" in
    1) CONFIG_GPU_IDS='[0]' ;;
    2) CONFIG_GPU_IDS='[0,1]' ;;
    *) echo "RGB-fusion workflow supports one or two ranks, got $NPROC_PER_NODE" >&2; exit 2 ;;
esac
if [ -n "$TASK_SEED" ] && ! [[ "$TASK_SEED" =~ ^[0-9]+$ ]]; then
    echo "TASK_CONFIG.SEED must be a non-negative integer: $TASK_SEED" >&2
    exit 2
fi

for path in \
    "$PYTHON_BIN" \
    "$TORCHRUN_BIN" \
    "$HABITAT_LAB_ROOT/habitat/__init__.py" \
    "$HABITAT_BASELINES_ROOT/common/baseline_registry.py" \
    "$RUNTIME_PYTHON/dtw/__init__.py" \
    "$REPO_ROOT/$CONFIG_FILE" \
    "$START_CKPT" \
    "$PRETRAIN_PATH" \
    "$NWM_CHECKPOINT" \
    "$NWM_STAT"; do
    [ -e "$path" ] || {
        echo "Missing RGB-fusion SFT dependency: $path" >&2
        exit 1
    }
done

verify_sha256() {
    local path=$1 expected=$2 label=$3 actual
    actual=$(sha256sum -- "$path" | awk '{print $1}')
    [ "$actual" = "$expected" ] || {
        echo "$label SHA256 mismatch: expected=$expected actual=$actual path=$path" >&2
        exit 1
    }
}
verify_sha256 "$START_CKPT" "$START_CKPT_SHA" "navigation base"
verify_sha256 "$NWM_CHECKPOINT" "$NWM_CHECKPOINT_SHA" "native CLS NWM"
verify_sha256 "$NWM_STAT" "$NWM_STAT_SHA" "NWM stat"

mkdir -p "$(dirname -- "$LOG_FILE")"
cd "$REPO_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-etpr1-rgb-fusion-server
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export MAGNUM_LOG=${MAGNUM_LOG:-quiet}
export HABITAT_SIM_LOG=${HABITAT_SIM_LOG:-quiet}
export PYTHONPATH="${REPO_ROOT}/scripts/benchmark_shims:${REPO_ROOT}:${REPO_ROOT}/vendor/legacy_clip:${HABITAT_LAB_ROOT}:${RUNTIME_PYTHON}"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$HABITAT_BASELINES_ROOT"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

resume_args=(IL.load_from_ckpt True IL.is_requeue False IL.ckpt_to_load "$START_CKPT")
if [ "$MODE" = resume ]; then
    resume_args=(IL.load_from_ckpt True IL.is_requeue True)
fi
seed_args=()
if [ -n "$TASK_SEED" ]; then
    seed_args=(TASK_CONFIG.SEED "$TASK_SEED")
fi

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "mode=$MODE"
    echo "source_commit=$(git rev-parse HEAD)"
    echo "config_file=$CONFIG_FILE"
    echo "exp_name=$EXP_NAME"
    echo "output_root=$OUTPUT_ROOT"
    echo "start_checkpoint=$START_CKPT"
    echo "start_checkpoint_sha256=$START_CKPT_SHA"
    echo "base_iteration=$BASE_ITERATION"
    echo "native_cls_nwm_checkpoint=$NWM_CHECKPOINT"
    echo "native_cls_nwm_checkpoint_sha256=$NWM_CHECKPOINT_SHA"
    echo "nwm_stat=$NWM_STAT"
    echo "nwm_stat_sha256=$NWM_STAT_SHA"
    echo "nwm_sampling=euler num_steps=10 final_only_euler=False"
    echo "active_lookahead_enabled=False"
    echo "rgb_fusion_enabled=True trainable=True gate_bias_init=0.0"
    echo "train_iters=$TRAIN_ITERS"
    echo "log_every=$LOG_EVERY"
    echo "nproc_per_node=$NPROC_PER_NODE"
    echo "num_environments=$NUM_ENVIRONMENTS"
    echo "batch_size=$BATCH_SIZE"
    echo "gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
    echo "global_batch_size=$((NPROC_PER_NODE * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
    echo "lr=1e-5 sample_ratio=0.75 decay_interval=3000 sample_ratio_iteration_offset=$BASE_ITERATION"
    echo "warmup_iters=0 min_lr_ratio=1.0 waypoint_aug=True"
    echo "checkpoint_sync_destination=$CHECKPOINT_SYNC_DESTINATION"
    echo "checkpoint_sync_enabled=$CHECKPOINT_SYNC_ENABLED"
    echo "task_seed=${TASK_SEED:-config_default}"
    "$PYTHON_BIN" -c 'import sys, torch, transformers, habitat, habitat_sim; habitat_version=getattr(habitat, "__version__", "unknown"); habitat_sim_version=getattr(habitat_sim, "__version__", "unknown"); print(f"versions=python:{sys.version.split()[0]} torch:{torch.__version__} cuda:{torch.version.cuda} transformers:{transformers.__version__} habitat:{habitat_version} habitat_sim:{habitat_sim_version}")'
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
} >>"$LOG_FILE"

set +e
"$TORCHRUN_BIN" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="${ETPR1_RGB_FUSION_MASTER_PORT:-24736}" \
    run.py \
    --exp_name "$EXP_NAME" \
    --run-type dagger \
    --exp-config "$CONFIG_FILE" \
    IL.iters "$TRAIN_ITERS" \
    IL.log_every "$LOG_EVERY" \
    IL.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    IL.lr 1.0e-5 \
    IL.sample_ratio 0.75 \
    IL.decay_interval 3000 \
    IL.sample_ratio_iteration_offset "$BASE_ITERATION" \
    IL.sample_ratio_zero_threshold 0.0 \
    IL.warmup_iters 0 \
    IL.min_lr_ratio 1.0 \
    IL.waypoint_aug True \
    IL.amp_init_scale 16384.0 \
    IL.use_fused_adamw True \
    IL.cudnn_benchmark True \
    IL.log_cuda_memory True \
    IL.resumable_checkpoints True \
    IL.keep_last_train_states 3 \
    IL.keep_train_state_every_n_iters 2000 \
    IL.checkpoint_sync_enabled "$CHECKPOINT_SYNC_ENABLED" \
    IL.checkpoint_sync_destination "$CHECKPOINT_SYNC_DESTINATION" \
    SIMULATOR_GPU_IDS "$CONFIG_GPU_IDS" \
    TORCH_GPU_IDS "$CONFIG_GPU_IDS" \
    GPU_NUMBERS "$NPROC_PER_NODE" \
    NUM_ENVIRONMENTS "$NUM_ENVIRONMENTS" \
    IL.batch_size "$BATCH_SIZE" \
    "${seed_args[@]}" \
    "${resume_args[@]}" \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    TASK_CONFIG.DATASET.SUFFIX _90 \
    ONLY_LAST_SAVEALL True \
    CHECKPOINT_FOLDER "$OUTPUT_ROOT/checkpoints/" \
    TENSORBOARD_DIR "$OUTPUT_ROOT/tensorboard/" \
    RESULTS_DIR "$OUTPUT_ROOT/results/" \
    MODEL.pretrained_path "$PRETRAIN_PATH" \
    MODEL.RAENWM.checkpoint_path "$NWM_CHECKPOINT" \
    MODEL.RAENWM.checkpoint_sha256 "$NWM_CHECKPOINT_SHA" \
    MODEL.RAENWM.stat_path "$NWM_STAT" \
    MODEL.RAENWM.stat_sha256 "$NWM_STAT_SHA" \
    MODEL.RAENWM.num_steps 10 \
    MODEL.RAENWM.final_only_euler False \
    MODEL.RAENWM.rgb_fusion_enabled True \
    MODEL.RAENWM.rgb_fusion_trainable True \
    MODEL.RAENWM.rgb_fusion_gate_bias_init 0.0 \
    MODEL.ACTIVE_LOOKAHEAD.enabled False \
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
