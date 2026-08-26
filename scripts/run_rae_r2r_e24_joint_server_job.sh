#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: run_rae_r2r_e24_joint_server_job.sh <start|resume> <log_file>}
LOG_FILE=${2:?Usage: run_rae_r2r_e24_joint_server_job.sh <start|resume> <log_file>}
CONFIG_FILE=${ETPR1_E24_JOINT_CONFIG_FILE:-run_r2r/iter_train_rae_dino_e24_joint.yaml}
EXP_NAME=${ETPR1_E24_JOINT_EXP_NAME:-etpr1_e24_joint_sft}
OUTPUT_ROOT=${ETPR1_E24_JOINT_OUTPUT_ROOT:-data/logs/active_lookahead/e24_joint_sft}
START_CKPT=${ETPR1_E24_JOINT_START_CKPT:-${REPO_ROOT}/pretrained/active_lookahead/base_iter14200.pth}
PRETRAIN_PATH=${ETPR1_E24_JOINT_PRETRAIN_PATH:-${REPO_ROOT}/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt}
CHECKPOINT_SYNC_DESTINATION=${ETPR1_E24_JOINT_SYNC_DESTINATION:-a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/data/logs/active_lookahead/e24_joint_sft/checkpoints/etpr1_e24_joint_sft}
CHECKPOINT_SYNC_ENABLED=${ETPR1_E24_JOINT_SYNC_ENABLED:-True}
TRAIN_ITERS=${ETPR1_E24_JOINT_ITERS:-10000}
LOG_EVERY=${ETPR1_E24_JOINT_LOG_EVERY:-200}
TASK_SEED=${ETPR1_E24_JOINT_TASK_SEED:-}
SMOKE_FREEZE_CHECK=${ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK:-False}
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

for integer_setting in "$TRAIN_ITERS" "$LOG_EVERY"; do
    [[ "$integer_setting" =~ ^[1-9][0-9]*$ ]] || {
        echo "Training iteration settings must be positive integers: $integer_setting" >&2
        exit 2
    }
done
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
    "$REPO_ROOT/pretrained/raenwm_stage0/checkpoint_step_70000.pth.tar" \
    "$REPO_ROOT/pretrained/raenwm_stage0/nwm_heads.pt" \
    "$REPO_ROOT/pretrained/raenwm_stage0/stat.pt" \
    "$REPO_ROOT/pretrained/active_lookahead/e24_avg3.pth" \
    "$REPO_ROOT/pretrained/active_lookahead/dino_cwp_best.pt"; do
    [ -e "$path" ] || { echo "Missing E24 joint SFT dependency: $path" >&2; exit 1; }
done

verify_sha256() {
    local path=$1 expected=$2 label=$3 actual
    actual=$(sha256sum -- "$path" | awk '{print $1}')
    [ "$actual" = "$expected" ] || {
        echo "$label SHA256 mismatch: expected=$expected actual=$actual path=$path" >&2
        exit 1
    }
}
verify_sha256 "$START_CKPT" 1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61 "base iter14200"
verify_sha256 "$REPO_ROOT/pretrained/active_lookahead/e24_avg3.pth" bae7a9664000235dfc6fb66b43a8a0a38e7645b876e6a6369f732eb7bf404ed8 "E24 avg3"
verify_sha256 "$REPO_ROOT/pretrained/active_lookahead/dino_cwp_best.pt" 6a45291219907dd027203d224f3f8400631651a83bd01c45b1dea55d93ec0979 "DINO-CWP"
verify_sha256 "$REPO_ROOT/pretrained/raenwm_stage0/checkpoint_step_70000.pth.tar" 392fe02045f7c11f006e1efb822914eee5826b8c6fdb2553c87e7de7c1c4df36 "NWM body"
verify_sha256 "$REPO_ROOT/pretrained/raenwm_stage0/nwm_heads.pt" a4d396021b8bf670c44565c383db1c9288c3bd8988624fa0a59064b6a81dc6e2 "NWM heads"
verify_sha256 "$REPO_ROOT/pretrained/raenwm_stage0/stat.pt" 84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59 "NWM stat"

mkdir -p "$(dirname -- "$LOG_FILE")"
cd "$REPO_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-etpr1-e24_joint-server
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export MAGNUM_LOG=${MAGNUM_LOG:-quiet}
export HABITAT_SIM_LOG=${HABITAT_SIM_LOG:-quiet}
export PYTHONPATH="${REPO_ROOT}/scripts/benchmark_shims:${REPO_ROOT}:${REPO_ROOT}/vendor/legacy_clip:${HABITAT_LAB_ROOT}:${RUNTIME_PYTHON}"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$HABITAT_BASELINES_ROOT"
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

resume_args=(IL.load_from_ckpt True IL.is_requeue False IL.ckpt_to_load "$START_CKPT")
if [ "$MODE" = resume ]; then
    resume_args=(IL.load_from_ckpt True IL.is_requeue True)
fi

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "mode=$MODE"
    echo "source_commit=$(git rev-parse HEAD)"
    echo "config_file=$CONFIG_FILE"
    echo "exp_name=$EXP_NAME"
    echo "output_root=$OUTPUT_ROOT"
    echo "start_checkpoint=$START_CKPT"
    echo "pretrain_path=$PRETRAIN_PATH"
    echo "checkpoint_sync_destination=$CHECKPOINT_SYNC_DESTINATION"
    echo "checkpoint_sync_enabled=$CHECKPOINT_SYNC_ENABLED"
    echo "train_iters=$TRAIN_ITERS"
    echo "log_every=$LOG_EVERY"
    echo "task_seed=${TASK_SEED:-config_default}"
    "$PYTHON_BIN" -c 'import sys, torch, transformers, habitat, habitat_sim; habitat_version=getattr(habitat, "__version__", "unknown"); habitat_sim_version=getattr(habitat_sim, "__version__", "unknown"); print(f"versions=python:{sys.version.split()[0]} torch:{torch.__version__} cuda:{torch.version.cuda} transformers:{transformers.__version__} habitat:{habitat_version} habitat_sim:{habitat_sim_version}")'
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
} >>"$LOG_FILE"

set +e
seed_args=()
if [ -n "$TASK_SEED" ]; then
    seed_args=(TASK_CONFIG.SEED "$TASK_SEED")
fi
"$TORCHRUN_BIN" \
    --nproc_per_node=2 \
    --master_port="${ETPR1_E24_JOINT_MASTER_PORT:-24724}" \
    run.py \
    --exp_name "$EXP_NAME" \
    --run-type dagger \
    --exp-config "$CONFIG_FILE" \
    IL.iters "$TRAIN_ITERS" \
    IL.log_every "$LOG_EVERY" \
    IL.checkpoint_sync_enabled "$CHECKPOINT_SYNC_ENABLED" \
    IL.checkpoint_sync_destination "$CHECKPOINT_SYNC_DESTINATION" \
    "${seed_args[@]}" \
    "${resume_args[@]}" \
    CHECKPOINT_FOLDER "$OUTPUT_ROOT/checkpoints/" \
    TENSORBOARD_DIR "$OUTPUT_ROOT/tensorboard/" \
    RESULTS_DIR "$OUTPUT_ROOT/results/" \
    MODEL.pretrained_path "$PRETRAIN_PATH" \
    MODEL.ACTIVE_LOOKAHEAD.smoke_freeze_check "$SMOKE_FREEZE_CHECK" \
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
