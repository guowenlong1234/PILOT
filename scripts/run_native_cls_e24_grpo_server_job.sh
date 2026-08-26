#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
DATASET=${1:?Usage: run_native_cls_e24_grpo_server_job.sh <r2r|rxr> <start|resume> <log_file>}
MODE=${2:?Usage: run_native_cls_e24_grpo_server_job.sh <r2r|rxr> <start|resume> <log_file>}
LOG_FILE=${3:?Usage: run_native_cls_e24_grpo_server_job.sh <r2r|rxr> <start|resume> <log_file>}

case "$DATASET:$MODE" in
    r2r:start|r2r:resume|rxr:start|rxr:resume) ;;
    *) echo "Expected r2r|rxr and start|resume" >&2; exit 2 ;;
esac

if [ "$DATASET" = r2r ]; then
    JOINT_CONFIG=run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml
    GRPO_CONFIG=run_r2r/grpo_native_cls_e24_frozen.yaml
    SOURCE_CHECKPOINT=${ETPR1_R2R_ACTIVE_GRPO_SOURCE_CHECKPOINT:-}
    EXP_NAME=${ETPR1_R2R_ACTIVE_GRPO_EXP_NAME:-r2r_native_cls_e24_frozen_grpo}
    OUTPUT_ROOT=${ETPR1_R2R_ACTIVE_GRPO_OUTPUT_ROOT:-data/logs/active_lookahead/r2r_native_cls_e24_grpo}
    MAX_TRAJ_LEN=15
    MAX_TEXT_LEN=150
    CWP_PATH=data/wp_pred/check_cwp_bestdist_hfov90
else
    JOINT_CONFIG=run_rxr/iter_train_rae_dino_native_cls_e24_joint.yaml
    GRPO_CONFIG=run_rxr/grpo_native_cls_e24_frozen.yaml
    SOURCE_CHECKPOINT=${ETPR1_RXR_ACTIVE_GRPO_SOURCE_CHECKPOINT:-}
    EXP_NAME=${ETPR1_RXR_ACTIVE_GRPO_EXP_NAME:-rxr_native_cls_e24_frozen_grpo}
    OUTPUT_ROOT=${ETPR1_RXR_ACTIVE_GRPO_OUTPUT_ROOT:-data/logs/active_lookahead/rxr_native_cls_e24_grpo}
    MAX_TRAJ_LEN=25
    MAX_TEXT_LEN=250
    CWP_PATH=data/wp_pred/check_cwp_bestdist_hfov63
fi

if [ -z "$SOURCE_CHECKPOINT" ] || [ ! -f "$SOURCE_CHECKPOINT" ]; then
    echo "Set the dataset-specific ACTIVE_GRPO_SOURCE_CHECKPOINT to a native joint-SFT checkpoint." >&2
    exit 1
fi

NPROC=${ETPR1_ACTIVE_GRPO_NPROC_PER_NODE:-2}
NUM_ENVIRONMENTS=${ETPR1_ACTIVE_GRPO_NUM_ENVIRONMENTS:-2}
BATCH_SIZE=${ETPR1_ACTIVE_GRPO_BATCH_SIZE:-$NUM_ENVIRONMENTS}
SAMPLE_NUM=${ETPR1_ACTIVE_GRPO_SAMPLE_NUM:-8}
GRPO_ITERS=${ETPR1_ACTIVE_GRPO_ITERS:-1000}
GRPO_LOG_EVERY=${ETPR1_ACTIVE_GRPO_LOG_EVERY:-10}
RUNTIME_ROOT=${ETPR1_SERVER_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/server_sft}
PYTHON_BIN=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}
TORCHRUN_BIN=${ETPR1_SERVER_TORCHRUN:-/home/gwl/miniconda3/envs/etpnav_unified/bin/torchrun}
HABITAT_LAB_ROOT=${RUNTIME_ROOT}/habitat-lab
RUNTIME_PYTHON=${RUNTIME_ROOT}/python

cd "$REPO_ROOT"
for path in "$PYTHON_BIN" "$TORCHRUN_BIN" "$SOURCE_CHECKPOINT" \
    "$JOINT_CONFIG" "$GRPO_CONFIG" "$CWP_PATH"; do
    [ -e "$path" ] || { echo "Missing active GRPO resource: $path" >&2; exit 1; }
done

SOURCE_SHA=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
resume_args=(GRPO.load_from_ckpt True GRPO.is_requeue False)
if [ "$MODE" = resume ]; then
    resume_args=(GRPO.load_from_ckpt True GRPO.is_requeue True)
fi

mkdir -p "$(dirname -- "$LOG_FILE")"
export MPLCONFIGDIR=/tmp/matplotlib-etpr1-active-grpo
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export MAGNUM_LOG=${MAGNUM_LOG:-quiet}
export HABITAT_SIM_LOG=${HABITAT_SIM_LOG:-quiet}
export PYTHONPATH="${REPO_ROOT}/scripts/benchmark_shims:${REPO_ROOT}:${REPO_ROOT}/vendor/legacy_clip:${HABITAT_LAB_ROOT}:${RUNTIME_PYTHON}"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "dataset=$DATASET"
    echo "mode=$MODE"
    echo "source_commit=$(git rev-parse HEAD)"
    echo "source_checkpoint=$SOURCE_CHECKPOINT"
    echo "source_checkpoint_sha256=$SOURCE_SHA"
    "$PYTHON_BIN" -c 'import sys, torch, transformers, habitat, habitat_sim; print(f"versions=python:{sys.version.split()[0]} torch:{torch.__version__} cuda:{torch.version.cuda} transformers:{transformers.__version__} habitat:{habitat.__version__} habitat_sim:{habitat_sim.__version__}")'
} >>"$LOG_FILE"

set +e
"$TORCHRUN_BIN" \
    --nproc_per_node="$NPROC" \
    --master_port="${ETPR1_ACTIVE_GRPO_MASTER_PORT:-24687}" \
    run.py \
    --exp_name "$EXP_NAME" \
    --run-type grpo \
    --exp-config "${JOINT_CONFIG},${GRPO_CONFIG}" \
    SIMULATOR_GPU_IDS "[0,1]" \
    TORCH_GPU_IDS "[0,1]" \
    GPU_NUMBERS "$NPROC" \
    NUM_ENVIRONMENTS "$NUM_ENVIRONMENTS" \
    GRPO.batch_size "$BATCH_SIZE" \
    GRPO.sample_num "$SAMPLE_NUM" \
    GRPO.iters "$GRPO_ITERS" \
    GRPO.log_every "$GRPO_LOG_EVERY" \
    GRPO.max_traj_len "$MAX_TRAJ_LEN" \
    GRPO.max_text_len "$MAX_TEXT_LEN" \
    GRPO.ckpt_to_load "$SOURCE_CHECKPOINT" \
    GRPO.reference_ckpt_to_load "$SOURCE_CHECKPOINT" \
    GRPO.reference_checkpoint_sha256 "$SOURCE_SHA" \
    "${resume_args[@]}" \
    MODEL.RAENWM.rgb_fusion_trainable False \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
    TASK_CONFIG.DATASET.SUFFIX _10 \
    CHECKPOINT_FOLDER "$OUTPUT_ROOT/checkpoints/$EXP_NAME" \
    TENSORBOARD_DIR "$OUTPUT_ROOT/tensorboard/$EXP_NAME" \
    RESULTS_DIR "$OUTPUT_ROOT/results/$EXP_NAME" \
    2>&1 | "$PYTHON_BIN" -u scripts/filter_habitat_startup_noise.py \
    | tee -a "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e
echo "exit_code=$exit_code" >>"$LOG_FILE"
exit "$exit_code"
