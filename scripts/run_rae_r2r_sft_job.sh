#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: run_rae_r2r_sft_job.sh <start|resume> <log_file>}
LOG_FILE=${2:?Usage: run_rae_r2r_sft_job.sh <start|resume> <log_file>}
EXP_NAME=${ETPR1_R2R_SFT_EXP_NAME:-rae_dinov2_r2r_sft}
OUTPUT_ROOT=${ETPR1_R2R_SFT_OUTPUT_ROOT:-data/logs/rae_dinov2/r2r_sft_formal}

case "$MODE" in
    start|resume) ;;
    *)
        echo "Unknown mode: $MODE" >&2
        exit 2
        ;;
esac

mkdir -p "$(dirname -- "$LOG_FILE")"
cd "$REPO_ROOT"

identity_file=${LOG_FILE%.log}_source_identity.json
manifest_file=${LOG_FILE%.log}_source_manifest.sha256
identity_args=(
    --project-root "$REPO_ROOT"
    --manifest-output "$manifest_file"
    --identity-output "$identity_file"
)
if [ -n "${ETPR1_SOURCE_COMMIT:-}" ]; then
    identity_args+=(--requested-commit "$ETPR1_SOURCE_COMMIT")
fi
python scripts/rae_smoke_source_identity.py "${identity_args[@]}" \
    >>"$LOG_FILE"
source_identity=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["identity"])' \
    "$identity_file")
source_kind=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["kind"])' \
    "$identity_file")

{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "mode=$MODE"
    echo "repo_root=$REPO_ROOT"
    echo "exp_name=$EXP_NAME"
    echo "output_root=$OUTPUT_ROOT"
    echo "source_identity=$source_identity"
    echo "source_kind=$source_kind"
    echo "source_manifest=$manifest_file"
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
} >>"$LOG_FILE"

resume_args=(IL.load_from_ckpt False IL.is_requeue False)
if [ "$MODE" = resume ]; then
    resume_args=(IL.load_from_ckpt True IL.is_requeue True)
fi

set +e
MPLCONFIGDIR=/tmp/matplotlib-etpr1 \
CUDA_VISIBLE_DEVICES=0 \
scripts/etpr1_rae_runtime_exec.sh python run.py \
    --exp_name "$EXP_NAME" \
    --run-type dagger \
    --exp-config run_r2r/iter_train_rae_dino.yaml \
    SIMULATOR_GPU_IDS "[0]" \
    TORCH_GPU_IDS "[0]" \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS 8 \
    IL.batch_size 8 \
    IL.gradient_accumulation_steps 4 \
    IL.iters 30000 \
    IL.log_every 200 \
    IL.lr 1e-5 \
    IL.ml_weight 1.0 \
    IL.sample_ratio 0.75 \
    IL.decay_interval 2000 \
    IL.warmup_iters 500 \
    IL.min_lr_ratio 1.0 \
    IL.waypoint_aug True \
    IL.amp_init_scale 16384.0 \
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
    MODEL.pretrained_path \
    pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/best/model_best_step_452500.pt \
    2>&1 | tee -a "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e

{
    echo "run_finished_at=$(date --iso-8601=seconds)"
    echo "exit_code=$exit_code"
} >>"$LOG_FILE"
exit "$exit_code"
