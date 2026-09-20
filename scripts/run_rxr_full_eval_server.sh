#!/usr/bin/env bash
# One checkpoint, full RxR val_unseen, two training-server GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD
CHECKPOINT=$(realpath "${1:?checkpoint required}")
OUTPUT_ROOT=$(realpath -m "${2:?new output directory required}")
PYTHON=/home/gwl/miniconda3/envs/etpnav_unified/bin/python
TORCHRUN=/home/gwl/miniconda3/envs/etpnav_unified/bin/torchrun
RUNTIME_ROOT=$REPO_ROOT/.runtime/server_sft
[ -s "$CHECKPOINT" ]
[ ! -e "$OUTPUT_ROOT" ] || { echo "Output directory already exists: $OUTPUT_ROOT" >&2; exit 2; }
while read -r used; do
    [ "$used" -le 1024 ] || { echo "GPU busy; refusing evaluation" >&2; exit 2; }
done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
mkdir -p "$OUTPUT_ROOT"
exec >"$OUTPUT_ROOT/launcher.log" 2>&1
export PYTHONPATH="$REPO_ROOT/scripts/benchmark_shims:$REPO_ROOT:$REPO_ROOT/vendor/legacy_clip:$RUNTIME_ROOT/habitat-lab:$RUNTIME_ROOT/python"
export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$RUNTIME_ROOT/habitat-baselines/habitat_baselines"
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MPLCONFIGDIR=/tmp/matplotlib-etpr1-rxr-full-eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
EXP_NAME=rxr_baseline_full_eval
{
    date -Is
    hostname
    whoami
    echo "pid=$$ checkpoint=$CHECKPOINT output=$OUTPUT_ROOT"
    git rev-parse HEAD
    "$PYTHON" -c 'import sys,torch,transformers,habitat,habitat_sim; print(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,habitat=habitat.__version__,habitat_sim=habitat_sim.__version__))'
} >"$OUTPUT_ROOT/metadata.log"
printf '%s\n' "$$" >"$OUTPUT_ROOT/supervisor.pid"
set +e
"$TORCHRUN" --nproc_per_node=2 --master_port=24703 \
    --log-dir "$OUTPUT_ROOT/torchrun" --redirects 3 \
    run.py --exp_name "$EXP_NAME" --run-type eval \
    --exp-config run_rxr/iter_train_rae_dino_sft.yaml \
    SIMULATOR_GPU_IDS '[0,1]' TORCH_GPU_IDS '[0,1]' GPU_NUMBERS 2 \
    NUM_ENVIRONMENTS 6 EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT -1 \
    EVAL.fast_eval False EVAL.SAVE_RESULTS True EVAL.USE_CKPT_CONFIG False \
    EVAL.CKPT_PATH_DIR "$CHECKPOINT" \
    TASK_CONFIG.DATASET.SUFFIX '' TASK_CONFIG.DATASET.ROLES "['guide']" \
    TASK_CONFIG.DATASET.LANGUAGES "['en-US','en-IN','hi-IN','te-IN']" \
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False \
    IL.back_algo control \
    IL.RECOLLECT_TRAINER.gt_file 'data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz' \
    CHECKPOINT_FOLDER "$OUTPUT_ROOT/checkpoints/" \
    TENSORBOARD_DIR "$OUTPUT_ROOT/tensorboard/" RESULTS_DIR "$OUTPUT_ROOT/results/"
result=$?
set -e
printf '%s\n' "$result" >"$OUTPUT_ROOT/exit_code"
date -Is >"$OUTPUT_ROOT/finished_at"
exit "$result"
