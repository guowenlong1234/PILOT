#!/usr/bin/env bash
# One GPU, one checkpoint; identical navigation rules on both machines.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO_ROOT=$PWD
MODE=${1:?server or eval}
GPU=${2:?physical GPU index}
CHECKPOINT=$(realpath "${3:?checkpoint}")
OUTPUT=$(realpath -m "${4:?new output directory}")
ENVS=${5:?environment count}
EPISODES=${6:--1}
[[ "$GPU" =~ ^[0-9]+$ && "$ENVS" =~ ^[1-9][0-9]*$ ]]
[ -s "$CHECKPOINT" ]
[ ! -e "$OUTPUT" ] || { echo "Output exists: $OUTPUT" >&2; exit 2; }
exec 9>"/tmp/etpr1-rxr-checkpoint-gpu-${GPU}.lock"
flock -n 9 || { echo 'GPU evaluation lock busy' >&2; exit 2; }
used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$used" -le 1024 ] || { echo "GPU $GPU busy: $used MiB" >&2; exit 2; }
mkdir -p "$OUTPUT"
exec >"$OUTPUT/launcher.log" 2>&1
finish() { rc=$?; printf '%s\n' "$rc" >"$OUTPUT/exit_code"; date -Is >"$OUTPUT/finished_at"; }
trap finish EXIT
export CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export MPLCONFIGDIR="/tmp/matplotlib-etpr1-rxr-gpu-${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export TORCHINDUCTOR_COMPILE_THREADS=1
case "$MODE" in
    server)
        [ "$REPO_ROOT" = /home/gwl/project/etpr1/ETP-R1 ]
        PYTHON=/home/gwl/miniconda3/envs/etpnav_unified/bin/python
        RUNTIME_ROOT=$REPO_ROOT/.runtime/server_sft
        export PYTHONPATH="$REPO_ROOT/scripts/benchmark_shims:$REPO_ROOT:$REPO_ROOT/vendor/legacy_clip:$RUNTIME_ROOT/habitat-lab:$RUNTIME_ROOT/python"
        export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$RUNTIME_ROOT/habitat-baselines/habitat_baselines"
        export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
        RUN=("$PYTHON")
        ;;
    eval)
        [ "$REPO_ROOT" = /home/a6000/gwl/ETP-R1 ]
        source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh
        conda activate etpr1_rae
        RUN=(bash scripts/etpr1_rae_runtime_exec.sh python)
        ;;
    *) echo 'Unknown machine mode' >&2; exit 2 ;;
esac
EXP_NAME="rxr_queue_$(basename "$OUTPUT")"
{
    date -Is
    hostname
    whoami
    echo "pid=$$ checkpoint=$CHECKPOINT gpu=$GPU environments=$ENVS episodes=$EPISODES"
    git rev-parse HEAD
    "${RUN[@]}" -c 'import sys,torch,transformers,habitat,habitat_sim; print(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,habitat=habitat.__version__,habitat_sim=habitat_sim.__version__))'
} >"$OUTPUT/metadata.log"
echo "$$" >"$OUTPUT/supervisor.pid"
CMD=("${RUN[@]}" run.py --exp_name "$EXP_NAME" --run-type eval
    --exp-config run_rxr/iter_train_rae_dino_sft.yaml
    SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' TORCH_GPU_ID 0 GPU_NUMBERS 1
    NUM_ENVIRONMENTS "$ENVS" EVAL.SPLIT val_unseen EVAL.EPISODE_COUNT "$EPISODES"
    EVAL.fast_eval False EVAL.SAVE_RESULTS True EVAL.USE_CKPT_CONFIG False
    EVAL.CKPT_PATH_DIR "$CHECKPOINT"
    TASK_CONFIG.DATASET.SUFFIX '' TASK_CONFIG.DATASET.ROLES "['guide']"
    TASK_CONFIG.DATASET.LANGUAGES "['en-US','en-IN','hi-IN','te-IN']"
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False IL.back_algo control
    IL.RECOLLECT_TRAINER.gt_file 'data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz'
    MODEL.RAENWM.enabled False MODEL.ACTIVE_LOOKAHEAD.enabled False
    CHECKPOINT_FOLDER "$OUTPUT/checkpoints/" TENSORBOARD_DIR "$OUTPUT/tensorboard/"
    RESULTS_DIR "$OUTPUT/results/")
printf '%q ' "${CMD[@]}" >"$OUTPUT/command.sh"
printf '\n' >>"$OUTPUT/command.sh"
"${CMD[@]}"
"${RUN[@]}" scripts/run_rxr_checkpoint_queue.py validate --output "$OUTPUT" --episodes "$EPISODES"
