#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: run_rxr_single_episode_eval_server.sh <baseline|joint> <checkpoint> <output-root>}
CHECKPOINT=$(realpath -- "${2:?checkpoint is required}")
OUTPUT_ROOT=${3:?output root is required}
PYTHON_BIN=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}
RUNTIME_ROOT=${ETPR1_SERVER_RUNTIME_ROOT:-${REPO_ROOT}/.runtime/server_sft}
HABITAT_LAB_ROOT=${RUNTIME_ROOT}/habitat-lab
HABITAT_BASELINES_ROOT=${RUNTIME_ROOT}/habitat-baselines/habitat_baselines
RUNTIME_PYTHON=${RUNTIME_ROOT}/python

case "$MODE" in
    baseline)
        CONFIG_FILE=run_rxr/iter_train_rae_dino_sft.yaml
        EXP_NAME=etpr1_rxr_sft_single_episode
        ;;
    joint)
        CONFIG_FILE=run_rxr/iter_train_rae_dino_native_cls_e24_joint.yaml
        EXP_NAME=etpr1_rxr_native_cls_e24_single_episode
        ;;
    *) echo "mode must be baseline or joint" >&2; exit 2 ;;
esac

for path in "$CHECKPOINT" "$PYTHON_BIN" "$REPO_ROOT/$CONFIG_FILE"; do
    [ -s "$path" ] || { echo "Missing RxR evaluation dependency: $path" >&2; exit 1; }
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
[ "$used" -le "${ETPR1_RXR_EVAL_GPU_IDLE_LIMIT_MIB:-1024}" ] || {
    echo "Training GPU is busy (${used} MiB); refusing single-episode evaluation" >&2
    exit 1
}

cd "$REPO_ROOT"
export MPLCONFIGDIR=/tmp/matplotlib-etpr1-rxr-single-eval
export GLOG_minloglevel=${GLOG_minloglevel:-2}
export MAGNUM_LOG=${MAGNUM_LOG:-quiet}
export HABITAT_SIM_LOG=${HABITAT_SIM_LOG:-quiet}
export PYTHONPATH="${REPO_ROOT}/scripts/benchmark_shims:${REPO_ROOT}:${REPO_ROOT}/vendor/legacy_clip:${HABITAT_LAB_ROOT}:${RUNTIME_PYTHON}"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$HABITAT_BASELINES_ROOT"
export CUDA_VISIBLE_DEVICES=${ETPR1_RXR_EVAL_CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.8}

extra_config_args=()
if [ "$MODE" = joint ]; then
    mapfile -t provenance_fields < <(
        "$PYTHON_BIN" - "$CHECKPOINT" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu")
provenance = checkpoint.get("e24_joint_provenance") or {}
for name in (
    "base_checkpoint_sha256", "base_iteration",
    "base_selection_manifest_sha256", "sample_ratio_iteration_offset",
):
    if name not in provenance:
        raise ValueError(f"missing RxR provenance field {name}")
    print(provenance[name])
PY
    )
    [ "${#provenance_fields[@]}" -eq 4 ]
    extra_config_args=(
        MODEL.ACTIVE_LOOKAHEAD.base_checkpoint_sha256 "${provenance_fields[0]}"
        MODEL.ACTIVE_LOOKAHEAD.base_iteration "${provenance_fields[1]}"
        MODEL.ACTIVE_LOOKAHEAD.base_selection_manifest_sha256 "${provenance_fields[2]}"
        IL.sample_ratio_iteration_offset "${provenance_fields[3]}"
    )
fi

mkdir -p "$OUTPUT_ROOT"
LOG_FILE=${OUTPUT_ROOT}/single_episode.log
{
    echo "run_started_at=$(date --iso-8601=seconds)"
    echo "mode=$MODE checkpoint=$CHECKPOINT source_commit=$(git rev-parse HEAD)"
    "$PYTHON_BIN" -c 'import sys, torch, transformers, habitat, habitat_sim; print("versions=python:%s torch:%s cuda:%s transformers:%s habitat:%s habitat_sim:%s" % (sys.version.split()[0], torch.__version__, torch.version.cuda, transformers.__version__, getattr(habitat, "__version__", "unknown"), getattr(habitat_sim, "__version__", "unknown")))'
} | tee "$LOG_FILE"

set +e
"$PYTHON_BIN" run.py \
    --exp_name "$EXP_NAME" \
    --run-type eval \
    --exp-config "$CONFIG_FILE" \
    SIMULATOR_GPU_IDS "[0]" \
    TORCH_GPU_IDS "[0]" \
    TORCH_GPU_ID 0 \
    GPU_NUMBERS 1 \
    NUM_ENVIRONMENTS 1 \
    EVAL.CKPT_PATH_DIR "$CHECKPOINT" \
    EVAL.EPISODE_COUNT 1 \
    EVAL.SAVE_RESULTS True \
    CHECKPOINT_FOLDER "$OUTPUT_ROOT/checkpoints/" \
    TENSORBOARD_DIR "$OUTPUT_ROOT/tensorboard/" \
    RESULTS_DIR "$OUTPUT_ROOT/results/" \
    TASK_CONFIG.DATASET.SUFFIX "" \
    TASK_CONFIG.DATASET.ROLES "['guide']" \
    TASK_CONFIG.DATASET.LANGUAGES "['en-US','en-IN','hi-IN','te-IN']" \
    "${extra_config_args[@]}" \
    2>&1 | "$PYTHON_BIN" -u scripts/filter_habitat_startup_noise.py | tee -a "$LOG_FILE"
exit_code=${PIPESTATUS[0]}
set -e
[ "$exit_code" -eq 0 ] || exit "$exit_code"

result_dir=${OUTPUT_ROOT}/results/${EXP_NAME}/eval_results
"$PYTHON_BIN" - "$MODE" "$result_dir" <<'PY'
import glob
import json
import math
import os
import sys
mode, result_dir = sys.argv[1:]
stats = glob.glob(os.path.join(result_dir, "stats_ckpt_*_val_unseen.json"))
if len(stats) != 1:
    raise ValueError(f"expected one aggregate result, got {stats}")
with open(stats[0], "r", encoding="utf-8") as stream:
    payload = json.load(stream)
if not payload or any(not math.isfinite(float(v)) for v in payload.values() if isinstance(v, (int, float))):
    raise ValueError("single-episode navigation result is invalid")
if mode == "joint":
    paths = glob.glob(os.path.join(result_dir, "lookahead_ckpt_*_val_unseen.json"))
    if len(paths) != 1:
        raise ValueError("joint single-episode diagnostic is missing")
    with open(paths[0], "r", encoding="utf-8") as stream:
        diagnostic = json.load(stream)
    if diagnostic.get("episodes") != 1 or float(diagnostic.get("oracle_q1_calls", -1)) != 0:
        raise ValueError("joint single-episode diagnostic violates the Oracle contract")
    metrics = diagnostic.get("metrics", {})
    failures = [name for name, value in metrics.items() if name.endswith(("batch_failures", "row_failures")) and float(value) != 0]
    if failures:
        raise ValueError(f"joint single-episode diagnostic contains failures: {failures}")
print(json.dumps({"mode": mode, "stats": stats[0], "validated": True}, sort_keys=True))
PY
