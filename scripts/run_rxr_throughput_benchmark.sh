#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
repo_root=$PWD
runtime_root=${ETPR1_SERVER_RUNTIME_ROOT:-$repo_root/.runtime/server_sft}
export PYTHONPATH="$repo_root/scripts/benchmark_shims:$repo_root:$repo_root/vendor/legacy_clip:$runtime_root/habitat-lab:$runtime_root/python"
export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$runtime_root/habitat-baselines/habitat_baselines"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
exec /home/gwl/miniconda3/envs/etpnav_unified/bin/torchrun --nproc_per_node=2 --master_port=24693 scripts/benchmark_rxr_throughput.py "$@"
