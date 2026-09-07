#!/usr/bin/env bash
# Use the already established project runtime without changing either conda env.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Each Habitat worker otherwise imports numerical libraries with a large
# thread pool; two single-GPU jobs can create thousands of idle CPU threads.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GLOG_minloglevel=2 MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export MPLCONFIGDIR=/tmp/matplotlib-etpr1-rgb-optimization
case "${1:?server or eval}" in
  server)
    shift
    runtime=${ETPR1_SERVER_RUNTIME_ROOT:-$PWD/.runtime/server_sft}
    export PYTHONPATH="$PWD/scripts/benchmark_shims:$PWD:$PWD/vendor/legacy_clip:$runtime/habitat-lab:$runtime/python"
    export ETPR1_BENCH_HABITAT_BASELINES_ROOT="$runtime/habitat-baselines/habitat_baselines"
    export LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0
    python_bin=${ETPR1_SERVER_PYTHON:-/home/gwl/miniconda3/envs/etpnav_unified/bin/python}
    ;;
  eval)
    shift
    exec scripts/etpr1_rae_runtime_exec.sh /home/a6000/gwl/miniconda3/envs/etpr1_rae/bin/python "$@"
    ;;
  *) exit 2 ;;
esac
exec "$python_bin" "$@"
