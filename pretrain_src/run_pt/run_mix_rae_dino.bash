#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

NODE_RANK=0
NUM_GPUS=1
MASTER_PORT=${1:?Usage: run_mix_rae_dino.bash <master_port> [train arguments...]}
shift
outdir=${ETPR1_PRETRAIN_OUTPUT_DIR:-pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768}

cd "$REPO_ROOT"
"${REPO_ROOT}/scripts/etpr1_rae_runtime_exec.sh" torchrun \
    --nproc_per_node=1 --node_rank "$NODE_RANK" --master_port="$MASTER_PORT" \
    pretrain_src/pretrain_src/train_r2r.py --world_size 1 \
    --vlnbert cmt \
    --model_config pretrain_src/run_pt/mix_model_config_rae_dino.json \
    --config pretrain_src/run_pt/mix_pretrain_rae_dino.json \
    --output_dir "$outdir" \
    "$@"
