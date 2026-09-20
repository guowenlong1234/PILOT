#!/usr/bin/env bash
# Separate outputs and explicit execution options for the measured fast variant.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ETPR1_RXR_SFT_CONFIG_FILE=run_rxr/iter_train_rae_dino_sft_fast.yaml
export ETPR1_RXR_SFT_EXP_NAME=${ETPR1_RXR_SFT_EXP_NAME:-rxr_dino_sft_fast}
export ETPR1_RXR_SFT_OUTPUT_ROOT=${ETPR1_RXR_SFT_OUTPUT_ROOT:-data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_fast}
export ETPR1_RXR_SFT_NUM_ENVIRONMENTS=${ETPR1_RXR_SFT_NUM_ENVIRONMENTS:-12}
export ETPR1_RXR_SFT_BATCH_SIZE=${ETPR1_RXR_SFT_BATCH_SIZE:-$ETPR1_RXR_SFT_NUM_ENVIRONMENTS}
export ETPR1_RXR_SFT_GRADIENT_ACCUMULATION_STEPS=${ETPR1_RXR_SFT_GRADIENT_ACCUMULATION_STEPS:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-2}
exec bash "$SCRIPT_DIR/manage_rae_rxr_sft_server.sh" "$@"
