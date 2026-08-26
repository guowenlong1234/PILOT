#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
STAGE=${ETPR1_RXR_SYNC_STAGE:-baseline}

case "$STAGE" in
    baseline)
        export ETPR1_R2R_SFT_SYNC_SOURCE_DIR=${ETPR1_RXR_SYNC_SOURCE_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/checkpoints/etpr1_rxr_rae_dino_sft}
        export ETPR1_R2R_SFT_SYNC_DEST_DIR=${ETPR1_RXR_SYNC_DEST_DIR:-/home/a6000/gwl/ETP-R1/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/checkpoints/etpr1_rxr_rae_dino_sft}
        ;;
    joint)
        export ETPR1_R2R_SFT_SYNC_SOURCE_DIR=${ETPR1_RXR_SYNC_SOURCE_DIR:-${REPO_ROOT}/data/logs/active_lookahead/rxr_native_cls_e24_joint_sft/checkpoints/etpr1_rxr_native_cls_e24_joint_sft}
        export ETPR1_R2R_SFT_SYNC_DEST_DIR=${ETPR1_RXR_SYNC_DEST_DIR:-/home/a6000/gwl/ETP-R1/data/logs/active_lookahead/rxr_native_cls_e24_joint_sft/checkpoints/etpr1_rxr_native_cls_e24_joint_sft}
        ;;
    *) echo "ETPR1_RXR_SYNC_STAGE must be baseline or joint" >&2; exit 2 ;;
esac
export ETPR1_R2R_SFT_SYNC_DEST_HOST=${ETPR1_RXR_SYNC_DEST_HOST:-a6000@10.10.10.2}
export ETPR1_R2R_SFT_SYNC_STATE_DIR=${ETPR1_RXR_SYNC_STATE_DIR:-${ETPR1_R2R_SFT_SYNC_SOURCE_DIR}/sync_to_eval}

exec "${SCRIPT_DIR}/manage_r2r_sft_checkpoint_sync_server.sh" "${1:-status}"
