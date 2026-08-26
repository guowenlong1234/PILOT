#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
MODE=${1:?Usage: finalize_rxr_sft_selection_host.sh <baseline|joint>}
CONTAINER=${ETPR1_RXR_EVAL_CONTAINER:-gwl-etpr1-rae}
DATASET=${ETPR1_RXR_EVAL_DATASET:-${REPO_ROOT}/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide_90.json.gz}
GT=${ETPR1_RXR_EVAL_GT:-${REPO_ROOT}/data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide_gt.json.gz}

case "$MODE" in
    baseline)
        CONFIG=${REPO_ROOT}/run_rxr/iter_train_rae_dino_sft.yaml
        CHECKPOINT_DIR=${ETPR1_RXR_BASE_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/checkpoints/etpr1_rxr_rae_dino_sft}
        RESULT_DIR=${ETPR1_RXR_BASE_EVAL_RESULT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/eval_watch_val_unseen/results/etpr1_rxr_rae_dino_sft_eval_watch/eval_results}
        OUTPUT_DIR=${ETPR1_RXR_BASE_SELECTION_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/selection}
        TOTAL_ITERATIONS=30000
        ;;
    joint)
        CONFIG=${REPO_ROOT}/run_rxr/iter_train_rae_dino_native_cls_e24_joint.yaml
        CHECKPOINT_DIR=${ETPR1_RXR_JOINT_EVAL_CKPT_DIR:-${REPO_ROOT}/data/logs/active_lookahead/rxr_native_cls_e24_joint_sft/checkpoints/etpr1_rxr_native_cls_e24_joint_sft}
        RESULT_DIR=${ETPR1_RXR_JOINT_EVAL_RESULT_DIR:-${REPO_ROOT}/data/logs/active_lookahead/rxr_native_cls_e24_joint_eval/results/etpr1_rxr_native_cls_e24_joint_eval_watch/eval_results}
        OUTPUT_DIR=${ETPR1_RXR_JOINT_SELECTION_DIR:-${REPO_ROOT}/data/logs/active_lookahead/rxr_native_cls_e24_joint_eval/selection}
        TOTAL_ITERATIONS=10000
        ;;
    *) echo "mode must be baseline or joint" >&2; exit 2 ;;
esac

for path in "$DATASET" "$GT" "$CONFIG" "$CHECKPOINT_DIR" "$RESULT_DIR"; do
    [ -e "$path" ] || { echo "Missing RxR selection dependency: $path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_DIR"
SOURCE_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD)

docker exec \
    -e MODE="$MODE" -e DATASET="$DATASET" -e GT="$GT" -e CONFIG="$CONFIG" \
    -e CHECKPOINT_DIR="$CHECKPOINT_DIR" -e RESULT_DIR="$RESULT_DIR" \
    -e OUTPUT_DIR="$OUTPUT_DIR" -e TOTAL_ITERATIONS="$TOTAL_ITERATIONS" \
    -e SOURCE_COMMIT="$SOURCE_COMMIT" "$CONTAINER" bash -lc '
        source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh
        conda activate etpr1_rae
        cd /home/a6000/gwl/ETP-R1
        python scripts/build_rxr_eval_manifest.py \
            --dataset "$DATASET" --gt "$GT" --config "$CONFIG" \
            --output "$OUTPUT_DIR/eval_manifest.json" \
            --source-commit "$SOURCE_COMMIT"
        python scripts/select_best_rxr_sft_checkpoint.py \
            --mode "$MODE" --checkpoint-dir "$CHECKPOINT_DIR" \
            --result-dir "$RESULT_DIR" --output-dir "$OUTPUT_DIR" \
            --episode-manifest "$OUTPUT_DIR/eval_manifest.json" \
            --total-iterations "$TOTAL_ITERATIONS" \
            --checkpoint-interval 200 --source-commit "$SOURCE_COMMIT"
    '
