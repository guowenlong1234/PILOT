#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
CHECKPOINT_DIR=${ETPR1_E24_TRAIN_CHECKPOINT_DIR:-${REPO_ROOT}/data/logs/active_lookahead/e24_joint_sft/checkpoints/etpr1_e24_joint_sft}
REMOTE_SELECTION=${ETPR1_E24_REMOTE_SELECTION:-/home/a6000/gwl/ETP-R1/data/logs/active_lookahead/e24_joint_eval/selection/best_selection.json}
OUTPUT_DIR=${ETPR1_E24_SELECTION_DIR:-${REPO_ROOT}/data/logs/active_lookahead/e24_joint_sft/selection}

mkdir -p "$OUTPUT_DIR"
temporary=${OUTPUT_DIR}/best_selection.json.tmp
ssh a6000@10.10.10.2 "test -s '$REMOTE_SELECTION' && cat '$REMOTE_SELECTION'" >"$temporary"
python -m json.tool "$temporary" >/dev/null
mv -f -- "$temporary" "$OUTPUT_DIR/best_selection.json"

read -r iteration expected_sha < <(
    python - "$OUTPUT_DIR/best_selection.json" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as stream:
    best = json.load(stream)["best"]
print(int(best["iteration"]), best["checkpoint_sha256"])
PY
)
source_checkpoint=${CHECKPOINT_DIR}/ckpt.iter${iteration}.pth
[ -s "$source_checkpoint" ] || { echo "Missing selected checkpoint: $source_checkpoint" >&2; exit 1; }
actual_sha=$(sha256sum -- "$source_checkpoint" | awk '{print $1}')
[ "$actual_sha" = "$expected_sha" ] || {
    echo "Selected checkpoint SHA256 mismatch: expected=$expected_sha actual=$actual_sha" >&2
    exit 1
}

best_dir=${CHECKPOINT_DIR}/best
best_link=${best_dir}/ckpt.best.iter${iteration}.pth
mkdir -p "$best_dir"
if [ -e "$best_link" ]; then
    [ "$(stat -c %i -- "$best_link")" = "$(stat -c %i -- "$source_checkpoint")" ] || {
        echo "Existing best path is not the selected hard link: $best_link" >&2
        exit 1
    }
else
    ln -- "$source_checkpoint" "$best_link"
fi
printf 'best_iteration=%s\nbest_sha256=%s\nbest_hardlink=%s\n' \
    "$iteration" "$actual_sha" "$best_link"
