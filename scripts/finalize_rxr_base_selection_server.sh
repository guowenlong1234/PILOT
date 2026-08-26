#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REMOTE_SELECTION=${ETPR1_RXR_REMOTE_BASE_SELECTION:-/home/a6000/gwl/ETP-R1/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/selection/best_selection.json}
CHECKPOINT_DIR=${ETPR1_RXR_BASE_CHECKPOINT_DIR:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/checkpoints/etpr1_rxr_rae_dino_sft}
OUTPUT=${ETPR1_RXR_LOCAL_BASE_SELECTION:-${REPO_ROOT}/data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_formal/selection/best_selection.json}

mkdir -p "$(dirname -- "$OUTPUT")"
temporary=${OUTPUT}.tmp.$$
trap 'rm -f -- "$temporary"' EXIT
ssh -n a6000@10.10.10.2 "test -s '$REMOTE_SELECTION' && cat '$REMOTE_SELECTION'" >"$temporary"

python - "$temporary" "$CHECKPOINT_DIR" "$OUTPUT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

source, checkpoint_dir, output = map(Path, sys.argv[1:])
with source.open("r", encoding="utf-8") as stream:
    payload = json.load(stream)
if payload.get("format_version") != "etpr1-rxr-sft-base-selection-v1":
    raise ValueError("remote selection has the wrong format")
best = payload["best"]
checkpoint = (checkpoint_dir / f"ckpt.iter{int(best['iteration'])}.pth").resolve(strict=True)
digest = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != best["checkpoint_sha256"]:
    raise ValueError("local RxR checkpoint differs from the evaluated checkpoint")
best["checkpoint_path"] = str(checkpoint)
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
with temporary.open("w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, output)
print(f"selection={output} checkpoint={checkpoint} sha256={digest.hexdigest()}")
PY
