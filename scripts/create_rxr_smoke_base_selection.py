#!/usr/bin/env python3
"""Create a clearly test-only RxR base manifest from a smoke checkpoint."""

import argparse
import json
from pathlib import Path
import subprocess

from select_best_rxr_sft_checkpoint import atomic_json, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    if args.iteration < 1 or checkpoint.name != f"ckpt.iter{args.iteration}.pth":
        raise ValueError("smoke checkpoint filename and iteration differ")
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    payload = {
        "format_version": "etpr1-rxr-sft-base-selection-v1",
        "mode": "baseline",
        "smoke_only": True,
        "task_type": "rxr",
        "dataset_role": "guide",
        "dataset_languages": ["en-US", "en-IN", "hi-IN", "te-IN"],
        "rgb_hfov": 63,
        "source_commit": commit,
        "best": {
            "iteration": args.iteration,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "metrics": None,
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload["best"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
