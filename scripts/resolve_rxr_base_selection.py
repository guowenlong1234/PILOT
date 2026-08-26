#!/usr/bin/env python3
"""Validate an RxR SFT base-selection manifest for joint training."""

import argparse
import hashlib
import json
from pathlib import Path


LANGUAGES = ["en-US", "en-IN", "hi-IN", "te-IN"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve(strict=True)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != "etpr1-rxr-sft-base-selection-v1":
        raise ValueError("wrong RxR base-selection format")
    if manifest.get("task_type") != "rxr":
        raise ValueError("RxR base-selection manifest has the wrong task")
    if manifest.get("dataset_role") != "guide":
        raise ValueError("RxR base-selection manifest must use guide data")
    if manifest.get("dataset_languages") != LANGUAGES:
        raise ValueError("RxR base-selection manifest has the wrong languages")
    if int(manifest.get("rgb_hfov", -1)) != 63:
        raise ValueError("RxR base-selection manifest has the wrong HFOV")
    smoke_only = bool(manifest.get("smoke_only", False))
    if smoke_only and not args.allow_smoke:
        raise ValueError("smoke-only RxR base cannot start formal joint training")
    best = manifest.get("best")
    if not isinstance(best, dict):
        raise ValueError("RxR base-selection manifest is missing best")
    checkpoint = (
        args.checkpoint
        if args.checkpoint is not None
        else Path(str(best.get("checkpoint_path", "")))
    ).resolve(strict=True)
    expected_sha = str(best.get("checkpoint_sha256", ""))
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError(
            f"RxR base checkpoint SHA256 mismatch: {actual_sha} != {expected_sha}"
        )
    iteration = int(best.get("iteration", -1))
    if iteration < 1:
        raise ValueError("RxR base-selection iteration must be positive")
    fields = (checkpoint, expected_sha, iteration, sha256_file(manifest_path))
    for field in fields:
        value = str(field)
        if "\n" in value or "\r" in value:
            raise ValueError("selection fields cannot contain newlines")
        print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
