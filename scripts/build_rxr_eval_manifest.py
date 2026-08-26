#!/usr/bin/env python3
"""Build the effective RxR evaluation episode identity manifest."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path


def load(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def ids_sha256(ids) -> str:
    payload = "\n".join(sorted(str(value) for value in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    dataset_payload = load(args.dataset)
    episodes = (
        dataset_payload.get("episodes")
        if isinstance(dataset_payload, dict)
        else dataset_payload
    )
    if not isinstance(episodes, list):
        raise ValueError("RxR dataset does not contain an episode list")
    dataset_ids = {
        str(episode["episode_id"])
        for episode in episodes
        if isinstance(episode, dict) and "episode_id" in episode
    }
    gt_payload = load(args.gt)
    if not isinstance(gt_payload, dict):
        raise ValueError("RxR NDTW ground truth must be an object")
    effective_ids = sorted(dataset_ids.intersection(str(key) for key in gt_payload))
    if not effective_ids:
        raise ValueError("effective RxR evaluation episode set is empty")
    config_sha = hashlib.sha256(args.config.read_bytes()).hexdigest()
    payload = {
        "format_version": "etpr1-rxr-eval-episodes-v1",
        "task_type": "rxr",
        "split": "val_unseen",
        "dataset_role": "guide",
        "dataset_languages": ["en-US", "en-IN", "hi-IN", "te-IN"],
        "dataset_path": str(args.dataset.resolve()),
        "gt_path": str(args.gt.resolve()),
        "episode_count": len(effective_ids),
        "episode_ids_sha256": ids_sha256(effective_ids),
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha,
        "source_commit": args.source_commit,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
