#!/usr/bin/env python3
"""Validate RxR evaluation outputs and atomically select the best checkpoint."""

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


CHECKPOINT_RE = re.compile(r"^ckpt\.iter([0-9]+)\.pth$")
RESULT_RE = re.compile(r"^stats_ckpt_([0-9]+)_val_unseen\.json$")
FAILURE_METRICS = (
    "q0_batch_failures",
    "q0_row_failures",
    "cwp_batch_failures",
    "cwp_row_failures",
    "q1_batch_failures",
    "q1_row_failures",
)
LANGUAGES = ["en-US", "en-IN", "hi-IN", "te-IN"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as stream:
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


def finite_metric(payload, *names) -> float:
    for name in names:
        if name in payload:
            value = float(payload[name])
            if not math.isfinite(value):
                raise ValueError(f"metric {name} is not finite")
            return value
    raise ValueError(f"missing metric; expected one of {names}")


def checkpoint_iteration(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return None if match is None else int(match.group(1))


def result_episode_ids(result_dir: Path, iteration: int) -> set[str]:
    episode_ids = set()
    pattern = f"stats_ep_ckpt_{iteration}_val_unseen_r*_w*.json"
    paths = sorted(result_dir.glob(pattern))
    if not paths:
        raise ValueError(
            f"iteration {iteration} has no per-episode result files matching {pattern}"
        )
    for path in paths:
        payload = load_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"per-episode result must be an object: {path}")
        overlap = episode_ids.intersection(str(key) for key in payload)
        if overlap:
            raise ValueError(f"duplicate episode ids across rank results: {sorted(overlap)[:3]}")
        episode_ids.update(str(key) for key in payload)
    return episode_ids


def ids_sha256(ids) -> str:
    payload = "\n".join(sorted(str(value) for value in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def dataset_episode_count(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    episodes = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(episodes, list):
        raise ValueError(f"RxR dataset does not contain an episode list: {path}")
    return len(episodes)


def source_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_joint_diagnostics(path: Path, expected_episodes: int) -> dict:
    diagnostic = load_json(path)
    if int(diagnostic.get("episodes", -1)) != expected_episodes:
        raise ValueError(f"joint diagnostic has wrong episode count: {path}")
    if float(diagnostic.get("oracle_q1_calls", -1)) != 0.0:
        raise ValueError(f"joint evaluation queried the q1 simulator Oracle: {path}")
    metrics = diagnostic.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"joint diagnostic is missing metrics: {path}")
    for name, raw_value in metrics.items():
        if isinstance(raw_value, (int, float)) and not math.isfinite(float(raw_value)):
            raise ValueError(f"joint diagnostic metric {name} is not finite: {path}")
    failures = {name: float(metrics.get(name, 0.0)) for name in FAILURE_METRICS}
    if any(value != 0.0 for value in failures.values()):
        raise ValueError(f"joint diagnostic contains NWM/CWP failures: {failures}")
    return diagnostic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "joint"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--episode-manifest", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--total-iterations", type=int)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    args = parser.parse_args()

    expected_episodes = args.expected_episodes
    expected_episode_ids_sha = None
    if args.episode_manifest is not None:
        episode_manifest = load_json(args.episode_manifest)
        if episode_manifest.get("format_version") != "etpr1-rxr-eval-episodes-v1":
            raise ValueError("wrong RxR episode-manifest format")
        manifest_count = int(episode_manifest["episode_count"])
        if expected_episodes is not None and expected_episodes != manifest_count:
            raise ValueError("episode manifest conflicts with expected episode count")
        expected_episodes = manifest_count
        expected_episode_ids_sha = str(episode_manifest["episode_ids_sha256"])
    if args.dataset is not None:
        dataset_count = dataset_episode_count(args.dataset)
        if expected_episodes is not None and expected_episodes != dataset_count:
            raise ValueError(
                f"dataset/result episode contract differs: {dataset_count} != {expected_episodes}"
            )
        expected_episodes = dataset_count
    if expected_episodes is None or expected_episodes <= 1:
        raise ValueError("formal RxR selection requires more than one episode")

    total_iterations = args.total_iterations or (
        30000 if args.mode == "baseline" else 10000
    )
    if total_iterations < 1 or args.checkpoint_interval < 1:
        raise ValueError("iteration settings must be positive")
    expected_iterations = list(
        range(args.checkpoint_interval, total_iterations + 1, args.checkpoint_interval)
    )
    if not expected_iterations or expected_iterations[-1] != total_iterations:
        expected_iterations.append(total_iterations)

    checkpoints = {}
    for path in args.checkpoint_dir.glob("ckpt.iter*.pth"):
        iteration = checkpoint_iteration(path)
        if iteration is not None:
            checkpoints[iteration] = path.resolve()
    if not checkpoints:
        raise ValueError(f"no RxR checkpoints found in {args.checkpoint_dir}")
    if sorted(checkpoints) != expected_iterations:
        raise ValueError(
            "checkpoint iterations differ from the formal contract: "
            f"expected={expected_iterations} actual={sorted(checkpoints)}"
        )
    result_iterations = sorted(
        int(match.group(1))
        for path in args.result_dir.glob("stats_ckpt_*_val_unseen.json")
        if (match := RESULT_RE.fullmatch(path.name)) is not None
    )
    if result_iterations != expected_iterations:
        raise ValueError(
            "result iterations differ from the formal contract: "
            f"expected={expected_iterations} actual={result_iterations}"
        )

    rows = []
    for iteration, checkpoint in sorted(checkpoints.items()):
        result_path = args.result_dir / f"stats_ckpt_{iteration}_val_unseen.json"
        episode_ids = result_episode_ids(args.result_dir, iteration)
        actual_episodes = len(episode_ids)
        if actual_episodes != expected_episodes:
            raise ValueError(
                f"iteration {iteration} has {actual_episodes} episodes; expected {expected_episodes}"
            )
        if (
            expected_episode_ids_sha is not None
            and ids_sha256(episode_ids) != expected_episode_ids_sha
        ):
            raise ValueError(
                f"iteration {iteration} episode identities differ from the manifest"
            )
        result = load_json(result_path)
        row = {
            "iteration": iteration,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "result_path": str(result_path.resolve()),
            "episodes": actual_episodes,
            "sdtw": finite_metric(result, "sdtw", "SDTW"),
            "ndtw": finite_metric(result, "ndtw", "NDTW"),
            "spl": finite_metric(result, "spl", "SPL"),
            "sr": finite_metric(result, "success", "sr", "SR"),
        }
        if args.mode == "joint":
            diagnostic_path = (
                args.result_dir / f"lookahead_ckpt_{iteration}_val_unseen.json"
            )
            diagnostic = validate_joint_diagnostics(
                diagnostic_path, expected_episodes
            )
            row["diagnostic_path"] = str(diagnostic_path.resolve())
            row["lookahead_metrics"] = diagnostic["metrics"]
        rows.append(row)
    if not rows:
        raise ValueError("no checkpoints have complete RxR evaluation results")

    best = max(
        rows,
        key=lambda row: (
            row["sdtw"],
            row["ndtw"],
            row["spl"],
            row["sr"],
            row["iteration"],
        ),
    )
    root = Path(__file__).resolve().parents[1]
    commit = args.source_commit or source_commit(root)
    selection_rule = ["sdtw", "ndtw", "spl", "sr", "iteration"]
    format_version = (
        "etpr1-rxr-sft-base-selection-v1"
        if args.mode == "baseline"
        else "etpr1-rxr-native-cls-e24-joint-selection-v1"
    )
    common = {
        "format_version": format_version,
        "mode": args.mode,
        "task_type": "rxr",
        "dataset_role": "guide",
        "dataset_languages": LANGUAGES,
        "rgb_hfov": 63,
        "episode_count": expected_episodes,
        "selection_rule": selection_rule,
        "source_commit": commit,
    }
    summary = {**common, "rows": rows, "checkpoint_count": len(rows)}
    selection = {**common, "best": best}
    atomic_json(args.output_dir / "checkpoint_summary.json", summary)
    atomic_json(args.output_dir / "best_selection.json", selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
