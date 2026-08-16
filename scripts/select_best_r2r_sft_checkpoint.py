#!/usr/bin/env python3
"""Validate complete SFT evaluations and select the best SR+SPL checkpoint."""

import argparse
import json
import math
import re
from pathlib import Path


RESULT_RE = re.compile(r"stats_ckpt_([0-9]+)_val_unseen[.]json$")


def parse_candidate(value):
    parts = value.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "candidate must be LABEL=RESULT_DIR=CHECKPOINT_DIR"
        )
    return tuple(parts)


def expected_iterations(total_iterations, interval):
    values = list(range(interval, total_iterations + 1, interval))
    if not values or values[-1] != total_iterations:
        values.append(total_iterations)
    return values


def finite_metric(metrics, key, path):
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: {key} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{path}: {key} must be finite")
    return value


def load_run(label, result_dir, checkpoint_dir, expected):
    result_dir = Path(result_dir)
    checkpoint_dir = Path(checkpoint_dir)
    found = {}
    for path in result_dir.glob("stats_ckpt_*_val_unseen.json"):
        match = RESULT_RE.fullmatch(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if iteration in found:
            raise ValueError(f"{label}: duplicate result for iteration {iteration}")
        found[iteration] = path

    expected_set = set(expected)
    found_set = set(found)
    missing = sorted(expected_set - found_set)
    extra = sorted(found_set - expected_set)
    if missing or extra:
        raise ValueError(
            f"{label}: incomplete result set; missing={missing}, extra={extra}"
        )

    rows = []
    for iteration in expected:
        result_path = found[iteration]
        with result_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        if not isinstance(metrics, dict):
            raise ValueError(f"{result_path}: expected a JSON object")
        success = finite_metric(metrics, "success", result_path)
        spl = finite_metric(metrics, "spl", result_path)
        checkpoint = checkpoint_dir / f"ckpt.iter{iteration}.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise ValueError(f"{label}: checkpoint missing or empty: {checkpoint}")
        rows.append(
            {
                "run": label,
                "iteration": iteration,
                "success": success,
                "spl": spl,
                "score": success + spl,
                "result_path": str(result_path.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_bytes": checkpoint.stat().st_size,
            }
        )
    return rows


def select_best(candidates, total_iterations, interval):
    expected = expected_iterations(total_iterations, interval)
    rows = []
    for label, result_dir, checkpoint_dir in candidates:
        rows.extend(load_run(label, result_dir, checkpoint_dir, expected))
    if not rows:
        raise ValueError("no candidates were provided")
    # Stable tie-break: SR+SPL, then SPL, SR, iteration, and run label.
    best = max(
        rows,
        key=lambda row: (
            row["score"],
            row["spl"],
            row["success"],
            row["iteration"],
            row["run"],
        ),
    )
    return {
        "selection_metric": "success+spl",
        "tie_break": ["spl", "success", "iteration", "run"],
        "total_iterations": total_iterations,
        "checkpoint_interval": interval,
        "expected_results_per_run": len(expected),
        "validated_runs": len(candidates),
        "validated_results": len(rows),
        **best,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        metavar="LABEL=RESULT_DIR=CHECKPOINT_DIR",
    )
    parser.add_argument("--total-iterations", type=int, default=15000)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    args = parser.parse_args(argv)
    if args.total_iterations <= 0 or args.checkpoint_interval <= 0:
        parser.error("iteration settings must be positive")
    try:
        selected = select_best(
            args.candidate, args.total_iterations, args.checkpoint_interval
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.exit(1, f"selection_failed: {error}\n")
    print(json.dumps(selected, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
