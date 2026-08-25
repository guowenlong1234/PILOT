#!/usr/bin/env python3
"""Validate all 50 E24 checkpoints and atomically select SR+SPL champion."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


ITERATION_RE = re.compile(r"^ckpt\.iter([0-9]+)\.pth$")


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
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=1839)
    args = parser.parse_args()

    checkpoints = {}
    for path in args.checkpoint_dir.glob("ckpt.iter*.pth"):
        match = ITERATION_RE.match(path.name)
        if match:
            checkpoints[int(match.group(1))] = path.resolve()
    expected_iterations = list(range(200, args.expected * 200 + 1, 200))
    if sorted(checkpoints) != expected_iterations:
        raise ValueError(
            f"checkpoint iterations differ from contract: {sorted(checkpoints)}"
        )

    rows = []
    for iteration in expected_iterations:
        result_path = args.result_dir / f"stats_ckpt_{iteration}_val_unseen.json"
        diagnostic_path = (
            args.result_dir / f"lookahead_ckpt_{iteration}_val_unseen.json"
        )
        result = load_json(result_path)
        diagnostic = load_json(diagnostic_path)
        if int(diagnostic.get("episodes", -1)) != args.episodes:
            raise ValueError(f"iteration {iteration} has wrong episode count")
        if float(diagnostic.get("oracle_q1_calls", -1)) != 0.0:
            raise ValueError(f"iteration {iteration} called q1 Oracle")
        metrics = diagnostic.get("metrics", {})
        sr = float(result["success"])
        spl = float(result["spl"])
        checkpoint = checkpoints[iteration]
        rows.append(
            {
                "iteration": iteration,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "result_path": str(result_path.resolve()),
                "diagnostic_path": str(diagnostic_path.resolve()),
                "sr": sr,
                "spl": spl,
                "sr_plus_spl": sr + spl,
                "q0_record_coverage": float(metrics.get("q0_record_coverage", 0.0)),
                "q0_context_coverage": float(metrics.get("q0_context_coverage", 0.0)),
                "q0_success_rate": float(metrics.get("q0_success_rate", 0.0)),
                "q1_success_rate": float(metrics.get("q1_success_rate", 0.0)),
                "future_valid_rate": float(metrics.get("valid_rate", 0.0)),
                "elapsed_seconds": float(diagnostic["elapsed_seconds"]),
            }
        )

    best = max(
        rows,
        key=lambda row: (
            row["sr_plus_spl"], row["spl"], row["sr"], row["iteration"]
        ),
    )
    summary = {
        "format_version": "etpr1-e24-joint-selection-v1",
        "selection_rule": ["sr_plus_spl", "spl", "sr", "iteration"],
        "baseline": {"iteration": 14200, "sr": 0.637303, "spl": 0.556054},
        "checkpoint_count": len(rows),
        "rows": rows,
    }
    selection = {
        "format_version": summary["format_version"],
        "selection_rule": summary["selection_rule"],
        "best": best,
        "baseline": summary["baseline"],
        "delta_sr": best["sr"] - summary["baseline"]["sr"],
        "delta_spl": best["spl"] - summary["baseline"]["spl"],
        "delta_sr_plus_spl": best["sr_plus_spl"]
        - summary["baseline"]["sr"]
        - summary["baseline"]["spl"],
    }
    atomic_json(args.output_dir / "checkpoint_summary.json", summary)
    atomic_json(args.output_dir / "best_selection.json", selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
