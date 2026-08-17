#!/usr/bin/env python3
"""Build monotonic TensorBoard runs from per-checkpoint eval JSON files."""

import argparse
import json
import math
import re
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)
from tensorboardX import SummaryWriter


RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TAG_PREFIX_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_run_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use NAME=RESULT_DIR")
    name, result_dir = value.split("=", 1)
    if not RUN_NAME_RE.fullmatch(name):
        raise argparse.ArgumentTypeError(
            "run name may contain only letters, digits, dot, underscore, and dash"
        )
    if not result_dir:
        raise argparse.ArgumentTypeError("result directory must not be empty")
    return name, Path(result_dir)


def discover_results(result_dir, split):
    pattern = re.compile(rf"^stats_ckpt_(\d+)_{re.escape(split)}\.json$")
    records = []
    for path in Path(result_dir).glob(f"stats_ckpt_*_{split}.json"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"result_not_ready path={path} error={exc}", flush=True)
            continue
        if not isinstance(metrics, dict):
            raise ValueError(f"Evaluation result must be an object: {path}")
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Metric {key!r} is not numeric in {path}")
            if not math.isfinite(value):
                raise ValueError(f"Metric {key!r} is not finite in {path}")
            numeric_metrics[key] = float(value)
        records.append((int(match.group(1)), numeric_metrics))
    records.sort(key=lambda item: item[0])
    return records


def load_maximum_steps(logdir):
    logdir = Path(logdir)
    if not logdir.exists() or not any(logdir.glob("events.out.tfevents.*")):
        return {}
    accumulator = EventAccumulator(
        str(logdir), size_guidance={"scalars": 0}
    ).Reload()
    maximum_steps = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)
        if values:
            maximum_steps[tag] = max(value.step for value in values)
    return maximum_steps


def metric_tag(key, split, tag_prefix="eval"):
    return f"{tag_prefix}_{key}/{split}"


def append_results(
    writer, records, maximum_steps, split, tag_prefix="eval"
):
    appended = 0
    maximum_iteration = -1
    for iteration, metrics in records:
        maximum_iteration = max(maximum_iteration, iteration)
        for key in sorted(metrics):
            tag = metric_tag(key, split, tag_prefix)
            if iteration <= maximum_steps.get(tag, -1):
                continue
            writer.add_scalar(tag, metrics[key], iteration)
            maximum_steps[tag] = iteration
            appended += 1
    if appended:
        writer.flush()
    return appended, maximum_iteration


def has_new_results(records, maximum_steps, split, tag_prefix="eval"):
    return any(
        iteration
        > maximum_steps.get(metric_tag(key, split, tag_prefix), -1)
        for iteration, metrics in records
        for key in metrics
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_spec,
        metavar="NAME=RESULT_DIR",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--tag-prefix", default="eval")
    parser.add_argument("--reload-interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.reload_interval <= 0:
        raise ValueError("--reload-interval must be positive")
    if not TAG_PREFIX_RE.fullmatch(args.tag_prefix):
        raise ValueError(
            "tag prefix may contain only letters, digits, dot, "
            "underscore, and dash"
        )
    names = [name for name, _path in args.run]
    if len(names) != len(set(names)):
        raise ValueError("run names must be unique")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runs = []
    for name, result_dir in args.run:
        if not result_dir.is_dir():
            raise ValueError(f"Result directory does not exist: {result_dir}")
        output_logdir = output_root / name
        output_logdir.mkdir(parents=True, exist_ok=True)
        runs.append((name, result_dir, output_logdir))

    while True:
        for name, result_dir, output_logdir in runs:
            records = discover_results(result_dir, args.split)
            maximum_steps = load_maximum_steps(output_logdir)
            if not has_new_results(
                records, maximum_steps, args.split, args.tag_prefix
            ):
                continue
            writer = SummaryWriter(str(output_logdir))
            try:
                appended, maximum_iteration = append_results(
                    writer,
                    records,
                    maximum_steps,
                    args.split,
                    args.tag_prefix,
                )
            finally:
                writer.close()
            print(
                f"run={name} appended_scalars={appended} "
                f"max_iteration={maximum_iteration}",
                flush=True,
            )
        if args.once:
            return
        time.sleep(args.reload_interval)


if __name__ == "__main__":
    main()
