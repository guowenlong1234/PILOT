#!/usr/bin/env python3

"""Mirror pretraining TensorBoard scalars with resume-aware global steps."""

import argparse
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    SCALARS,
)
from tensorboard.backend.event_processing.event_file_loader import (
    LegacyEventFileLoader,
)
from tensorboardX import SummaryWriter


EVENT_TIME_PATTERN = re.compile(r"events\.out\.tfevents\.(\d+)\.")
RESUME_STEP_PATTERN = re.compile(
    r"Resumed complete training state .* at global step (\d+)"
)
RUN_START_PREFIX = "run_started_at="
RELATIVE_TAG_PREFIXES = ("loss/", "valid")
SHARED_VALIDATION_TAG_PREFIX = "valid_unseen_"
VALIDATION_DATASETS = ("r2r", "rxr")


@dataclass(frozen=True)
class ScalarRecord:
    wall_time: float
    step: int
    value: float


class IncrementalScalarSource:
    """Read each event record once while keeping the file cursor open."""

    def __init__(self, event_path, offset):
        self.event_path = Path(event_path)
        self.offset = offset
        self.loader = LegacyEventFileLoader(str(self.event_path))
        self.relative_tags = {}
        self.validation_occurrences = {}

    def load_new_records(self):
        records = {}
        for event in self.loader.Load():
            if not event.HasField("summary"):
                continue
            for item in event.summary.value:
                if item.WhichOneof("value") != "simple_value":
                    continue
                relative = self.relative_tags.get(item.tag)
                if relative is None:
                    relative = (
                        self.offset > 0
                        and is_relative_tag(item.tag)
                        and event.step < self.offset
                    )
                    self.relative_tags[item.tag] = relative
                step = event.step + self.offset if relative else event.step
                tag = validation_tag_with_dataset(
                    item.tag,
                    step,
                    self.validation_occurrences,
                )
                records.setdefault(tag, []).append(
                    ScalarRecord(
                        wall_time=event.wall_time,
                        step=step,
                        value=item.simple_value,
                    )
                )
        return records


def parse_iso_timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def discover_supervisor_starts(supervisor_dir):
    starts = []
    for path in Path(supervisor_dir).glob("*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        started_at = None
        for line in text.splitlines():
            if line.startswith(RUN_START_PREFIX):
                started_at = parse_iso_timestamp(
                    line[len(RUN_START_PREFIX):].strip()
                )
                break
        if started_at is None:
            continue
        match = RESUME_STEP_PATTERN.search(text)
        starts.append((started_at, int(match.group(1)) if match else 0, path))
    return sorted(starts)


def event_file_timestamp(path):
    match = EVENT_TIME_PATTERN.search(Path(path).name)
    if not match:
        raise ValueError(f"Cannot read creation time from event file: {path}")
    return float(match.group(1))


def match_resume_offset(event_path, supervisor_starts, max_gap_seconds=300):
    event_time = event_file_timestamp(event_path)
    candidates = [
        item for item in supervisor_starts
        if 0 <= event_time - item[0] <= max_gap_seconds
    ]
    if not candidates:
        raise ValueError(
            f"No supervisor start matches TensorBoard event file {event_path}"
        )
    _, offset, _ = max(candidates, key=lambda item: item[0])
    return offset


def is_relative_tag(tag):
    return tag.startswith(RELATIVE_TAG_PREFIXES)


def validation_tag_with_dataset(tag, step, occurrences):
    if not tag.startswith(SHARED_VALIDATION_TAG_PREFIX):
        return tag
    key = (tag, step)
    occurrence = occurrences.get(key, 0)
    occurrences[key] = occurrence + 1
    dataset = VALIDATION_DATASETS[occurrence % len(VALIDATION_DATASETS)]
    return tag.replace("valid_", f"valid_{dataset}_", 1)


def load_scalar_records(event_path, offset):
    accumulator = EventAccumulator(
        str(event_path),
        size_guidance={SCALARS: 0},
        purge_orphaned_data=False,
    )
    accumulator.Reload()
    records = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = accumulator.Scalars(tag)
        if not values:
            continue
        old_relative_steps = (
            offset > 0
            and is_relative_tag(tag)
            and min(value.step for value in values) < offset
        )
        records[tag] = [
            ScalarRecord(
                wall_time=value.wall_time,
                step=value.step + offset if old_relative_steps else value.step,
                value=value.value,
            )
            for value in values
        ]
    return records


def build_initial_records(source_logdir, supervisor_dir):
    supervisor_starts = discover_supervisor_starts(supervisor_dir)
    if not supervisor_starts:
        raise ValueError(f"No supervisor starts found in {supervisor_dir}")

    sources = {}
    deduplicated = {}
    for event_path in sorted(Path(source_logdir).glob("events.out.tfevents.*")):
        offset = match_resume_offset(event_path, supervisor_starts)
        source = IncrementalScalarSource(event_path, offset)
        records = source.load_new_records()
        sources[event_path] = source
        for tag, values in records.items():
            for value in values:
                key = (tag, value.step)
                current = deduplicated.get(key)
                if current is None or value.wall_time >= current.wall_time:
                    deduplicated[key] = value
    return sources, deduplicated


def write_initial_records(writer, records):
    maximum_steps = {}
    ordered = sorted(
        (
            (tag, record)
            for (tag, _step), record in records.items()
        ),
        key=lambda item: (
            item[1].step,
            item[1].wall_time,
            item[0],
        ),
    )
    for tag, record in ordered:
        writer.add_scalar(
            tag,
            record.value,
            record.step,
            walltime=record.wall_time,
        )
        maximum_steps[tag] = max(
            maximum_steps.get(tag, -1),
            record.step,
        )
    writer.flush()
    return maximum_steps


def append_new_records(
    writer,
    source_logdir,
    supervisor_dir,
    sources,
    maximum_steps,
):
    supervisor_starts = discover_supervisor_starts(supervisor_dir)
    appended = 0
    for event_path in sorted(Path(source_logdir).glob("events.out.tfevents.*")):
        if event_path not in sources:
            sources[event_path] = IncrementalScalarSource(
                event_path,
                match_resume_offset(event_path, supervisor_starts),
            )
        source = sources[event_path]
        records = source.load_new_records()
        for tag, values in records.items():
            for record in values:
                if record.step <= maximum_steps.get(tag, -1):
                    continue
                writer.add_scalar(
                    tag,
                    record.value,
                    record.step,
                    walltime=record.wall_time,
                )
                maximum_steps[tag] = record.step
                appended += 1
    if appended:
        writer.flush()
    return appended


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-logdir", required=True)
    parser.add_argument("--supervisor-dir", required=True)
    parser.add_argument("--output-logdir", required=True)
    parser.add_argument("--reload-interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.reload_interval <= 0:
        raise ValueError("--reload-interval must be positive")

    output_logdir = Path(args.output_logdir)
    output_logdir.mkdir(parents=True, exist_ok=False)
    sources, records = build_initial_records(
        args.source_logdir,
        args.supervisor_dir,
    )
    writer = SummaryWriter(str(output_logdir))
    maximum_steps = write_initial_records(writer, records)
    print(
        f"normalized_scalars={len(records)} "
        f"source_files={len(sources)} "
        f"output={output_logdir}",
        flush=True,
    )
    if args.once:
        writer.close()
        return

    try:
        while True:
            time.sleep(args.reload_interval)
            appended = append_new_records(
                writer,
                args.source_logdir,
                args.supervisor_dir,
                sources,
                maximum_steps,
            )
            if appended:
                print(
                    f"appended_scalars={appended} "
                    f"max_step={max(maximum_steps.values())}",
                    flush=True,
                )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
