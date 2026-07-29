from datetime import datetime, timezone

import pytest
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
    SCALARS,
)
from tensorboardX import SummaryWriter

from pretrain_src.pretrain_src.utils.logger import TensorboardLogger
from scripts.normalize_pretrain_tensorboard import (
    append_new_records,
    build_initial_records,
    event_file_timestamp,
    is_relative_tag,
    validation_tag_with_dataset,
    write_initial_records,
)


class _ScalarRecorder:
    def __init__(self):
        self.values = []

    def add_scalar(self, tag, value, step):
        self.values.append((tag, value, step))


def test_tensorboard_logger_can_resume_at_global_step():
    recorder = _ScalarRecorder()
    logger = TensorboardLogger()
    logger._logger = recorder

    logger.set_step(117500)
    logger.step()
    logger.log_scalar_dict({"loss/mlm": 0.25})

    assert logger.global_step == 117501
    assert recorder.values == [("loss/mlm", 0.25, 117501)]


def test_tensorboard_logger_rejects_negative_step():
    with pytest.raises(ValueError, match="non-negative"):
        TensorboardLogger().set_step(-1)


def test_relative_tag_classification():
    assert is_relative_tag("loss/mlm")
    assert is_relative_tag("valid_unseen_mlm/val_unseen_mlm_acc")
    assert not is_relative_tag("lr")
    assert not is_relative_tag("grad_norm")


def test_shared_validation_tags_are_split_in_arrival_order():
    occurrences = {}
    tag = "valid_unseen_mlm/val_unseen_mlm_acc"

    assert validation_tag_with_dataset(tag, 180000, occurrences) == (
        "valid_r2r_unseen_mlm/val_unseen_mlm_acc"
    )
    assert validation_tag_with_dataset(tag, 180000, occurrences) == (
        "valid_rxr_unseen_mlm/val_unseen_mlm_acc"
    )
    assert validation_tag_with_dataset(tag, 182500, occurrences) == (
        "valid_r2r_unseen_mlm/val_unseen_mlm_acc"
    )
    assert validation_tag_with_dataset("loss/mlm", 180000, occurrences) == (
        "loss/mlm"
    )


def test_normalizer_keeps_both_validation_datasets(tmp_path):
    source = tmp_path / "source"
    supervisor = tmp_path / "supervisor"
    source.mkdir()
    supervisor.mkdir()

    writer = SummaryWriter(str(source))
    tag = "valid_unseen_mlm/val_unseen_mlm_acc"
    writer.add_scalar(tag, 0.80, 0, walltime=1000)
    writer.add_scalar(tag, 0.90, 0, walltime=1001)
    writer.close()

    event_path = next(source.glob("events.out.tfevents.*"))
    started_at = datetime.fromtimestamp(
        event_file_timestamp(event_path) - 1,
        tz=timezone.utc,
    ).isoformat()
    (supervisor / "resume.log").write_text(
        "\n".join(
            [
                f"run_started_at={started_at}",
                (
                    "Resumed complete training state from train_state_117500.pt "
                    "at global step 117500"
                ),
            ]
        ),
        encoding="utf-8",
    )

    _sources, records = build_initial_records(source, supervisor)

    assert records[
        ("valid_r2r_unseen_mlm/val_unseen_mlm_acc", 117500)
    ].value == pytest.approx(0.80)
    assert records[
        ("valid_rxr_unseen_mlm/val_unseen_mlm_acc", 117500)
    ].value == pytest.approx(0.90)


def test_normalizer_offsets_old_relative_steps_but_keeps_global_steps(tmp_path):
    source = tmp_path / "source"
    supervisor = tmp_path / "supervisor"
    output = tmp_path / "normalized"
    source.mkdir()
    supervisor.mkdir()

    writer = SummaryWriter(str(source))
    writer.add_scalar("loss/mlm", 0.5, 0, walltime=1000)
    writer.add_scalar("loss/mlm", 0.4, 1, walltime=1001)
    writer.add_scalar("lr", 1e-4, 117500, walltime=1000)
    writer.add_scalar("lr", 9e-5, 117501, walltime=1001)
    writer.close()

    event_path = next(source.glob("events.out.tfevents.*"))
    started_at = datetime.fromtimestamp(
        event_file_timestamp(event_path) - 1,
        tz=timezone.utc,
    ).isoformat()
    (supervisor / "resume.log").write_text(
        "\n".join(
            [
                f"run_started_at={started_at}",
                "mode=resume",
                (
                    "Resumed complete training state from train_state_117500.pt "
                    "at global step 117500"
                ),
            ]
        ),
        encoding="utf-8",
    )

    _sources, records = build_initial_records(source, supervisor)
    normalized_writer = SummaryWriter(str(output))
    write_initial_records(normalized_writer, records)
    normalized_writer.close()

    normalized_path = next(output.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(
        str(normalized_path),
        size_guidance={SCALARS: 0},
        purge_orphaned_data=False,
    )
    accumulator.Reload()

    assert [item.step for item in accumulator.Scalars("loss/mlm")] == [
        117500,
        117501,
    ]
    assert [item.step for item in accumulator.Scalars("lr")] == [
        117500,
        117501,
    ]


def test_normalizer_reads_only_new_records_after_initial_scan(tmp_path):
    source = tmp_path / "source"
    supervisor = tmp_path / "supervisor"
    output = tmp_path / "normalized"
    source.mkdir()
    supervisor.mkdir()

    source_writer = SummaryWriter(str(source))
    source_writer.add_scalar("loss/mlm", 0.5, 0, walltime=1000)
    source_writer.add_scalar("loss/mlm", 0.4, 1, walltime=1001)
    source_writer.flush()

    event_path = next(source.glob("events.out.tfevents.*"))
    started_at = datetime.fromtimestamp(
        event_file_timestamp(event_path) - 1,
        tz=timezone.utc,
    ).isoformat()
    (supervisor / "resume.log").write_text(
        "\n".join(
            [
                f"run_started_at={started_at}",
                (
                    "Resumed complete training state from train_state_117500.pt "
                    "at global step 117500"
                ),
            ]
        ),
        encoding="utf-8",
    )

    sources, records = build_initial_records(source, supervisor)
    normalized_writer = SummaryWriter(str(output))
    maximum_steps = write_initial_records(normalized_writer, records)

    source_writer.add_scalar("loss/mlm", 0.3, 2, walltime=1002)
    source_writer.add_scalar("loss/mlm", 0.2, 3, walltime=1003)
    source_writer.flush()
    appended = append_new_records(
        normalized_writer,
        source,
        supervisor,
        sources,
        maximum_steps,
    )
    assert appended > 0
    maximum_after_append = dict(maximum_steps)
    assert append_new_records(
        normalized_writer,
        source,
        supervisor,
        sources,
        maximum_steps,
    ) == 0
    assert maximum_steps == maximum_after_append
    source_writer.close()
    normalized_writer.close()
