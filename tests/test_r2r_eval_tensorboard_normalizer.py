import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "normalize_r2r_eval_tensorboard.py"
)
SPEC = importlib.util.spec_from_file_location("eval_tb_normalizer", MODULE_PATH)
normalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(normalizer)


class FakeWriter:
    def __init__(self):
        self.values = []
        self.flushes = 0

    def add_scalar(self, tag, value, step):
        self.values.append((tag, value, step))

    def flush(self):
        self.flushes += 1


def write_result(root, iteration, **metrics):
    path = root / f"stats_ckpt_{iteration}_val_unseen.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_discover_and_append_results_are_strictly_iteration_ordered(tmp_path):
    write_result(tmp_path, 600, spl=0.6, success=0.7)
    write_result(tmp_path, 200, spl=0.2, success=0.3)
    write_result(tmp_path, 400, spl=0.4, success=0.5)

    records = normalizer.discover_results(tmp_path, "val_unseen")
    writer = FakeWriter()
    appended, maximum = normalizer.append_results(
        writer, records, {}, "val_unseen"
    )

    assert [iteration for iteration, _metrics in records] == [200, 400, 600]
    assert [step for tag, _value, step in writer.values if tag == "eval_spl/val_unseen"] == [200, 400, 600]
    assert appended == 6
    assert maximum == 600
    assert writer.flushes == 1


def test_restart_only_appends_steps_newer_for_each_tag(tmp_path):
    write_result(tmp_path, 200, spl=0.2, success=0.3)
    write_result(tmp_path, 400, spl=0.4, success=0.5)
    records = normalizer.discover_results(tmp_path, "val_unseen")
    writer = FakeWriter()
    maximum_steps = {
        "eval_spl/val_unseen": 400,
        "eval_success/val_unseen": 200,
    }

    appended, _maximum = normalizer.append_results(
        writer, records, maximum_steps, "val_unseen"
    )

    assert writer.values == [("eval_success/val_unseen", 0.5, 400)]
    assert appended == 1


def test_existing_event_file_restores_real_maximum_steps(tmp_path):
    writer = normalizer.SummaryWriter(str(tmp_path))
    writer.add_scalar("eval_spl/val_unseen", 0.2, 200)
    writer.add_scalar("eval_spl/val_unseen", 0.6, 600)
    writer.close()

    assert normalizer.load_maximum_steps(tmp_path) == {
        "eval_spl/val_unseen": 600
    }


@pytest.mark.parametrize("value", ["missing_separator", "bad/name=/tmp"])
def test_run_spec_rejects_ambiguous_names(value):
    with pytest.raises(Exception):
        normalizer.parse_run_spec(value)
