import json
from types import SimpleNamespace

import pytest

from scripts.stress_pretrain_dataloader import (
    build_parser,
    dataloader_worker_pids,
    load_configs,
    read_process_status,
    validate_args,
)


def test_stress_parser_defaults_match_original_crashing_path():
    args = build_parser().parse_args([])

    assert args.batch_size == 16
    assert args.workers == 1
    assert args.pin_memory is True
    assert args.start_method == "fork"
    assert args.accum_steps == 8


def test_stress_parser_accepts_safe_single_process_path():
    args = build_parser().parse_args(
        ["--workers", "0", "--no-pin-memory", "--micro-batches", "10"]
    )

    validate_args(args)
    assert args.workers == 0
    assert args.pin_memory is False
    assert args.micro_batches == 10


def test_stress_parser_accepts_bounded_two_worker_spawn_path():
    args = build_parser().parse_args(
        [
            "--workers",
            "2",
            "--start-method",
            "spawn",
            "--no-pin-memory",
            "--prefetch-factor",
            "1",
            "--feature-cache-size-mb",
            "256",
        ]
    )

    validate_args(args)
    assert args.workers == 2
    assert args.start_method == "spawn"
    assert args.pin_memory is False
    assert args.prefetch_factor == 1
    assert args.feature_cache_size_mb == 256
    assert args.lazy_annotations is True


def test_validate_args_rejects_unused_start_method():
    args = build_parser().parse_args(
        ["--workers", "0", "--start-method", "spawn"]
    )

    with pytest.raises(ValueError, match="only applies"):
        validate_args(args)


def test_load_configs_reads_both_json_files(tmp_path):
    config_path = tmp_path / "config.json"
    model_config_path = tmp_path / "model.json"
    config_path.write_text(json.dumps({"seed": 7}), encoding="utf-8")
    model_config_path.write_text(json.dumps({"hidden_size": 768}), encoding="utf-8")

    config, model_config = load_configs(config_path, model_config_path)

    assert config == {"seed": 7}
    assert model_config == {"hidden_size": 768}


def test_read_process_status_reports_current_process():
    status = read_process_status("self")

    assert int(status["Threads"]) >= 1
    assert status["VmRSS"].endswith("kB")


def test_dataloader_worker_pids_uses_live_meta_loader_iterators():
    worker_one = SimpleNamespace(pid=101)
    worker_two = SimpleNamespace(pid=202)
    meta_loader = SimpleNamespace(
        name2iter={
            "mlm": SimpleNamespace(_workers=[worker_one]),
            "sap": SimpleNamespace(_workers=[worker_two, worker_one]),
        }
    )

    assert dataloader_worker_pids(meta_loader) == [101, 202]
