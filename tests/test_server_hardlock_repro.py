import argparse

import pytest

from scripts.reproduce_server_hardlock import (
    build_parser,
    diskstats_snapshot,
    nonnegative_int,
    positive_int,
    validate_args,
)


def parse(*arguments):
    return build_parser().parse_args(arguments)


def test_gpu_defaults_do_not_enable_checkpoint_io():
    args = parse("--mode", "gpu")

    assert args.io_dir is None
    assert args.io_gib == 0.0
    assert args.duration_sec == 300


def test_checkpoint_io_requires_both_size_and_directory(tmp_path):
    missing_directory = parse(
        "--mode",
        "distributed",
        "--io-gib",
        "4.5",
    )
    with pytest.raises(ValueError, match="--io-dir is required"):
        validate_args(missing_directory)

    missing_size = parse(
        "--mode",
        "distributed",
        "--io-dir",
        str(tmp_path),
    )
    with pytest.raises(ValueError, match="--io-gib must be nonzero"):
        validate_args(missing_size)


@pytest.mark.parametrize("mode", ("monitor", "cpu-memory"))
def test_non_cuda_modes_reject_checkpoint_io(tmp_path, mode):
    args = parse(
        "--mode",
        mode,
        "--io-dir",
        str(tmp_path),
        "--io-gib",
        "1",
    )

    with pytest.raises(ValueError, match="gpu or distributed"):
        validate_args(args)


def test_positive_int_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_nonnegative_int_accepts_zero_and_rejects_negative():
    assert nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_int("-1")


def test_distributed_mode_rejects_explicit_gpu_index():
    args = parse("--mode", "distributed", "--gpu-index", "1")

    with pytest.raises(ValueError, match="LOCAL_RANK"):
        validate_args(args)


def test_diskstats_selects_whole_nvme_devices(monkeypatch):
    contents = "\n".join(
        (
            "259 0 nvme0n1 1 2 3 4",
            "259 1 nvme0n1p1 5 6 7 8",
            "259 2 nvme1n1 9 10 11 12",
            "8 0 sda 13 14 15 16",
        )
    )
    monkeypatch.setattr(
        "scripts.reproduce_server_hardlock.read_text",
        lambda path: contents,
    )

    assert diskstats_snapshot() == [
        "259 0 nvme0n1 1 2 3 4",
        "259 2 nvme1n1 9 10 11 12",
    ]
