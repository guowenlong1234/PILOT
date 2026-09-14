"""Offline orchestration tests for streaming ghost checkpoints."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def stream(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "stream_ghost_checkpoints_under_test",
        ROOT / "scripts/stream_ghost_checkpoints.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingRoot:
    """Keep the production workspace guard while redirecting files to pytest."""

    def __init__(self, temporary_root):
        self.temporary_root = temporary_root

    def __str__(self):
        return "/home/gwl/project/etpr1/ETP-R1"

    def __truediv__(self, path):
        return self.temporary_root / path


def checkpoint_path(tmp_path, output, iteration):
    return (
        tmp_path
        / output
        / "train/ghost_concat_v1_joint_persistent_train/checkpoints"
        / "ghost_concat_v1_joint_persistent_train"
        / f"ckpt.iter{iteration}.pth"
    )


def invoke(monkeypatch, stream, tmp_path, arguments, remote_python):
    sync_calls = []
    saved = []
    monkeypatch.setattr(stream, "ROOT", TrainingRoot(tmp_path))
    monkeypatch.setattr(stream, "remote_python", remote_python)
    monkeypatch.setattr(
        stream.subprocess,
        "run",
        lambda command, **kwargs: sync_calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(stream, "save", lambda path, value: saved.append((path, value)))
    monkeypatch.setattr(sys, "argv", ["stream_ghost_checkpoints.py", *arguments])
    stream.main()
    return sync_calls, saved


def test_persistent_subset_uses_matching_paths_and_releases_only_replicas(
    tmp_path, monkeypatch, stream
):
    output = "data/logs/persistent_eval"
    requested = [400, 1000]
    originals = []
    for iteration in requested:
        source = checkpoint_path(tmp_path, output, iteration)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(f"checkpoint-{iteration}".encode())
        originals.append(source)

    manifest_reads = {}
    manifest_paths = []
    cleanup_calls = []

    def remote_python(code, *args):
        if "hashlib" in code:
            cleanup_calls.append(args)
            return "verified_replica_released\n"
        manifest = Path(args[0])
        manifest_paths.append(manifest)
        iteration = int(re.search(r"iter(\d+)", str(manifest)).group(1))
        read_count = manifest_reads.get(iteration, 0)
        manifest_reads[iteration] = read_count + 1
        if read_count == 0:
            return "{}\n"
        return json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": stream.sha(
                    checkpoint_path(tmp_path, output, iteration)
                ),
                "validation": {"episodes": 1839},
            }
        )

    sync_calls, saved = invoke(
        monkeypatch,
        stream,
        tmp_path,
        [
            "--output",
            output,
            "--memory-mode",
            "persistent_node_state",
            "--iterations",
            "400,1000",
        ],
        remote_python,
    )

    assert len(sync_calls) == 2
    assert [Path(call[0][call[0].index("--source") + 1]).name for call in sync_calls] == [
        "ckpt.iter400.pth",
        "ckpt.iter1000.pth",
    ]
    assert all(call[1]["check"] is True for call in sync_calls)
    assert all("ghost_concat_v1_joint_persistent_train" in call[0][-1] for call in sync_calls)
    assert {
        path.parent.name for path in manifest_paths
    } == {
        "ghost_concat_v1_joint_persistent_eval_iter400",
        "ghost_concat_v1_joint_persistent_eval_iter1000",
    }
    assert [Path(args[0]).name for args in cleanup_calls] == [
        "ckpt.iter400.pth",
        "ckpt.iter1000.pth",
    ]
    assert all(str(stream.REMOTE) in str(args[0]) for args in cleanup_calls)
    assert all(source.is_file() for source in originals)
    assert [record[1]["requested_iterations"] for record in saved] == [
        requested,
        requested,
    ]
    assert [record[1]["status"] for record in saved] == ["running", "completed"]


def test_bad_evaluation_digest_refuses_cleanup_and_delivery_completion(
    tmp_path, monkeypatch, stream
):
    output = "data/logs/persistent_eval"
    source = checkpoint_path(tmp_path, output, 400)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"trusted checkpoint")
    cleanup_calls = []

    def remote_python(code, *args):
        if "hashlib" in code:
            cleanup_calls.append(args)
            return "verified_replica_released\n"
        return json.dumps(
            {
                "status": "completed",
                "checkpoint_sha256": "0" * 64,
                "validation": {"episodes": 1839},
            }
        )

    with pytest.raises(RuntimeError, match="digest or full episode count mismatch"):
        invoke(
            monkeypatch,
            stream,
            tmp_path,
            [
                "--output",
                output,
                "--memory-mode",
                "persistent_node_state",
                "--iterations",
                "400",
            ],
            remote_python,
        )

    assert cleanup_calls == []
    assert source.is_file()


@pytest.mark.parametrize(
    "iterations", ["400,200", "200,200", "0", "201", "10001", "x"]
)
def test_invalid_explicit_subset_is_rejected(
    tmp_path, monkeypatch, stream, iterations
):
    monkeypatch.setattr(stream, "ROOT", TrainingRoot(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stream_ghost_checkpoints.py",
            "--output",
            "data/logs/persistent_eval",
            "--iterations",
            iterations,
        ],
    )

    with pytest.raises(SystemExit) as exc:
        stream.main()

    assert exc.value.code == 2
