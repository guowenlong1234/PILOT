import importlib.util
import subprocess
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "vlnce_baselines"
    / "common"
    / "checkpoint_sync.py"
)
SPEC = importlib.util.spec_from_file_location("checkpoint_sync", MODULE_PATH)
checkpoint_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checkpoint_sync)


def test_local_checkpoint_sync_publishes_atomically(tmp_path):
    source = tmp_path / "source" / "ckpt.iter200.pth"
    source.parent.mkdir()
    source.write_bytes(b"complete checkpoint")
    destination = tmp_path / "destination"

    published = checkpoint_sync.sync_checkpoint(source, destination)

    assert published == destination / source.name
    assert published.read_bytes() == source.read_bytes()
    assert list((destination / ".incoming").iterdir()) == []


def test_launch_checkpoint_sync_is_detached_and_nonblocking(
    tmp_path, monkeypatch
):
    source = tmp_path / "ckpt.iter200.pth"
    source.write_bytes(b"checkpoint")
    captured = {}

    class FakeProcess:
        pid = 31415

    def fake_popen(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        return FakeProcess()

    monkeypatch.setattr(checkpoint_sync.subprocess, "Popen", fake_popen)

    pid = checkpoint_sync.launch_checkpoint_sync(
        source,
        "a6000@10.10.10.2:/checkpoints",
    )

    assert pid == 31415
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert "a6000@10.10.10.2:/checkpoints" in captured["command"]


@pytest.mark.parametrize(
    ("destination", "expected"),
    (
        ("/tmp/checkpoints", None),
        (
            "a6000@10.10.10.2:/data/checkpoints",
            ("a6000@10.10.10.2", "/data/checkpoints"),
        ),
        (
            "gwl@10.10.10.1:/data/checkpoints",
            ("gwl@10.10.10.1", "/data/checkpoints"),
        ),
    ),
)
def test_sync_destination_supports_both_direct_link_directions(
    destination, expected
):
    assert checkpoint_sync._remote_destination(destination) == expected


def test_remote_checkpoint_sync_uses_incoming_then_atomic_rename(
    tmp_path, monkeypatch
):
    source = tmp_path / "ckpt.iter400.pth"
    source.write_bytes(b"123456")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if "stat -c %s" in command[-1]:
            if kwargs.get("check") is False:
                return subprocess.CompletedProcess(command, 1, "", "")
            return subprocess.CompletedProcess(command, 0, "6\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(checkpoint_sync.subprocess, "run", fake_run)

    published = checkpoint_sync.sync_checkpoint(
        source,
        "a6000@10.10.10.2:/data/checkpoints",
    )

    assert published.endswith(":/data/checkpoints/ckpt.iter400.pth")
    commands = [call[0] for call in calls]
    rsync = next(command for command in commands if command[0] == "rsync")
    assert "/.incoming/ckpt.iter400.pth.part." in rsync[-1]
    rename = commands[-1][-1]
    assert rename.startswith("mv -f -- ")
    assert "/.incoming/" in rename
    assert rename.endswith("/data/checkpoints/ckpt.iter400.pth")
