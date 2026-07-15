import json
import subprocess

import pytest

from scripts.rae_smoke_source_identity import resolve_source_identity


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _clean_git_repo(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Smoke Test")
    _git(tmp_path, "config", "user.email", "smoke@example.com")
    source = tmp_path / "source.py"
    source.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", "source.py")
    _git(tmp_path, "commit", "-m", "initial")
    return source, _git(tmp_path, "rev-parse", "HEAD")


def test_git_checkout_rejects_modified_tracked_file(tmp_path):
    source, head = _clean_git_repo(tmp_path)
    source.write_text("modified\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dirty.*modified.*source.py"):
        resolve_source_identity(
            tmp_path,
            requested_commit=head,
            manifest_path=tmp_path / "manifest.sha256",
        )


def test_git_checkout_rejects_staged_change(tmp_path):
    source, head = _clean_git_repo(tmp_path)
    source.write_text("staged\n", encoding="utf-8")
    _git(tmp_path, "add", "source.py")

    with pytest.raises(ValueError, match="dirty.*staged.*source.py"):
        resolve_source_identity(
            tmp_path,
            requested_commit=head,
            manifest_path=tmp_path / "manifest.sha256",
        )


def test_git_checkout_rejects_untracked_file(tmp_path):
    _, head = _clean_git_repo(tmp_path)
    (tmp_path / "untracked.py").write_text("new\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dirty.*untracked.*untracked.py"):
        resolve_source_identity(
            tmp_path,
            requested_commit=head,
            manifest_path=tmp_path / "manifest.sha256",
        )


def test_git_checkout_ignores_its_own_manifest_output(tmp_path):
    _, head = _clean_git_repo(tmp_path)
    manifest = tmp_path / "manifest.sha256"

    first = resolve_source_identity(
        tmp_path,
        requested_commit=head,
        manifest_path=manifest,
    )
    second = resolve_source_identity(
        tmp_path,
        requested_commit=head,
        manifest_path=manifest,
    )

    assert first["identity"] == head
    assert second["identity"] == head


def test_git_checkout_rejects_requested_commit_that_is_not_head(tmp_path):
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Smoke Test")
    _git(tmp_path, "config", "user.email", "smoke@example.com")
    source = tmp_path / "source.py"
    source.write_text("first\n", encoding="utf-8")
    _git(tmp_path, "add", "source.py")
    _git(tmp_path, "commit", "-m", "first")
    first = _git(tmp_path, "rev-parse", "HEAD")
    source.write_text("second\n", encoding="utf-8")
    _git(tmp_path, "commit", "-am", "second")

    with pytest.raises(ValueError, match="does not match git HEAD"):
        resolve_source_identity(
            tmp_path,
            requested_commit=first,
            manifest_path=tmp_path / "manifest.sha256",
        )


def test_non_git_source_uses_deterministic_controlled_manifest(tmp_path):
    source = tmp_path / "scripts" / "run.py"
    source.parent.mkdir()
    source.write_text("print('stable')\n", encoding="utf-8")
    excluded = tmp_path / "data" / "logs" / "huge.log"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("first ignored payload\n", encoding="utf-8")
    task_log = tmp_path / ".task2_logs" / "runtime.log"
    task_log.parent.mkdir()
    task_log.write_text("first task log\n", encoding="utf-8")
    bert_weight = tmp_path / "bert_config" / "model" / "pytorch_model.bin"
    bert_weight.parent.mkdir(parents=True)
    bert_weight.write_bytes(b"first model asset")

    manifest = tmp_path / "source.manifest"
    first = resolve_source_identity(
        tmp_path,
        requested_commit="unverifiable-remote-label",
        manifest_path=manifest,
    )
    excluded.write_text("second ignored payload\n", encoding="utf-8")
    task_log.write_text("second task log\n", encoding="utf-8")
    bert_weight.write_bytes(b"second model asset")
    second = resolve_source_identity(
        tmp_path,
        requested_commit="another-label",
        manifest_path=manifest,
    )

    assert first["kind"] == "manifest"
    assert first["identity"] == second["identity"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert "scripts/run.py" in manifest.read_text()
    assert "data/logs/huge.log" not in manifest.read_text()
    assert ".task2_logs/runtime.log" not in manifest.read_text()
    assert "bert_config/model/pytorch_model.bin" not in manifest.read_text()


def test_smoke_script_persists_stage_statuses_and_final_summary():
    text = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts"
        / "smoke_rae_dino.sh"
    ).read_text(encoding="utf-8")

    assert "source_identity.json" in text
    assert "source_manifest.sha256" in text
    assert "summary.log" in text
    assert "status=PASS exit=0" in text
    assert "status=FAIL exit=" in text
    assert "ALL_PASS" in text
