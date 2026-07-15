#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


EXCLUDED_TOP_LEVEL = {
    ".git",
    ".runtime",
    ".task2_logs",
    "data",
    "pretrained",
}
EXCLUDED_PREFIXES = {
    ("pretrain_src", "datasets"),
    ("pretrain_src", "img_features"),
}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache"}


def _git(project_root, *args):
    return subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _is_git_checkout(project_root):
    try:
        return _git(project_root, "rev-parse", "--is-inside-work-tree") == "true"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _git_paths(project_root, *args):
    result = subprocess.run(
        ["git", "-C", str(project_root), *args, "-z"],
        check=True,
        capture_output=True,
    )
    return [
        item.decode("utf-8", errors="surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    ]


def _validate_clean_git_checkout(project_root, manifest_path):
    modified = _git_paths(project_root, "diff", "--name-only")
    staged = _git_paths(project_root, "diff", "--cached", "--name-only")
    untracked = _git_paths(
        project_root,
        "ls-files",
        "--others",
        "--exclude-standard",
    )
    try:
        manifest_relative = manifest_path.resolve().relative_to(project_root)
    except ValueError:
        manifest_relative = None
    if manifest_relative is not None:
        manifest_name = manifest_relative.as_posix()
        untracked = [path for path in untracked if path != manifest_name]
    if modified or staged or untracked:
        raise ValueError(
            "dirty git worktree cannot be identified as HEAD; "
            f"modified={modified}, staged={staged}, untracked={untracked}"
        )


def _controlled_source_files(project_root, manifest_path):
    manifest_path = manifest_path.resolve()
    for path in sorted(project_root.rglob("*")):
        if not path.is_file() or path.resolve() == manifest_path:
            continue
        relative = path.relative_to(project_root)
        parts = relative.parts
        if not parts or parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if any(tuple(parts[: len(prefix)]) == prefix for prefix in EXCLUDED_PREFIXES):
            continue
        if any(part in EXCLUDED_PARTS for part in parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if parts[0] == "bert_config" and path.suffix == ".bin":
            continue
        if path.name in {"source_manifest.sha256", "source_identity.json"}:
            continue
        yield relative, path


def _write_manifest(project_root, manifest_path):
    lines = []
    for relative, path in _controlled_source_files(project_root, manifest_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative.as_posix()}\n")
    payload = "".join(lines).encode("utf-8")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(lines)


def resolve_source_identity(
    project_root,
    requested_commit=None,
    manifest_path=None,
):
    project_root = Path(project_root).resolve()
    manifest_path = Path(manifest_path or project_root / "source_manifest.sha256")
    if _is_git_checkout(project_root):
        _validate_clean_git_checkout(project_root, manifest_path)
        head = _git(project_root, "rev-parse", "HEAD")
        if requested_commit:
            try:
                requested = _git(
                    project_root,
                    "rev-parse",
                    "--verify",
                    f"{requested_commit}^{{commit}}",
                )
            except subprocess.CalledProcessError as error:
                raise ValueError(
                    f"ETPR1_SOURCE_COMMIT is not a valid commit: {requested_commit}"
                ) from error
            if requested != head:
                raise ValueError(
                    "ETPR1_SOURCE_COMMIT does not match git HEAD: "
                    f"requested={requested}, head={head}"
                )
        manifest_sha256, file_count = _write_manifest(
            project_root,
            manifest_path,
        )
        return {
            "kind": "git",
            "identity": head,
            "git_head": head,
            "requested_commit": requested_commit,
            "manifest_sha256": manifest_sha256,
            "manifest_file_count": file_count,
        }
    manifest_sha256, file_count = _write_manifest(project_root, manifest_path)
    return {
        "kind": "manifest",
        "identity": f"tree-{manifest_sha256}",
        "git_head": None,
        "requested_commit": requested_commit,
        "manifest_sha256": manifest_sha256,
        "manifest_file_count": file_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--requested-commit")
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--identity-output", required=True)
    args = parser.parse_args()
    identity = resolve_source_identity(
        args.project_root,
        requested_commit=args.requested_commit,
        manifest_path=args.manifest_output,
    )
    Path(args.identity_output).write_text(
        json.dumps(identity, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(identity, sort_keys=True))


if __name__ == "__main__":
    main()
