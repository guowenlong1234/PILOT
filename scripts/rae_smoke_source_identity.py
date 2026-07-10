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
    manifest_sha256, file_count = _write_manifest(project_root, manifest_path)
    if _is_git_checkout(project_root):
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
        return {
            "kind": "git",
            "identity": head,
            "git_head": head,
            "requested_commit": requested_commit,
            "manifest_sha256": manifest_sha256,
            "manifest_file_count": file_count,
        }
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
