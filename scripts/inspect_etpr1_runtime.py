#!/usr/bin/env python3

import ast
import importlib
import os
import platform
from pathlib import Path


def _import_or_none(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        return exc


def _read_version_from_file(version_file: Path):
    tree = ast.parse(version_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"__version__", "VERSION"}:
                return ast.literal_eval(node.value)
    raise ValueError(f"No version assignment found in {version_file}")


def _fallback_runtime_version(name: str):
    runtime_root = os.environ.get("ETPR1_RUNTIME_ROOT", "").strip()
    if not runtime_root:
        pythonpath = os.environ.get("PYTHONPATH", "")
        for entry in pythonpath.split(os.pathsep):
            path = Path(entry)
            if path.name == "site-packages" and path.parent.name == "prefix":
                runtime_root = str(path.parent)
                break
            if path.name == "habitat-baselines" and path.parent.name == "prefix":
                runtime_root = str(path.parent)
                break
    if not runtime_root:
        return None

    root = Path(runtime_root)
    version_files = {
        "habitat": [
            root / "site-packages" / "habitat" / "version.py",
            root / "site-packages" / "habitat" / "__init__.py",
        ],
        "habitat_sim": [
            root / "site-packages" / "habitat_sim" / "__init__.py",
        ],
        "habitat_baselines": [
            root / "habitat-baselines" / "habitat_baselines" / "version.py",
            root / "habitat-baselines" / "habitat_baselines" / "__init__.py",
        ],
    }.get(name)

    if version_files is None:
        return None

    for candidate in version_files:
        if not candidate.is_file():
            continue
        try:
            return _read_version_from_file(candidate)
        except Exception:
            continue
    return None


def _print_core_runtime():
    print(f"python={platform.python_version()}")

    torch = _import_or_none("torch")
    if isinstance(torch, Exception):
        print(f"torch=IMPORT_ERROR:{torch}")
        print("cuda=IMPORT_ERROR:torch unavailable")
        print("cuda_available=IMPORT_ERROR:torch unavailable")
    else:
        print(f"torch={getattr(torch, '__version__', 'missing')}")
        print(f"cuda={getattr(torch.version, 'cuda', 'missing')}")
        print(f"cuda_available={torch.cuda.is_available()}")

    transformers = _import_or_none("transformers")
    if isinstance(transformers, Exception):
        print(f"transformers=IMPORT_ERROR:{transformers}")
    else:
        print(f"transformers={getattr(transformers, '__version__', 'missing')}")


def main():
    print(f"ETPR1_RUNTIME_ACTIVE={os.environ.get('ETPR1_RUNTIME_ACTIVE', '')}")
    print(f"ETPR1_RUNTIME_ROOT={os.environ.get('ETPR1_RUNTIME_ROOT', '')}")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH', '')}")
    print(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}")
    _print_core_runtime()

    for name in ("habitat", "habitat_sim", "habitat_baselines"):
        imported = _import_or_none(name)
        if isinstance(imported, Exception):
            print(f"{name}=IMPORT_ERROR:{imported}")
            fallback_version = _fallback_runtime_version(name)
            if fallback_version is None:
                print(f"{name}_version=IMPORT_ERROR:{imported}")
            else:
                print(f"{name}_version={fallback_version}")
        else:
            module_file = getattr(imported, "__file__", None) or "builtin_module"
            module_version = getattr(imported, "__version__", "missing")
            print(f"{name}={module_file}")
            print(f"{name}_version={module_version}")


if __name__ == "__main__":
    main()
