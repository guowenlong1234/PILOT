#!/usr/bin/env python3

import importlib
import os
import platform
from pathlib import Path
import sys


EXPECTED_HABITAT_VERSION = "0.3.3"
FORBIDDEN_PATH_PARTS = ("/ETPNav", "/dino_cwp", "/_deps")


def _has_forbidden_path(value: str) -> bool:
    return any(part in value for part in FORBIDDEN_PATH_PARTS)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _import_module(name: str, errors):
    try:
        return importlib.import_module(name)
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{exc}")
        errors.append(f"{name} import failed: {exc}")
        return None


def _record_core_runtime(errors):
    print(f"python={platform.python_version()}")

    torch = _import_module("torch", errors)
    if torch is None:
        print("torch_version=IMPORT_ERROR")
        print("cuda=IMPORT_ERROR:torch unavailable")
        print("cuda_available=IMPORT_ERROR:torch unavailable")
    else:
        torch_version = getattr(torch, "__version__", "missing")
        print(f"torch={torch_version}")
        print(f"cuda={getattr(torch.version, 'cuda', 'missing')}")
        print(f"cuda_available={torch.cuda.is_available()}")

    transformers = _import_module("transformers", errors)
    if transformers is None:
        print("transformers_version=IMPORT_ERROR")
    else:
        print(f"transformers={getattr(transformers, '__version__', 'missing')}")


def _record_owned_module(name: str, runtime_root: Path, errors):
    module = _import_module(name, errors)
    if module is None:
        print(f"{name}_version=IMPORT_ERROR")
        return

    module_file = getattr(module, "__file__", None)
    module_version = str(getattr(module, "__version__", "missing"))
    print(f"{name}={module_file or 'builtin_module'}")
    print(f"{name}_version={module_version}")

    if module_version != EXPECTED_HABITAT_VERSION:
        errors.append(
            f"{name} version is {module_version}, expected {EXPECTED_HABITAT_VERSION}"
        )
    if not module_file:
        errors.append(f"{name} has no filesystem path")
        return

    resolved = Path(module_file).resolve()
    if _has_forbidden_path(str(resolved)):
        errors.append(f"{name} uses forbidden path: {resolved}")
    if not _is_within(resolved, runtime_root):
        errors.append(f"{name} escapes runtime root {runtime_root}: {resolved}")


def main():
    errors = []
    runtime_root_text = os.environ.get("ETPR1_RUNTIME_ROOT", "").strip()
    runtime_prefix_text = os.environ.get("ETPR1_RUNTIME_PREFIX", "").strip()
    pythonpath = os.environ.get("PYTHONPATH", "")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")

    print(f"ETPR1_RUNTIME_ACTIVE={os.environ.get('ETPR1_RUNTIME_ACTIVE', '')}")
    print(f"ETPR1_RUNTIME_ROOT={runtime_root_text}")
    print(f"ETPR1_RUNTIME_PREFIX={runtime_prefix_text}")
    print(f"PYTHONPATH={pythonpath}")
    print(f"LD_LIBRARY_PATH={ld_library_path}")

    if os.environ.get("ETPR1_RUNTIME_ACTIVE") != "1":
        errors.append("ETPR1_RUNTIME_ACTIVE is not 1")
    if not runtime_root_text:
        errors.append("ETPR1_RUNTIME_ROOT is empty")
        runtime_root = Path("/__missing_etpr1_runtime__")
    else:
        runtime_root = Path(runtime_root_text).resolve()
    if not runtime_prefix_text:
        errors.append("ETPR1_RUNTIME_PREFIX is empty")
    else:
        runtime_prefix = Path(runtime_prefix_text).resolve()
        if not _is_within(runtime_prefix, runtime_root):
            errors.append(
                f"runtime prefix escapes runtime root {runtime_root}: {runtime_prefix}"
            )

    for label, value in (
        ("ETPR1_RUNTIME_ROOT", runtime_root_text),
        ("ETPR1_RUNTIME_PREFIX", runtime_prefix_text),
        ("PYTHONPATH", pythonpath),
        ("LD_LIBRARY_PATH", ld_library_path),
    ):
        if _has_forbidden_path(value):
            errors.append(f"{label} contains forbidden path: {value}")

    _record_core_runtime(errors)
    for name in ("habitat", "habitat_sim", "habitat_baselines"):
        _record_owned_module(name, runtime_root, errors)

    if errors:
        for error in errors:
            print(f"runtime_error={error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
