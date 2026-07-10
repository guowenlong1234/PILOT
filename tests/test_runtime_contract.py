import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_LIB = ROOT / ".runtime" / "etpr1_habitat" / "prefix" / "lib"
NVIDIA_LIBRARY_DIRS = (
    "/usr/local/nvidia/lib",
    "/usr/local/nvidia/lib64",
)


def test_runtime_wrapper_uses_etpr1_owned_paths():
    script = (ROOT / "scripts" / "etpr1_rae_runtime_exec.sh").read_text()
    pythonpath_export = next(
        line for line in script.splitlines() if line.startswith("export PYTHONPATH=")
    )

    assert ".runtime/etpr1_habitat" in script
    assert "vendor/legacy_clip" in script
    assert "ETPR1_RUNTIME_ACTIVE=1" in script
    assert "${REPO_ROOT}" in pythonpath_export
    assert "ETPNav" not in pythonpath_export
    assert "_deps" not in pythonpath_export
    assert "dino_cwp" not in pythonpath_export
    assert "/ETPNav" not in script


def test_habitat_builder_uses_etpr1_runtime_root():
    script = (ROOT / "scripts" / "build_etpr1_habitat.sh").read_text()

    assert ".runtime/etpr1_habitat" in script
    assert "ETPR1_HABITAT_LAB_SOURCE" in script
    assert "ETPR1_HABITAT_SIM_SOURCE" in script


def test_runtime_wrapper_preserves_system_egl_environment_contract():
    script = (ROOT / "scripts" / "etpr1_rae_runtime_exec.sh").read_text()

    assert (
        "ETPR1_SYSTEM_GL_DISPATCH:-/lib/x86_64-linux-gnu/libGLdispatch.so.0"
        in script
    )
    assert "LD_LIBRARY_PATH" in script
    assert "LD_PRELOAD" in script


def _runtime_environment(
    *,
    ld_preload: str,
    system_gl_dispatch: str | None = None,
    initial_ld_library_path: str = ":".join(NVIDIA_LIBRARY_DIRS),
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("ETPR1_SYSTEM_GL_DISPATCH", None)
    environment.update(
        {
            "LD_LIBRARY_PATH": initial_ld_library_path,
            "LD_PRELOAD": ld_preload,
        }
    )
    if system_gl_dispatch is not None:
        environment["ETPR1_SYSTEM_GL_DISPATCH"] = system_gl_dispatch
    result = _run_runtime_wrapper(environment)
    assert "cannot be preloaded" not in result.stderr.lower(), result.stderr
    result.check_returncode()
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def _run_runtime_wrapper(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(ROOT / "scripts" / "etpr1_rae_runtime_exec.sh"), "env"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _expected_ld_library_path() -> str:
    return ":".join((str(RUNTIME_LIB), *NVIDIA_LIBRARY_DIRS))


def test_runtime_wrapper_prepends_gl_dispatch_and_preserves_nvidia_libraries():
    system_gl_dispatch = "/lib/x86_64-linux-gnu/libGLdispatch.so.0"
    existing_preload = "/lib/x86_64-linux-gnu/libm.so.6"

    environment = _runtime_environment(ld_preload=existing_preload)

    assert environment["LD_PRELOAD"] == f"{system_gl_dispatch} {existing_preload}"
    assert environment["LD_LIBRARY_PATH"] == _expected_ld_library_path()


def test_runtime_wrapper_deduplicates_nested_runtime_library_path():
    initial = ":".join(
        (
            str(RUNTIME_LIB),
            NVIDIA_LIBRARY_DIRS[1],
            str(RUNTIME_LIB),
            NVIDIA_LIBRARY_DIRS[0],
        )
    )

    environment = _runtime_environment(
        ld_preload="/lib/x86_64-linux-gnu/libm.so.6",
        initial_ld_library_path=initial,
    )

    assert environment["LD_LIBRARY_PATH"] == _expected_ld_library_path()
    assert environment["LD_LIBRARY_PATH"].split(":").count(str(RUNTIME_LIB)) == 1


@pytest.mark.parametrize(
    "forbidden_entry",
    (
        "/home/a6000/gwl/ETPNav/lib",
        "/home/a6000/gwl/dino_cwp/lib",
        "/home/a6000/gwl/_deps/lib",
        "/tmp/external-lib",
    ),
)
def test_runtime_wrapper_rejects_inherited_external_library_path(forbidden_entry):
    environment = os.environ.copy()
    environment.update(
        {
            "LD_LIBRARY_PATH": ":".join(
                (NVIDIA_LIBRARY_DIRS[0], forbidden_entry, NVIDIA_LIBRARY_DIRS[1])
            ),
            "LD_PRELOAD": "/lib/x86_64-linux-gnu/libm.so.6",
        }
    )

    result = _run_runtime_wrapper(environment)

    assert result.returncode != 0
    assert "inherited LD_LIBRARY_PATH" in result.stderr
    assert "forbidden" in result.stderr.lower()
    assert forbidden_entry in result.stderr
    assert "ETPR1_RUNTIME_ACTIVE=1" not in result.stdout


def test_runtime_wrapper_rejects_empty_inherited_library_path_entry():
    environment = os.environ.copy()
    environment.update(
        {
            "LD_LIBRARY_PATH": f"{NVIDIA_LIBRARY_DIRS[0]}::{NVIDIA_LIBRARY_DIRS[1]}",
            "LD_PRELOAD": "/lib/x86_64-linux-gnu/libm.so.6",
        }
    )

    result = _run_runtime_wrapper(environment)

    assert result.returncode != 0
    assert "empty inherited LD_LIBRARY_PATH entry" in result.stderr
    assert "ETPR1_RUNTIME_ACTIVE=1" not in result.stdout


def test_runtime_wrapper_does_not_duplicate_existing_gl_dispatch_preload():
    system_gl_dispatch = "/lib/x86_64-linux-gnu/libGLdispatch.so.0"
    existing_preload = "/lib/x86_64-linux-gnu/libm.so.6"

    environment = _runtime_environment(
        ld_preload=f"{system_gl_dispatch} {existing_preload}"
    )

    assert environment["LD_PRELOAD"] == f"{system_gl_dispatch} {existing_preload}"


def test_runtime_wrapper_does_not_duplicate_colon_separated_gl_dispatch():
    system_gl_dispatch = "/lib/x86_64-linux-gnu/libGLdispatch.so.0"
    existing_preload = "/lib/x86_64-linux-gnu/libm.so.6"
    original_preload = f"{system_gl_dispatch}:{existing_preload}"

    environment = _runtime_environment(ld_preload=original_preload)

    assert environment["LD_PRELOAD"] == original_preload
    assert environment["LD_PRELOAD"].split(":").count(system_gl_dispatch) == 1


def test_runtime_wrapper_honors_custom_gl_dispatch_without_duplication():
    custom_gl_dispatch = "/lib/x86_64-linux-gnu/libm.so.6"
    existing_preload = "/lib/x86_64-linux-gnu/libGLdispatch.so.0"
    original_preload = f"{existing_preload}:{custom_gl_dispatch}"

    environment = _runtime_environment(
        ld_preload=original_preload,
        system_gl_dispatch=custom_gl_dispatch,
    )

    assert environment["LD_PRELOAD"] == original_preload
    assert environment["LD_PRELOAD"].split(":").count(custom_gl_dispatch) == 1


def test_runtime_wrapper_prepends_custom_gl_dispatch_override():
    custom_gl_dispatch = "/lib/x86_64-linux-gnu/libm.so.6"
    existing_preload = "/lib/x86_64-linux-gnu/libGLdispatch.so.0"

    environment = _runtime_environment(
        ld_preload=existing_preload,
        system_gl_dispatch=custom_gl_dispatch,
    )

    assert environment["LD_PRELOAD"] == f"{custom_gl_dispatch} {existing_preload}"


def test_runtime_wrapper_does_not_treat_gl_dispatch_path_prefix_as_same_token(
    tmp_path,
):
    system_gl_dispatch = "/lib/x86_64-linux-gnu/libGLdispatch.so.0"
    similar_preload = tmp_path / "libGLdispatch.so.0.backup"
    similar_preload.symlink_to("/lib/x86_64-linux-gnu/libm.so.6")

    environment = _runtime_environment(ld_preload=str(similar_preload))

    assert environment["LD_PRELOAD"] == f"{system_gl_dispatch} {similar_preload}"
