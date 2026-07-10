from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
