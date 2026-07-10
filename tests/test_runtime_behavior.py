import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_etpr1_habitat.sh"
WRAPPER = ROOT / "scripts" / "etpr1_rae_runtime_exec.sh"
INSPECTOR = ROOT / "scripts" / "inspect_etpr1_runtime.py"
RUNTIME_ROOT = ROOT / ".runtime" / "etpr1_habitat"
RUNTIME_PREFIX = RUNTIME_ROOT / "prefix"
HABITAT_LAB_SOURCE = Path("/home/a6000/gwl/_deps/habitat-lab-v0.3.3")
HABITAT_SIM_SOURCE = Path("/home/a6000/gwl/_deps/habitat-sim-v0.3.3")


def _run(command, *, env=None, cwd=ROOT):
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def _runtime_env():
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(RUNTIME_ROOT)
    env["ETPR1_RUNTIME_PREFIX"] = str(RUNTIME_PREFIX)
    return env


def test_wrapper_rejects_incomplete_runtime(tmp_path):
    runtime_root = tmp_path / "runtime"
    legacy_clip_root = tmp_path / "vendor" / "legacy_clip"
    runtime_root.mkdir()
    (legacy_clip_root / "clip").mkdir(parents=True)
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_LEGACY_CLIP_ROOT"] = str(legacy_clip_root)

    result = _run([WRAPPER, sys.executable, "-c", "print('unexpected')"], env=env)

    assert result.returncode != 0
    assert "runtime" in result.stdout.lower()


def test_wrapper_rejects_runtime_symlink_escape(tmp_path):
    runtime_root = tmp_path / "runtime"
    site_packages = runtime_root / "prefix" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "habitat").symlink_to(
        RUNTIME_PREFIX / "site-packages" / "habitat",
        target_is_directory=True,
    )
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)

    result = _run([WRAPPER, sys.executable, "-c", "print('unexpected')"], env=env)

    assert result.returncode != 0
    assert "escapes ETP-R1 owner" in result.stdout


def test_wrapper_exports_outer_root_and_explicit_prefix():
    code = (
        "import json, os; "
        "print(json.dumps({key: os.environ.get(key) for key in "
        "('ETPR1_RUNTIME_ROOT', 'ETPR1_RUNTIME_PREFIX', 'ETPR1_RUNTIME_ACTIVE')}))"
    )
    result = _run([WRAPPER, sys.executable, "-c", code], env=_runtime_env())

    assert result.returncode == 0, result.stdout
    values = json.loads(result.stdout.strip())
    assert values["ETPR1_RUNTIME_ROOT"] == str(RUNTIME_ROOT)
    assert values["ETPR1_RUNTIME_PREFIX"] == str(RUNTIME_PREFIX)
    assert values["ETPR1_RUNTIME_ACTIVE"] == "1"


def test_wrapper_only_changes_child_environment():
    env = _runtime_env()
    script = (
        "unset ETPR1_RUNTIME_ACTIVE; "
        '"$1" "$2" -c "import os; assert '
        "os.environ['ETPR1_RUNTIME_ACTIVE'] == '1'\"; "
        'test -z "${ETPR1_RUNTIME_ACTIVE:-}"'
    )
    result = _run(
        [
            "bash",
            "-c",
            script,
            "bash",
            WRAPPER,
            sys.executable,
        ],
        env=env,
    )

    assert result.returncode == 0, result.stdout


def test_inspector_fails_when_habitat_imports_are_missing(tmp_path):
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ACTIVE"] = "1"
    env["ETPR1_RUNTIME_ROOT"] = str(tmp_path / "runtime")
    env["ETPR1_RUNTIME_PREFIX"] = str(tmp_path / "runtime" / "prefix")
    env["PYTHONPATH"] = str(tmp_path / "empty")
    env["LD_LIBRARY_PATH"] = str(tmp_path / "runtime" / "prefix" / "lib")

    result = _run([sys.executable, INSPECTOR], env=env, cwd=tmp_path)

    assert result.returncode != 0
    assert "IMPORT_ERROR" in result.stdout


def test_inspector_fails_when_torch_import_is_broken(tmp_path):
    broken_modules = tmp_path / "broken-modules"
    broken_modules.mkdir()
    (broken_modules / "torch.py").write_text("raise RuntimeError('broken torch')\n")
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ACTIVE"] = "1"
    env["ETPR1_RUNTIME_ROOT"] = str(tmp_path / "runtime")
    env["ETPR1_RUNTIME_PREFIX"] = str(tmp_path / "runtime" / "prefix")
    env["PYTHONPATH"] = str(broken_modules)

    result = _run([sys.executable, INSPECTOR], env=env, cwd=tmp_path)

    assert result.returncode != 0
    assert "torch=IMPORT_ERROR:broken torch" in result.stdout


def test_inspector_rejects_forbidden_pythonpath():
    command = [
        WRAPPER,
        "bash",
        "-c",
        'export PYTHONPATH="$PYTHONPATH:/tmp/ETPNav"; exec python "$1"',
        "bash",
        INSPECTOR,
    ]
    result = _run(command, env=_runtime_env())

    assert result.returncode != 0
    assert "forbidden" in result.stdout.lower()


def test_normal_wrapper_and_inspector_succeed():
    result = _run([WRAPPER, sys.executable, INSPECTOR], env=_runtime_env())

    assert result.returncode == 0, result.stdout
    assert "habitat_version=0.3.3" in result.stdout
    assert "habitat_sim_version=0.3.3" in result.stdout
    assert "habitat_baselines_version=0.3.3" in result.stdout


def test_builder_runtime_is_importable_and_idempotent():
    assert HABITAT_LAB_SOURCE.is_dir()
    assert HABITAT_SIM_SOURCE.is_dir()
    env = _runtime_env()
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(HABITAT_LAB_SOURCE)
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(HABITAT_SIM_SOURCE)

    first = _run(["bash", BUILDER], env=env)
    second = _run(["bash", BUILDER], env=env)

    assert first.returncode == 0, first.stdout
    assert second.returncode == 0, second.stdout
    assert "Reusing Habitat-Sim native binding" in second.stdout
    import_manifest = RUNTIME_ROOT / "manifests" / "runtime-imports.txt"
    native_manifest = RUNTIME_ROOT / "manifests" / "habitat-sim-native.txt"
    assert "habitat_sim=0.3.3" in import_manifest.read_text()
    native_text = native_manifest.read_text()
    assert "native_mode=" in native_text
    assert "habitat_sim_bindings_sha256=" in native_text


def test_builder_rejects_source_revision_mismatch(tmp_path):
    runtime_root = tmp_path / "runtime"
    lab_source = tmp_path / "lab-source"
    sim_source = tmp_path / "sim-source"
    lab_source.mkdir()
    sim_source.mkdir()
    (lab_source / "source.txt").write_text("lab-v1\n")
    (sim_source / "source.txt").write_text("sim-v1\n")

    for name, source in (("habitat-lab", lab_source), ("habitat-sim", sim_source)):
        target = runtime_root / "src" / name
        target.mkdir(parents=True)
        (target / ".etpr1_source_origin").write_text(
            f"source={source}\nrevision=tree:stale\n"
        )

    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(lab_source)
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(sim_source)
    result = _run(["bash", BUILDER], env=env)

    assert result.returncode != 0
    assert "source_revision_mismatch" in result.stdout


def test_builder_cleans_only_its_stale_copy_directories(tmp_path):
    runtime_root = tmp_path / "runtime"
    src_root = runtime_root / "src"
    owned_tmp = src_root / ".habitat-lab.etpr1-tmp.owned"
    external_tmp = src_root / ".habitat-lab.etpr1-tmp.external"
    owned_tmp.mkdir(parents=True)
    external_tmp.mkdir(parents=True)
    (owned_tmp / ".etpr1_builder_tmp").write_text("ETP-R1 habitat builder\n")

    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(tmp_path / "missing-lab")
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(tmp_path / "missing-sim")
    result = _run(["bash", BUILDER], env=env)

    assert result.returncode != 0
    assert not owned_tmp.exists()
    assert external_tmp.exists()
