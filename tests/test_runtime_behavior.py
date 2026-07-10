import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_etpr1_habitat.sh"
WRAPPER = ROOT / "scripts" / "etpr1_rae_runtime_exec.sh"
INSPECTOR = ROOT / "scripts" / "inspect_etpr1_runtime.py"
RUNTIME_ROOT = ROOT / ".runtime" / "etpr1_habitat"
RUNTIME_PREFIX = RUNTIME_ROOT / "prefix"
RUNTIME_LIB = RUNTIME_PREFIX / "lib"
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
    env["LD_LIBRARY_PATH"] = ":".join(
        (
            str(RUNTIME_LIB),
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
        )
    )
    return env


def _init_git_source(path):
    path.mkdir(parents=True)
    (path / "source.txt").write_text("version-1\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "ETP-R1 Test"],
        ["git", "config", "user.email", "etpr1-test@example.invalid"],
        ["git", "add", "source.txt"],
        ["git", "commit", "-q", "-m", "initial"],
    ):
        result = _run(command, cwd=path)
        assert result.returncode == 0, result.stdout
    return _run(["git", "rev-parse", "HEAD"], cwd=path).stdout.strip()


def _write_source_marker(runtime_root, name, source, revision):
    target = runtime_root / "src" / name
    target.mkdir(parents=True)
    (target / ".etpr1_source_origin").write_text(
        f"source={source.resolve()}\nrevision={revision}\n"
    )


def test_wrapper_rejects_incomplete_runtime():
    (ROOT / ".runtime").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wrapper-incomplete-", dir=ROOT / ".runtime"
    ) as temp_dir:
        runtime_root = Path(temp_dir)
        prefix = runtime_root / "prefix"
        (prefix / "site-packages").mkdir(parents=True)
        (prefix / "habitat-baselines").mkdir()
        (prefix / "lib").mkdir()
        env = os.environ.copy()
        env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
        env["ETPR1_RUNTIME_PREFIX"] = str(prefix)

        result = _run(
            [WRAPPER, sys.executable, "-c", "print('unexpected')"], env=env
        )

        missing_path = prefix / "site-packages" / "habitat"
        assert result.returncode != 0
        assert f"missing habitat package: {missing_path}" in result.stdout


def test_wrapper_rejects_runtime_symlink_escape():
    (ROOT / ".runtime").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wrapper-escape-", dir=ROOT / ".runtime"
    ) as temp_dir:
        runtime_root = Path(temp_dir)
        prefix = runtime_root / "prefix"
        site_packages = prefix / "site-packages"
        (site_packages / "habitat_sim" / "_ext").mkdir(parents=True)
        (prefix / "habitat-baselines" / "habitat_baselines").mkdir(parents=True)
        (prefix / "lib").mkdir()
        (site_packages / "habitat_sim" / "_ext" / "habitat_sim_bindings.so").touch()
        (site_packages / "_corrade.so").touch()
        (site_packages / "_magnum.so").touch()
        escaped_target = RUNTIME_PREFIX / "site-packages" / "habitat"
        (site_packages / "habitat").symlink_to(
            escaped_target,
            target_is_directory=True,
        )
        env = os.environ.copy()
        env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
        env["ETPR1_RUNTIME_PREFIX"] = str(prefix)

        result = _run(
            [WRAPPER, sys.executable, "-c", "print('unexpected')"], env=env
        )

        assert result.returncode != 0
        assert "habitat package escapes ETP-R1 owner" in result.stdout
        assert str(escaped_target.resolve()) in result.stdout


def test_wrapper_rejects_missing_native_binding():
    (ROOT / ".runtime").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wrapper-native-", dir=ROOT / ".runtime"
    ) as temp_dir:
        runtime_root = Path(temp_dir)
        prefix = runtime_root / "prefix"
        site_packages = prefix / "site-packages"
        (site_packages / "habitat").mkdir(parents=True)
        (site_packages / "habitat_sim" / "_ext").mkdir(parents=True)
        (prefix / "habitat-baselines" / "habitat_baselines").mkdir(parents=True)
        (prefix / "lib").mkdir()
        env = os.environ.copy()
        env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
        env["ETPR1_RUNTIME_PREFIX"] = str(prefix)

        result = _run(
            [WRAPPER, sys.executable, "-c", "print('unexpected')"], env=env
        )

        assert result.returncode != 0
        assert "missing Habitat-Sim native binding" in result.stdout


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


def test_builder_uses_head_revision_for_clean_git_sources(tmp_path):
    runtime_root = tmp_path / "runtime"
    lab_source = tmp_path / "lab-source"
    sim_source = tmp_path / "sim-source"
    lab_head = _init_git_source(lab_source)
    sim_head = _init_git_source(sim_source)
    _write_source_marker(runtime_root, "habitat-lab", lab_source, f"git:{lab_head}")
    _write_source_marker(runtime_root, "habitat-sim", sim_source, "git:stale")
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(lab_source)
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(sim_source)

    result = _run(["bash", BUILDER], env=env)

    assert result.returncode != 0
    assert "source_revision_mismatch" in result.stdout
    assert f"git:{sim_head}" in result.stdout


def test_builder_rejects_dirty_tracked_git_source(tmp_path):
    runtime_root = tmp_path / "runtime"
    lab_source = tmp_path / "lab-source"
    sim_source = tmp_path / "sim-source"
    lab_head = _init_git_source(lab_source)
    sim_head = _init_git_source(sim_source)
    _write_source_marker(runtime_root, "habitat-lab", lab_source, f"git:{lab_head}")
    _write_source_marker(runtime_root, "habitat-sim", sim_source, f"git:{sim_head}")
    (lab_source / "source.txt").write_text("modified\n")
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(lab_source)
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(sim_source)

    result = _run(["bash", BUILDER], env=env)

    assert result.returncode != 0
    assert "source_dirty" in result.stdout
    assert str(lab_source) in result.stdout
    assert "source.txt" in result.stdout


def test_builder_rejects_dirty_untracked_git_source(tmp_path):
    runtime_root = tmp_path / "runtime"
    lab_source = tmp_path / "lab-source"
    sim_source = tmp_path / "sim-source"
    lab_head = _init_git_source(lab_source)
    sim_head = _init_git_source(sim_source)
    _write_source_marker(runtime_root, "habitat-lab", lab_source, f"git:{lab_head}")
    _write_source_marker(runtime_root, "habitat-sim", sim_source, f"git:{sim_head}")
    (lab_source / "untracked.txt").write_text("untracked\n")
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(lab_source)
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(sim_source)

    result = _run(["bash", BUILDER], env=env)

    assert result.returncode != 0
    assert "source_dirty" in result.stdout
    assert str(lab_source) in result.stdout
    assert "untracked.txt" in result.stdout


def test_builder_does_not_copy_new_dirty_tracked_git_source(tmp_path):
    runtime_root = tmp_path / "runtime"
    lab_source = tmp_path / "lab-source"
    sim_source = tmp_path / "sim-source"
    _init_git_source(lab_source)
    _init_git_source(sim_source)
    (lab_source / "source.txt").write_text("modified\n")
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(lab_source)
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(sim_source)

    result = _run(["bash", BUILDER], env=env)

    target = runtime_root / "src" / "habitat-lab"
    assert result.returncode != 0
    assert "source_dirty" in result.stdout
    assert not target.exists()
    assert not (target / ".etpr1_source_origin").exists()


def test_builder_does_not_copy_new_dirty_untracked_git_source(tmp_path):
    runtime_root = tmp_path / "runtime"
    lab_source = tmp_path / "lab-source"
    sim_source = tmp_path / "sim-source"
    _init_git_source(lab_source)
    _init_git_source(sim_source)
    (lab_source / "untracked.txt").write_text("untracked\n")
    env = os.environ.copy()
    env["ETPR1_RUNTIME_ROOT"] = str(runtime_root)
    env["ETPR1_HABITAT_LAB_SOURCE"] = str(lab_source)
    env["ETPR1_HABITAT_SIM_SOURCE"] = str(sim_source)

    result = _run(["bash", BUILDER], env=env)

    target = runtime_root / "src" / "habitat-lab"
    assert result.returncode != 0
    assert "source_dirty" in result.stdout
    assert not target.exists()
    assert not (target / ".etpr1_source_origin").exists()


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
