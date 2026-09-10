"""Entry-level routing checks; execute training/evaluation only on the target host."""
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "ghost_concat_job_under_test", ROOT / "scripts/ghost_concat_job.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def invoke(monkeypatch, job, arguments):
    calls = []
    monkeypatch.setattr(job, "execute", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(sys, "argv", ["ghost_concat_job.py", *arguments])
    job.main()
    return calls


def test_default_preserves_existing_experiment(monkeypatch, job):
    (args, name, opts, checkpoint, mode, iterations), kwargs = invoke(
        monkeypatch, job, ["train", "--dry-run"]
    )[0]
    assert args.output == "data/logs/ghost_concat_20260908"
    assert name == "ghost_concat_v1_train"
    assert opts["MODEL.RAENWM.ghost_concat_memory_mode"] == "current_step_only"
    assert opts["IL.freeze_navigation_backbone"] is True
    assert checkpoint == job.BASE
    assert mode == "dagger"
    assert kwargs["config_file"] == job.CONFIG


def test_persistent_joint_train_separates_names_paths_and_sync(monkeypatch, job):
    (args, name, opts, checkpoint, mode, iterations), kwargs = invoke(
        monkeypatch, job,
        ["train", "--dry-run", "--memory-mode", "persistent_node_state",
         "--train-policy", "--gpus", "0,1", "--compile-model"],
    )[0]
    assert args.output == "data/logs/ghost_concat_persistent"
    assert name == "ghost_concat_v1_joint_persistent_train"
    assert opts["MODEL.RAENWM.ghost_concat_memory_mode"] == "persistent_node_state"
    assert opts["IL.freeze_navigation_backbone"] is False
    assert opts["IL.lr"] == 2e-6
    assert opts["IL.rgb_fusion_lr"] == 1e-5
    assert opts["MODEL.RAENWM.torch_compile"] is True
    assert opts["IL.checkpoint_sync_destination"].endswith(
        f"/{args.output}/train/{name}/checkpoints/{name}"
    )
    assert kwargs["config_file"] == job.PERSISTENT_CONFIG


def test_persistent_watch_uses_matching_checkpoint_names(monkeypatch, job):
    calls = invoke(monkeypatch, job, [
        "watch", "--machine", "eval", "--dry-run", "--train-policy",
        "--ghost-concat-memory-mode", "persistent_node_state",
        "--output", "data/logs/persistent_smoke", "--eval-iterations", "2,4",
    ])
    for (args, name, opts, checkpoint, mode), kwargs in calls:
        assert name.startswith("ghost_concat_v1_joint_persistent_eval_iter")
        assert "/ghost_concat_v1_joint_persistent_train/" in checkpoint
        assert opts["EVAL.USE_CKPT_CONFIG"] is False
        assert opts["MODEL.RAENWM.ghost_concat_memory_mode"] == "persistent_node_state"
        assert kwargs["config_file"] == job.PERSISTENT_CONFIG
    assert [call[0][3].split("/")[-1] for call in calls] == ["ckpt.iter2.pth", "ckpt.iter4.pth"]


@pytest.mark.parametrize("workspace", [
    "/home/gwl/project/etpr1/ETP-R1",
    "/home/gwl/project/etpr1/ETP-R1-persistent-ghost",
])
def test_exact_server_workspaces_are_accepted(monkeypatch, job, workspace):
    monkeypatch.setattr(job, "ROOT", Path(workspace))
    assert len(invoke(monkeypatch, job, ["train"])) == 1


@pytest.mark.parametrize("workspace", [
    "/home/gwl/project/etpr1/ETP-R1-persistent-ghost-other",
    "/home/sia/project/ETP-R1-persistent-ghost",
    "/home/a6000/gwl/ETP-R1",
])
def test_other_server_workspaces_are_rejected(monkeypatch, job, workspace):
    monkeypatch.setattr(job, "ROOT", Path(workspace))
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, job, ["train"])
    assert exc.value.code == 2


def test_no_grpo_entry_is_exposed(monkeypatch, job):
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, job, ["grpo", "--dry-run", "--memory-mode", "persistent_node_state"])
    assert exc.value.code == 2


def test_persistent_config_uses_direct_fp16_without_implicit_compile(job):
    import yaml

    config = yaml.safe_load((ROOT / job.PERSISTENT_CONFIG).read_text())
    nwm = config["MODEL"]["RAENWM"]
    assert nwm["ghost_concat_memory_mode"] == "persistent_node_state"
    assert nwm["panorama_observation_source"] == "direct"
    assert nwm["panorama_visual_precision"] == "fp16"
    assert nwm["panorama_encode_batch_size"] == 64
    assert nwm["panorama_prediction_batch_size"] == 64
    assert nwm["torch_compile"] is False
    assert config["TRAINER_NAME"] == "SS-ETP-R1"


def test_persistent_visual_settings_remain_overridable(monkeypatch, job):
    args, kwargs = invoke(monkeypatch, job, [
        "train", "--dry-run", "--memory-mode", "persistent_node_state",
        "--observation-source", "cube", "--visual-precision", "float32",
        "--dino-batch", "16", "--nwm-batch", "8",
    ])[0]
    opts = args[2]
    assert opts["MODEL.RAENWM.panorama_observation_source"] == "cube"
    assert opts["MODEL.RAENWM.panorama_visual_precision"] == "float32"
    assert opts["MODEL.RAENWM.panorama_encode_batch_size"] == 16
    assert opts["MODEL.RAENWM.panorama_prediction_batch_size"] == 8
