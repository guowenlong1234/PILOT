from pathlib import Path
import json
import random
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pretrain_src.pretrain_src.utils.save import (
    BestModelSaver,
    ModelSaver,
    build_joint_accuracy_metrics,
    capture_rng_state,
    load_training_state,
    resolve_resume_meta_loader_step,
    resolve_resume_checkpoint,
    restore_rng_state,
    validate_resume_config,
)


def _opts(tmp_path, **overrides):
    model_config = tmp_path / "model.json"
    model_config.write_text("{}", encoding="utf-8")
    values = {
        "gradient_accumulation_steps": 8,
        "train_batch_size": 16,
        "world_size": 1,
        "model_config": str(model_config),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _updated_model_and_optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model, optimizer


def test_resumable_checkpoint_round_trip_contains_complete_state(tmp_path):
    model, optimizer = _updated_model_and_optimizer()
    opts = _opts(tmp_path)
    saver = ModelSaver(str(tmp_path / "ckpts"))

    model_path, state_path = saver.save(
        model,
        25,
        optimizer=optimizer,
        meta_loader_step=200,
        opts=opts,
    )

    resolved = resolve_resume_checkpoint("latest", str(tmp_path / "ckpts"))
    state, referenced_model = load_training_state(resolved)
    assert resolved == state_path
    assert referenced_model == model_path
    assert state["step"] == 25
    assert state["meta_loader_step"] == 200
    assert state["optimizer"]["state"]
    assert state["training_config"]["gradient_accumulation_steps"] == 8
    assert validate_resume_config(state, opts) is None


def test_resume_configuration_rejects_effective_batch_change(tmp_path):
    model, optimizer = _updated_model_and_optimizer()
    saver = ModelSaver(str(tmp_path / "ckpts"))
    _, state_path = saver.save(model, 2, optimizer=optimizer, opts=_opts(tmp_path))
    state, _ = load_training_state(state_path)

    with pytest.raises(ValueError, match="effective batch size"):
        validate_resume_config(
            state, _opts(tmp_path, gradient_accumulation_steps=4)
        )


def test_resume_allows_equivalent_effective_batch_geometry(tmp_path):
    model, optimizer = _updated_model_and_optimizer()
    saved_opts = _opts(
        tmp_path,
        train_batch_size=32,
        gradient_accumulation_steps=2,
        world_size=2,
    )
    saver = ModelSaver(str(tmp_path / "ckpts"))
    _, state_path = saver.save(
        model,
        10,
        optimizer=optimizer,
        meta_loader_step=20,
        opts=saved_opts,
    )
    state, _ = load_training_state(state_path)
    resumed_opts = _opts(
        tmp_path,
        train_batch_size=64,
        gradient_accumulation_steps=1,
        world_size=2,
    )

    assert validate_resume_config(state, resumed_opts) is None
    assert resolve_resume_meta_loader_step(state, resumed_opts) == 10


def test_resume_rejects_inconsistent_saved_meta_loader_step(tmp_path):
    opts = _opts(tmp_path)
    state = {
        "step": 10,
        "meta_loader_step": 79,
        "training_config": {
            "gradient_accumulation_steps": 8,
        },
    }

    with pytest.raises(ValueError, match="MetaLoader step mismatch"):
        resolve_resume_meta_loader_step(state, opts)


def test_latest_resume_ignores_model_without_training_state(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    torch.save({}, checkpoint_dir / "model_step_100.pt")
    torch.save({}, checkpoint_dir / "model_step_50.pt")
    torch.save(
        {
            "format_version": 1,
            "step": 50,
            "model_checkpoint": "model_step_50.pt",
            "optimizer": {},
            "rng_state": {},
            "meta_loader_step": 400,
            "training_config": {},
        },
        checkpoint_dir / "train_state_50.pt",
    )

    assert resolve_resume_checkpoint("latest", str(checkpoint_dir)).endswith(
        "train_state_50.pt"
    )


def test_checkpoint_pruning_keeps_recent_pairs_and_sparse_models(tmp_path):
    model, optimizer = _updated_model_and_optimizer()
    opts = _opts(tmp_path)
    checkpoint_dir = tmp_path / "ckpts"
    saver = ModelSaver(str(checkpoint_dir))

    for step in (10, 20, 30, 40):
        saver.save(
            model,
            step,
            optimizer=optimizer,
            opts=opts,
            keep_last_checkpoints=2,
            keep_every_n_steps=20,
        )

    assert sorted(path.name for path in checkpoint_dir.glob("train_state_*.pt")) == [
        "train_state_30.pt",
        "train_state_40.pt",
    ]
    assert sorted(path.name for path in checkpoint_dir.glob("model_step_*.pt")) == [
        "model_step_20.pt",
        "model_step_30.pt",
        "model_step_40.pt",
    ]


def _validation_metrics(mlm_acc, sap_gacc):
    return {
        "val_unseen_mlm_acc": mlm_acc,
        "val_unseen_sap_gacc": sap_gacc,
    }


def test_joint_accuracy_score_uses_equal_r2r_rxr_means():
    metrics = build_joint_accuracy_metrics(
        2500,
        _validation_metrics(0.2, 0.4),
        _validation_metrics(0.6, 0.8),
    )

    assert metrics["mlm_acc_mean"] == pytest.approx(0.4)
    assert metrics["sap_gacc_mean"] == pytest.approx(0.6)
    assert metrics["score"] == pytest.approx(1.0)
    assert metrics["step"] == 2500


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_joint_accuracy_score_rejects_nonfinite_metrics(value):
    with pytest.raises(ValueError, match="finite"):
        build_joint_accuracy_metrics(
            2500,
            _validation_metrics(value, 0.4),
            _validation_metrics(0.6, 0.8),
        )


def test_best_model_saver_keeps_only_strictly_better_joint_score(tmp_path):
    checkpoint_dir = tmp_path / "ckpts"
    checkpoint_dir.mkdir()
    first_model = checkpoint_dir / "model_step_2500.pt"
    second_model = checkpoint_dir / "model_step_5000.pt"
    third_model = checkpoint_dir / "model_step_7500.pt"
    first_model.write_bytes(b"first")
    second_model.write_bytes(b"second")
    third_model.write_bytes(b"third")
    saver = BestModelSaver(str(tmp_path / "best"))

    first_metrics = build_joint_accuracy_metrics(
        2500,
        _validation_metrics(0.2, 0.4),
        _validation_metrics(0.6, 0.8),
    )
    worse_metrics = build_joint_accuracy_metrics(
        5000,
        _validation_metrics(0.1, 0.4),
        _validation_metrics(0.5, 0.8),
    )
    better_metrics = build_joint_accuracy_metrics(
        7500,
        _validation_metrics(0.3, 0.5),
        _validation_metrics(0.7, 0.9),
    )

    assert saver.maybe_save(str(first_model), first_metrics) is True
    first_best = tmp_path / "best" / "model_best_step_2500.pt"
    assert first_best.stat().st_ino == first_model.stat().st_ino
    assert saver.maybe_save(str(second_model), worse_metrics) is False
    assert first_best.exists()
    assert saver.maybe_save(str(third_model), better_metrics) is True

    assert not first_best.exists()
    third_best = tmp_path / "best" / "model_best_step_7500.pt"
    assert third_best.stat().st_ino == third_model.stat().st_ino
    metadata = json.loads(
        (tmp_path / "best" / "best_metrics.json").read_text(encoding="utf-8")
    )
    assert metadata["step"] == 7500
    assert metadata["model_checkpoint"] == "model_best_step_7500.pt"
    assert metadata["score"] == pytest.approx(better_metrics["score"])


def test_rng_state_round_trip_restores_python_numpy_and_torch():
    state = capture_rng_state()
    expected_python = random.random()
    expected_numpy = np.random.rand(3)
    expected_torch = torch.rand(3)

    restore_rng_state(state)

    assert random.random() == expected_python
    assert np.array_equal(np.random.rand(3), expected_numpy)
    assert torch.equal(torch.rand(3), expected_torch)


def test_management_scripts_have_valid_bash_syntax():
    root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/manage_rae_pretrain.sh",
        "scripts/manage_rae_pretrain_host.sh",
        "scripts/run_rae_pretrain_job.sh",
        "pretrain_src/run_pt/run_mix_rae_dino.bash",
    ):
        path = root / relative_path
        result = subprocess.run(
            ["bash", "-n", str(path)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


def test_supervised_job_records_manifest_identity_without_requiring_git():
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/run_rae_pretrain_job.sh").read_text(
        encoding="utf-8"
    )

    assert "rae_smoke_source_identity.py" in source
    assert "source_manifest" in source
    assert "git rev-parse HEAD" not in source
