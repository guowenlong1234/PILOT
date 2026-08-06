import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.audit_rae_smoke import _assert_online_checkpoint, audit_pretrain
from vlnce_baselines.models import checkpoint_utils as checkpoint_module
from scripts.prepare_rae_smoke_pretrain import (
    prepare_smoke_config,
    snapshot_initial_img_linear,
)


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_prepare_smoke_config_uses_one_real_record_per_split(tmp_path):
    train = tmp_path / "train.jsonl"
    r2r_val = tmp_path / "r2r_val.jsonl"
    rxr_val = tmp_path / "rxr_val.jsonl"
    _write_jsonl(train, [{"id": "train-first"}, {"id": "train-second"}])
    _write_jsonl(r2r_val, [{"id": "r2r-first"}, {"id": "r2r-second"}])
    _write_jsonl(rxr_val, [{"id": "rxr-first"}, {"id": "rxr-second"}])
    source = {
        "train_batch_size": 32,
        "val_batch_size": 32,
        "val_sample_num": 5000,
        "n_workers": 4,
        "pin_mem": True,
        "num_train_steps": 500000,
        "valid_steps": 2500,
        "log_steps": 2500,
        "warmup_steps": 20000,
        "seed": 0,
        "fp16": True,
        "train_datasets": {
            "R2R": {
                "train_traj_files": [str(train), "unused.jsonl"],
                "val_unseen_r2r_traj_files": [str(r2r_val)],
                "val_unseen_rxr_traj_files": [str(rxr_val)],
                "img_ft_file": "formal-image.hdf5",
                "dep_ft_file": "formal-depth.hdf5",
                "scanvp_cands_file": "formal-candidates.json",
                "connectivity_dir": "formal-connectivity",
                "tasks": ["mlm", "sap"],
                "val_tasks": ["mlm", "sap"],
            }
        },
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    output_path = tmp_path / "fixtures" / "smoke.json"

    prepared = prepare_smoke_config(source_path, output_path, seed=20260710)

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == prepared
    dataset = loaded["train_datasets"]["R2R"]
    for key, expected_id in (
        ("train_traj_files", "train-first"),
        ("val_unseen_r2r_traj_files", "r2r-first"),
        ("val_unseen_rxr_traj_files", "rxr-first"),
    ):
        assert len(dataset[key]) == 1
        fixture = Path(dataset[key][0])
        assert fixture.parent == output_path.parent
        assert json.loads(fixture.read_text(encoding="utf-8"))["id"] == expected_id
    assert dataset["img_ft_file"] == "formal-image.hdf5"
    assert dataset["dep_ft_file"] == "formal-depth.hdf5"
    assert dataset["scanvp_cands_file"] == "formal-candidates.json"
    assert dataset["connectivity_dir"] == "formal-connectivity"
    assert {
        key: loaded[key]
        for key in (
            "train_batch_size",
            "val_batch_size",
            "val_sample_num",
            "n_workers",
            "pin_mem",
            "num_train_steps",
            "valid_steps",
            "log_steps",
            "warmup_steps",
            "seed",
            "fp16",
        )
    } == {
        "train_batch_size": 1,
        "val_batch_size": 1,
        "val_sample_num": 1,
        "n_workers": 0,
        "pin_mem": False,
        "num_train_steps": 1,
        "valid_steps": 1,
        "log_steps": 1,
        "warmup_steps": 0,
        "seed": 20260710,
        "fp16": False,
    }


def test_prepare_smoke_config_rejects_empty_source_jsonl(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    source = {
        "train_datasets": {
            "R2R": {
                "train_traj_files": [str(empty)],
                "val_unseen_r2r_traj_files": [str(empty)],
                "val_unseen_rxr_traj_files": [str(empty)],
            }
        }
    }
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="empty JSONL"):
        prepare_smoke_config(source_path, tmp_path / "out" / "smoke.json")


def _img_linear_state(value=1.0):
    return {
        "bert.img_embeddings.img_linear.weight": torch.full(
            (768, 768), value
        ),
        "bert.img_embeddings.img_linear.bias": torch.full((768,), value),
    }


def test_pretrain_audit_rejects_img_linear_only_checkpoint(tmp_path):
    checkpoint = _img_linear_state()
    path = tmp_path / "img_linear_only.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="complete formal model"):
        audit_pretrain(path)


def test_pretrain_audit_accepts_complete_formal_checkpoint(tmp_path):
    checkpoint = _img_linear_state()
    checkpoint["bert.embeddings.word_embeddings.weight"] = torch.ones(1)
    checkpoint["bert.global_encoder.encoder.x_layers.0.lang_self_att.self.query.weight"] = torch.ones(1)
    path = tmp_path / "formal.pt"
    torch.save(checkpoint, path)

    initial_path = tmp_path / "initial.pt"
    torch.save(
        {
            key: torch.zeros_like(value)
            for key, value in checkpoint.items()
            if "img_embeddings.img_linear." in key
        },
        initial_path,
    )

    audit_pretrain(path, initial_path)


def test_pretrain_audit_rejects_unchanged_formal_img_linear(tmp_path):
    checkpoint = _img_linear_state()
    checkpoint["bert.embeddings.word_embeddings.weight"] = torch.ones(1)
    checkpoint["bert.global_encoder.encoder.x_layers.0.lang_self_att.self.query.weight"] = torch.ones(1)
    formal_path = tmp_path / "formal.pt"
    initial_path = tmp_path / "initial.pt"
    torch.save(checkpoint, formal_path)
    torch.save(
        {
            key: value
            for key, value in checkpoint.items()
            if "img_embeddings.img_linear." in key
        },
        initial_path,
    )

    with pytest.raises(ValueError, match="did not update"):
        audit_pretrain(formal_path, initial_path)


def test_snapshot_initial_img_linear_saves_weight_and_bias(tmp_path):
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = torch.nn.Module()
            self.bert.img_embeddings = torch.nn.Module()
            self.bert.img_embeddings.img_linear = torch.nn.Linear(3, 2)
            self.other = torch.nn.Linear(2, 2)

    path = tmp_path / "initial.pt"

    snapshot_initial_img_linear(FakeModel(), path)

    checkpoint = torch.load(path, map_location="cpu")
    assert len(checkpoint) == 2
    assert set(checkpoint) == {
        "bert.img_embeddings.img_linear.weight",
        "bert.img_embeddings.img_linear.bias",
    }


def _online_audit_checkpoint(tmp_path):
    model_dir = tmp_path / "pretrained" / "rae"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.safetensors"
    model_path.write_bytes(b"model")
    config = SimpleNamespace(
        MODEL=SimpleNamespace(
            RGB_ENCODER=SimpleNamespace(
                type="rae_dinov2",
                model_dir=str(model_dir),
                output_size=768,
                cls_residual_mlp_enabled=True,
                cls_residual_mlp_hidden_dim=768,
                cls_residual_mlp_zero_init=True,
            )
        )
    )
    state = {
        "net.vln_bert.img_embeddings.img_linear.weight": torch.ones(768, 768),
        "net.vln_bert.img_embeddings.img_linear.bias": torch.ones(768),
    }
    for suffix in ("0.weight", "2.weight", "4.weight"):
        state[
            f"net.rgb_encoder.cls_residual_mlp.layers.{suffix}"
        ] = torch.zeros(768, 768)
    for suffix in ("0.bias", "2.bias", "4.bias"):
        state[
            f"net.rgb_encoder.cls_residual_mlp.layers.{suffix}"
        ] = torch.zeros(768)
    state["net.vln_bert.global_encoder.weight"] = torch.ones(1)
    metadata = {
        "type": "rae_dinov2",
        "pipeline": "etpnav_raw_cls_residual_mlp_v1",
        "model_dir": "pretrained/rae",
        "model_sha256": hashlib.sha256(b"model").hexdigest(),
        "cls_normalization": "none",
        "output_size": 768,
        "cls_residual_mlp_enabled": True,
        "cls_residual_mlp_hidden_dim": 768,
        "cls_residual_mlp_zero_init": True,
    }
    return {
        "state_dict": state,
        "rgb_encoder": metadata,
        "config": config,
        "iteration": 1,
        "optim_state": {"state": {0: {"step": 1}}},
        "scheduler_state": {"last_epoch": 1, "base_lrs": [1e-5]},
    }


def test_online_audit_rejects_missing_scheduler_state(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "_PROJECT_ROOT", tmp_path)
    checkpoint = _online_audit_checkpoint(tmp_path)
    del checkpoint["scheduler_state"]

    with pytest.raises(ValueError, match="scheduler_state"):
        _assert_online_checkpoint(checkpoint, 1)


def test_online_audit_accepts_separate_training_state(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "_PROJECT_ROOT", tmp_path)
    checkpoint = _online_audit_checkpoint(tmp_path)
    training_state = {
        "iteration": checkpoint["iteration"],
        "optim_state": checkpoint.pop("optim_state"),
        "scheduler_state": checkpoint.pop("scheduler_state"),
    }

    metadata = _assert_online_checkpoint(checkpoint, 1, training_state)

    assert metadata["type"] == "rae_dinov2"


@pytest.mark.parametrize(
    ("missing_field", "message"),
    (
        ("pipeline", "pipeline"),
        ("model_dir", "model_dir"),
        ("model_sha256", "model_sha256"),
        ("cls_normalization", "cls_normalization"),
        ("cls_residual_mlp_enabled", "cls_residual_mlp_enabled"),
    ),
)
def test_online_audit_rejects_incomplete_rae_metadata(
    tmp_path,
    monkeypatch,
    missing_field,
    message,
):
    monkeypatch.setattr(checkpoint_module, "_PROJECT_ROOT", tmp_path)
    checkpoint = _online_audit_checkpoint(tmp_path)
    del checkpoint["rgb_encoder"][missing_field]

    with pytest.raises(ValueError, match=message):
        _assert_online_checkpoint(checkpoint, 1)


def test_online_audit_rejects_direct_rgb_backbone_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_module, "_PROJECT_ROOT", tmp_path)
    checkpoint = _online_audit_checkpoint(tmp_path)
    checkpoint["state_dict"]["rgb_encoder.backbone.layer.weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="frozen DINO backbone"):
        _assert_online_checkpoint(checkpoint, 1)
