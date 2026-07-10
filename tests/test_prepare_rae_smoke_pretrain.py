import json
from pathlib import Path

import pytest
import torch

from scripts.audit_rae_smoke import audit_pretrain
from scripts.prepare_rae_smoke_pretrain import (
    prepare_smoke_config,
    snapshot_initial_projection,
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


def test_pretrain_audit_rejects_projection_only_checkpoint(tmp_path):
    checkpoint = {
        f"bert.img_embeddings.rgb_projection.{suffix}": torch.ones(1)
        for suffix in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
    }
    path = tmp_path / "projection_only.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="complete formal model"):
        audit_pretrain(path)


def test_pretrain_audit_accepts_complete_formal_checkpoint(tmp_path):
    checkpoint = {
        f"bert.img_embeddings.rgb_projection.{suffix}": torch.ones(1)
        for suffix in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
    }
    checkpoint["bert.embeddings.word_embeddings.weight"] = torch.ones(1)
    checkpoint["bert.global_encoder.encoder.x_layers.0.lang_self_att.self.query.weight"] = torch.ones(1)
    path = tmp_path / "formal.pt"
    torch.save(checkpoint, path)

    initial_path = tmp_path / "initial.pt"
    torch.save(
        {
            key: torch.zeros_like(value)
            for key, value in checkpoint.items()
            if "rgb_projection." in key
        },
        initial_path,
    )

    audit_pretrain(path, initial_path)


def test_pretrain_audit_rejects_unchanged_formal_projection(tmp_path):
    checkpoint = {
        f"bert.img_embeddings.rgb_projection.{suffix}": torch.ones(1)
        for suffix in ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
    }
    checkpoint["bert.embeddings.word_embeddings.weight"] = torch.ones(1)
    checkpoint["bert.global_encoder.encoder.x_layers.0.lang_self_att.self.query.weight"] = torch.ones(1)
    formal_path = tmp_path / "formal.pt"
    initial_path = tmp_path / "initial.pt"
    torch.save(checkpoint, formal_path)
    torch.save(
        {
            key: value
            for key, value in checkpoint.items()
            if "rgb_projection." in key
        },
        initial_path,
    )

    with pytest.raises(ValueError, match="did not update"):
        audit_pretrain(formal_path, initial_path)


def test_snapshot_initial_projection_saves_exactly_six_parameters(tmp_path):
    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = torch.nn.Module()
            self.bert.img_embeddings = torch.nn.Module()
            self.bert.img_embeddings.rgb_projection = torch.nn.Sequential(
                torch.nn.Linear(3, 3),
                torch.nn.GELU(),
                torch.nn.Linear(3, 3),
                torch.nn.GELU(),
                torch.nn.Linear(3, 2),
            )
            self.other = torch.nn.Linear(2, 2)

    path = tmp_path / "initial.pt"

    snapshot_initial_projection(FakeModel(), path)

    checkpoint = torch.load(path, map_location="cpu")
    assert len(checkpoint) == 6
    assert all("rgb_projection." in key for key in checkpoint)
