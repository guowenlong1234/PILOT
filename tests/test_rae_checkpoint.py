import hashlib
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.GRPO_trainer_ETP_R1 import RLTrainer as GrpoTrainer
from vlnce_baselines import ss_trainer_ETP_R1 as sft_trainer_module
from vlnce_baselines.models import checkpoint_utils as checkpoint_module
from vlnce_baselines.models.checkpoint_utils import (
    navigation_state_dict,
    report_navigation_incompatible_keys,
    sha256_file,
    validate_rgb_checkpoint_metadata,
)
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer as SftTrainer


PROJECTION_PARAMETER_SUFFIXES = (
    "0.weight",
    "0.bias",
    "2.weight",
    "2.bias",
    "4.weight",
    "4.bias",
)


@pytest.fixture(autouse=True)
def _use_test_project_root(tmp_path, monkeypatch):
    monkeypatch.setattr(
        checkpoint_module,
        "_PROJECT_ROOT",
        tmp_path,
        raising=False,
    )


class _FakeRgbEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.backbone.requires_grad_(False)
        self.register_buffer("latent_var", torch.ones(2))


class _FakeImageEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_projection = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, 768),
            torch.nn.GELU(),
            torch.nn.Linear(768, 512),
        )


class _FakeVlnBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_embeddings = _FakeImageEmbeddings()
        self.global_encoder = torch.nn.Linear(2, 2)


class _FakeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_encoder = _FakeRgbEncoder()
        self.vln_bert = _FakeVlnBert()


class _FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _FakeNet()


def _write_rae_assets(tmp_path):
    model_dir = (
        tmp_path / "pretrained" / "rae_dinov2_with_registers_base"
    )
    model_dir.mkdir(parents=True)
    model_path = model_dir / "model.safetensors"
    stat_path = model_dir / "stat.pt"
    model_path.write_bytes(b"real model bytes")
    stat_path.write_bytes(b"real stat bytes")
    return model_dir, model_path, stat_path


def _config(tmp_path, encoder_type="rae_dinov2"):
    model_dir, model_path, stat_path = _write_rae_assets(tmp_path)
    raw_output_size = 768 if encoder_type == "rae_dinov2" else 512
    return (
        SimpleNamespace(
            MODEL=SimpleNamespace(
                RGB_ENCODER=SimpleNamespace(
                    type=encoder_type,
                    model_dir=str(model_dir),
                    stat_path=str(stat_path),
                    raw_output_size=raw_output_size,
                    output_size=512,
                )
            )
        ),
        model_path,
        stat_path,
    )


def _rae_checkpoint(policy, config):
    state_dict, metadata = navigation_state_dict(policy, config)
    return {"state_dict": state_dict, "rgb_encoder": metadata}


def _data_parallel_policy():
    policy = _FakePolicy()
    policy.net = torch.nn.DataParallel(policy.net)
    return policy


def _checkpoint_with_plain_and_wrapped_projection(config):
    plain_state, metadata = navigation_state_dict(_FakePolicy(), config)
    wrapped_state, _ = navigation_state_dict(_data_parallel_policy(), config)
    return {
        "state_dict": {**plain_state, **wrapped_state},
        "rgb_encoder": metadata,
    }


def test_sha256_file_streams_real_file_and_rejects_invalid_paths(tmp_path):
    payload = b"checkpoint metadata must hash file contents"
    file_path = tmp_path / "asset.bin"
    file_path.write_bytes(payload)

    assert sha256_file(file_path) == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sha256_file(tmp_path / "missing.bin")
    with pytest.raises(ValueError, match="not a file"):
        sha256_file(tmp_path)


def test_rae_navigation_state_filters_wrapped_backbones_but_keeps_projection(
    tmp_path,
):
    config, model_path, stat_path = _config(tmp_path)
    policy = _data_parallel_policy()

    state_dict, metadata = navigation_state_dict(policy, config)

    assert not any("rgb_encoder.backbone" in key for key in state_dict)
    assert (
        "net.module.vln_bert.img_embeddings.rgb_projection.0.weight"
        in state_dict
    )
    assert not any(key.startswith("net.vln_bert") for key in state_dict)
    assert metadata == {
        "type": "rae_dinov2",
        "model_dir": "pretrained/rae_dinov2_with_registers_base",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "stat_sha256": hashlib.sha256(stat_path.read_bytes()).hexdigest(),
        "raw_output_size": 768,
        "output_size": 512,
    }
    assert len(metadata["model_sha256"]) == 64
    assert len(metadata["stat_sha256"]) == 64


def test_clip_navigation_state_is_complete_without_reading_rae_files(tmp_path):
    config = SimpleNamespace(
        MODEL=SimpleNamespace(
            RGB_ENCODER=SimpleNamespace(
                type="clip",
                model_dir=str(tmp_path / "missing-model"),
                stat_path=str(tmp_path / "missing-stat.pt"),
                raw_output_size=512,
                output_size=512,
            )
        )
    )
    policy = _FakePolicy()

    state_dict, metadata = navigation_state_dict(policy, config)

    assert set(state_dict) == set(policy.state_dict())
    assert metadata == {"type": "clip"}


def test_rae_navigation_state_resolves_relative_assets_from_project_root(
    tmp_path, monkeypatch
):
    config, model_path, stat_path = _config(tmp_path)
    config.MODEL.RGB_ENCODER.model_dir = (
        "pretrained/rae_dinov2_with_registers_base"
    )
    config.MODEL.RGB_ENCODER.stat_path = (
        "pretrained/rae_dinov2_with_registers_base/stat.pt"
    )
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    _, metadata = navigation_state_dict(_FakePolicy(), config)

    assert metadata["model_dir"] == (
        "pretrained/rae_dinov2_with_registers_base"
    )
    assert metadata["model_sha256"] == hashlib.sha256(
        model_path.read_bytes()
    ).hexdigest()
    assert metadata["stat_sha256"] == hashlib.sha256(
        stat_path.read_bytes()
    ).hexdigest()


def test_rae_navigation_state_normalizes_project_absolute_model_dir(tmp_path):
    config, model_path, _ = _config(tmp_path)

    _, metadata = navigation_state_dict(_FakePolicy(), config)

    assert metadata["model_dir"] == model_path.parent.relative_to(
        tmp_path
    ).as_posix()


def test_rae_navigation_state_rejects_model_dir_outside_project(
    tmp_path, monkeypatch
):
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setattr(checkpoint_module, "_PROJECT_ROOT", project_root)
    config, _, _ = _config(tmp_path / "external")

    with pytest.raises(ValueError, match="model_dir.*outside.*project root"):
        navigation_state_dict(_FakePolicy(), config)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("raw_output_size", 768.5),
        ("raw_output_size", True),
        ("output_size", 512.5),
        ("output_size", False),
    ),
)
def test_rae_navigation_state_strictly_rejects_non_integer_dimensions(
    tmp_path, field, value
):
    config, _, _ = _config(tmp_path)
    setattr(config.MODEL.RGB_ENCODER, field, value)

    with pytest.raises(ValueError, match=f"{field}.*integer"):
        navigation_state_dict(_FakePolicy(), config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda checkpoint: checkpoint["rgb_encoder"].update(type="clip"),
         "type.*rae_dinov2.*clip"),
        (lambda checkpoint: checkpoint["rgb_encoder"].update(model_sha256="0" * 64),
         "model.*SHA256.*mismatch"),
        (lambda checkpoint: checkpoint["rgb_encoder"].update(stat_sha256="0" * 64),
         "stat.*SHA256.*mismatch"),
        (lambda checkpoint: checkpoint["rgb_encoder"].update(raw_output_size=512),
         "raw_output_size.*768.*512"),
        (lambda checkpoint: checkpoint["rgb_encoder"].update(output_size=768),
         "output_size.*512.*768"),
        (
            lambda checkpoint: checkpoint["state_dict"].__setitem__(
                "net.unrelated.weight",
                checkpoint["state_dict"].pop(
                    "net.vln_bert.img_embeddings.rgb_projection.0.weight"
                ),
            ),
            "RAE/DINOv2.*missing.*rgb_projection.0.weight",
        ),
    ),
)
def test_rae_checkpoint_validation_rejects_incompatible_metadata(
    tmp_path, mutation, message
):
    config, _, _ = _config(tmp_path)
    checkpoint = _rae_checkpoint(_FakePolicy(), config)
    mutation(checkpoint)

    with pytest.raises(ValueError, match=message):
        validate_rgb_checkpoint_metadata(checkpoint, config)


def test_rae_checkpoint_rejects_tampered_model_dir_metadata(tmp_path):
    config, _, _ = _config(tmp_path)
    checkpoint = _rae_checkpoint(_FakePolicy(), config)
    checkpoint["rgb_encoder"]["model_dir"] = "pretrained/tampered-model"

    with pytest.raises(ValueError, match="model_dir.*mismatch"):
        validate_rgb_checkpoint_metadata(checkpoint, config)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("raw_output_size", 768.0),
        ("raw_output_size", True),
        ("output_size", 512.0),
        ("output_size", False),
    ),
)
def test_rae_checkpoint_metadata_dimensions_require_non_boolean_integers(
    tmp_path, field, value
):
    config, _, _ = _config(tmp_path)
    checkpoint = _rae_checkpoint(_FakePolicy(), config)
    checkpoint["rgb_encoder"][field] = value

    with pytest.raises(
        ValueError,
        match=f"{field}.*non-boolean integer",
    ):
        validate_rgb_checkpoint_metadata(checkpoint, config)


@pytest.mark.parametrize("missing_suffix", PROJECTION_PARAMETER_SUFFIXES)
def test_rae_checkpoint_requires_every_projection_parameter(
    tmp_path, missing_suffix
):
    config, _, _ = _config(tmp_path)
    checkpoint = _rae_checkpoint(_FakePolicy(), config)
    semantic_suffix = (
        f"img_embeddings.rgb_projection.{missing_suffix}"
    )
    key = next(
        key
        for key in checkpoint["state_dict"]
        if key.endswith(semantic_suffix)
    )
    del checkpoint["state_dict"][key]

    with pytest.raises(
        ValueError,
        match=rf"missing.*{re.escape(missing_suffix)}",
    ):
        validate_rgb_checkpoint_metadata(checkpoint, config)


def test_rae_checkpoint_does_not_merge_partial_projection_wrappers(tmp_path):
    config, _, _ = _config(tmp_path)
    checkpoint = _checkpoint_with_plain_and_wrapped_projection(config)
    del checkpoint["state_dict"][
        "net.vln_bert.img_embeddings.rgb_projection.4.bias"
    ]
    del checkpoint["state_dict"][
        "net.module.vln_bert.img_embeddings.rgb_projection.0.weight"
    ]

    with pytest.raises(ValueError, match="complete.*rgb_projection"):
        validate_rgb_checkpoint_metadata(checkpoint, config)


def test_rae_checkpoint_accepts_one_complete_projection_wrapper(tmp_path):
    config, _, _ = _config(tmp_path)
    checkpoint = _checkpoint_with_plain_and_wrapped_projection(config)
    del checkpoint["state_dict"][
        "net.module.vln_bert.img_embeddings.rgb_projection.4.bias"
    ]

    assert validate_rgb_checkpoint_metadata(checkpoint, config) is None


@pytest.mark.parametrize("encoder_type", ("clip", "rae_dinov2"))
def test_checkpoint_validation_rejects_empty_state_dict(
    tmp_path, encoder_type
):
    config, _, _ = _config(tmp_path, encoder_type)
    if encoder_type == "rae_dinov2":
        checkpoint = _rae_checkpoint(_FakePolicy(), config)
        checkpoint["state_dict"] = {}
    else:
        checkpoint = {
            "state_dict": {},
            "rgb_encoder": {"type": "clip"},
        }

    with pytest.raises(ValueError, match="state_dict.*empty"):
        validate_rgb_checkpoint_metadata(checkpoint, config)


@pytest.mark.parametrize("trainer_class", (SftTrainer, GrpoTrainer))
def test_trainer_validates_checkpoint_before_accessing_first_state_key(
    trainer_class,
):
    source = inspect.getsource(trainer_class._initialize_policy)

    assert source.index(
        "validate_rgb_checkpoint_metadata(ckpt_dict, config)"
    ) < source.index("list(ckpt_dict['state_dict'].keys())[0]")


def test_rae_checkpoint_requires_metadata(tmp_path):
    config, _, _ = _config(tmp_path)
    checkpoint = {"state_dict": _FakePolicy().state_dict()}

    with pytest.raises(ValueError, match="RAE/DINOv2.*metadata.*missing"):
        validate_rgb_checkpoint_metadata(checkpoint, config)


def test_clip_checkpoint_accepts_legacy_state_but_rejects_rae_metadata_or_mlp(
    tmp_path,
):
    config, _, _ = _config(tmp_path, "clip")
    legacy_checkpoint = {
        "state_dict": {"net.vln_bert.global_encoder.weight": torch.ones(1)}
    }

    assert validate_rgb_checkpoint_metadata(legacy_checkpoint, config) is None

    with pytest.raises(ValueError, match="CLIP.*RAE/DINOv2.*metadata"):
        validate_rgb_checkpoint_metadata(
            {
                **legacy_checkpoint,
                "rgb_encoder": {"type": "rae_dinov2"},
            },
            config,
        )

    with pytest.raises(ValueError, match="CLIP.*RAE/DINOv2.*rgb_projection"):
        validate_rgb_checkpoint_metadata(
            {
                "state_dict": {
                    "net.vln_bert.img_embeddings.rgb_projection.0.weight": (
                        torch.ones(1)
                    )
                },
                "rgb_encoder": {"type": "clip"},
            },
            config,
        )


def test_incompatible_report_only_ignores_rae_backbone_missing_keys(tmp_path):
    config, _, _ = _config(tmp_path)
    incompatible = SimpleNamespace(
        missing_keys=[
            "net.rgb_encoder.backbone.layer.weight",
            "net.module.rgb_encoder.backbone.layer.bias",
            "net.vln_bert.img_embeddings.rgb_projection.4.weight",
            "net.other.weight",
        ],
        unexpected_keys=["net.unexpected.weight"],
    )
    messages = []

    report = report_navigation_incompatible_keys(
        incompatible,
        config,
        print_fn=messages.append,
    )

    assert report["ignored_missing_keys"] == [
        "net.module.rgb_encoder.backbone.layer.bias",
        "net.rgb_encoder.backbone.layer.weight",
    ]
    assert report["missing_keys"] == [
        "net.other.weight",
        "net.vln_bert.img_embeddings.rgb_projection.4.weight",
    ]
    assert report["unexpected_keys"] == ["net.unexpected.weight"]
    rendered = "\n".join(messages)
    assert "net.other.weight" in rendered
    assert "net.vln_bert.img_embeddings.rgb_projection.4.weight" in rendered
    assert "net.unexpected.weight" in rendered
    assert "ignored frozen RAE/DINOv2 backbone" in rendered


def test_grpo_setup_training_parts_keeps_projection_and_backbone_frozen():
    trainer = object.__new__(GrpoTrainer)
    trainer.policy = _FakePolicy()
    trainer.trainable_parts = [trainer.policy.net.vln_bert.global_encoder]

    trainer.setup_training_parts()

    projection = trainer.policy.net.vln_bert.img_embeddings.rgb_projection
    assert all(not parameter.requires_grad for parameter in projection.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in trainer.policy.net.rgb_encoder.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in trainer.policy.net.vln_bert.global_encoder.parameters()
    )


class _StateHolder:
    def state_dict(self):
        return {"value": 1}


class _FakeVectorEnvs:
    def __init__(self, environment_states):
        self.num_envs = len(environment_states)
        self.environment_states = environment_states
        self.calls = []
        self.resume_count = 0

    def resume_all(self):
        self.resume_count += 1

    def call(self, function_names, function_args_list=None):
        self.calls.append((function_names, function_args_list))
        if function_names[0] == "get_episode_iterator_state":
            return self.environment_states
        return [None for _ in function_names]


@pytest.mark.parametrize(
    ("trainer_class", "stage", "iteration", "only_last"),
    (
        (SftTrainer, "IL", 1, True),
        (SftTrainer, "IL", 2, False),
        (GrpoTrainer, "GRPO", 1, True),
        (GrpoTrainer, "GRPO", 2, False),
    ),
)
def test_trainer_save_branches_include_rgb_metadata_and_keep_sft_projection_trainable(
    tmp_path, monkeypatch, trainer_class, stage, iteration, only_last
):
    config, _, _ = _config(tmp_path)
    setattr(config, "CHECKPOINT_FOLDER", str(tmp_path))
    setattr(config, "ONLY_LAST_SAVEALL", only_last)
    setattr(config, stage, SimpleNamespace(iters=2))
    trainer = object.__new__(trainer_class)
    trainer.config = config
    trainer.policy = _FakePolicy()
    trainer.optimizer = _StateHolder()
    trainer.scheduler = _StateHolder()
    captured = {}
    monkeypatch.setattr(
        torch,
        "save",
        lambda *, obj, f: captured.update(obj=obj, path=f),
    )

    trainer.save_checkpoint(iteration)

    assert captured["obj"]["rgb_encoder"]["type"] == "rae_dinov2"
    assert not any(
        "rgb_encoder.backbone" in key
        for key in captured["obj"]["state_dict"]
    )
    assert any(
        "img_embeddings.rgb_projection.0.weight" in key
        for key in captured["obj"]["state_dict"]
    )
    if trainer_class is SftTrainer:
        projection = trainer.policy.net.vln_bert.img_embeddings.rgb_projection
        assert all(
            parameter.requires_grad for parameter in projection.parameters()
        )


def test_resumable_sft_saves_model_and_training_state_separately(
    tmp_path, monkeypatch
):
    config, _, _ = _config(tmp_path)
    config.CHECKPOINT_FOLDER = str(tmp_path)
    config.ONLY_LAST_SAVEALL = True
    config.IL = SimpleNamespace(
        iters=2,
        resumable_checkpoints=True,
        keep_last_train_states=3,
        keep_train_state_every_n_iters=5000,
    )
    trainer = object.__new__(SftTrainer)
    trainer.config = config
    trainer.policy = _FakePolicy()
    trainer.optimizer = _StateHolder()
    trainer.scheduler = _StateHolder()
    trainer.scaler = _StateHolder()
    saved = []
    monkeypatch.setattr(
        sft_trainer_module,
        "atomic_torch_save",
        lambda obj, path: saved.append((obj, path)),
    )
    monkeypatch.setattr(
        sft_trainer_module,
        "prune_training_states",
        lambda *args: [],
    )
    episode_iterator_state = {
        "format_version": 1,
        "world_size": 1,
        "ranks": [],
    }

    trainer.save_checkpoint(
        1,
        episode_iterator_state=episode_iterator_state,
    )

    assert len(saved) == 2
    model, model_path = saved[0]
    training_state, training_state_path = saved[1]
    assert model_path.endswith("ckpt.iter1.pth")
    assert training_state_path.endswith(
        "train_states/train_state.iter1.pth"
    )
    assert "optim_state" not in model
    assert "scheduler_state" not in model
    assert "scaler_state" not in model
    assert training_state["model_checkpoint"] == "ckpt.iter1.pth"
    assert training_state["iteration"] == 1
    assert training_state["optim_state"] == {"value": 1}
    assert training_state["scheduler_state"] == {"value": 1}
    assert training_state["scaler_state"] == {"value": 1}
    assert training_state["format_version"] == 2
    assert (
        training_state["episode_iterator_state"]
        is episode_iterator_state
    )


def test_sft_captures_and_restores_all_local_episode_iterators():
    environment_states = [{"worker": 0}, {"worker": 1}]
    trainer = object.__new__(SftTrainer)
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.envs = _FakeVectorEnvs(environment_states)

    captured = trainer._capture_episode_iterator_state()

    assert trainer.envs.resume_count == 1
    assert captured == {
        "format_version": 1,
        "world_size": 1,
        "ranks": [
            {
                "rank": 0,
                "num_envs": 2,
                "environments": environment_states,
            }
        ],
    }

    trainer._restore_episode_iterator_state(captured)

    function_names, function_args_list = trainer.envs.calls[-1]
    assert function_names == [
        "set_episode_iterator_state",
        "set_episode_iterator_state",
    ]
    assert function_args_list == [
        {"state": environment_states[0]},
        {"state": environment_states[1]},
    ]


def test_sft_episode_restore_rejects_changed_parallelism():
    trainer = object.__new__(SftTrainer)
    trainer.local_rank = 0
    trainer.world_size = 1
    trainer.envs = _FakeVectorEnvs([{"worker": 0}])
    state = {
        "format_version": 1,
        "world_size": 2,
        "ranks": [],
    }

    with pytest.raises(ValueError, match="number of training ranks"):
        trainer._restore_episode_iterator_state(state)


def test_sft_episode_restore_accepts_legacy_empty_snapshot():
    trainer = object.__new__(SftTrainer)
    trainer.local_rank = 0
    trainer.world_size = 2
    trainer.envs = _FakeVectorEnvs([{"worker": 0}])
    state = {
        "format_version": 1,
        "world_size": 2,
        "ranks": [
            {"rank": 0, "num_envs": 0, "environments": []},
            {"rank": 1, "num_envs": 0, "environments": []},
        ],
    }

    trainer._restore_episode_iterator_state(state)

    assert trainer.envs.calls == []
