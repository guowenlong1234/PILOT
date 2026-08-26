from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import PretrainedConfig

from pretrain_src.pretrain_src.model.vilmodel import (
    ImageEmbeddings as PretrainImageEmbeddings,
)
from vlnce_baselines.models import R1Policy as policy_module
from vlnce_baselines.models.etp import ETP_R1_vlnbert_init as init_module
from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import (
    GlocalTextPathNavCMT,
    ImageEmbeddings,
)


ROOT = Path(__file__).resolve().parents[1]


def _image_config(encoder_type="rae_dinov2"):
    return PretrainedConfig(
        rgb_encoder_type=encoder_type,
        image_feat_size=768 if encoder_type == "rae_dinov2" else 512,
        hidden_size=768,
        depth_feat_size=128,
        angle_feat_size=4,
        hidden_dropout_prob=0.0,
        num_pano_layers=0,
        layer_norm_eps=1e-5,
        use_depth_embedding=True,
        obj_feat_size=0,
    )


def _model_config(encoder_type):
    is_rae = encoder_type == "rae_dinov2"
    return SimpleNamespace(
        TORCH_GPU_ID=0,
        spatial_output=False,
        DEPTH_ENCODER=SimpleNamespace(
            cnn_type="VlnResnetDepthEncoder",
            output_size=128,
            ddppo_checkpoint="depth.pth",
            backbone="resnet50",
        ),
        RGB_ENCODER=SimpleNamespace(
            type=encoder_type,
            precision="ambient",
            model_dir="rae-model",
            output_size=768 if is_rae else 512,
            cls_residual_mlp_enabled=is_rae,
            cls_residual_mlp_hidden_dim=768,
            cls_residual_mlp_zero_init=True,
        ),
    )


def _init_config(encoder_type="rae_dinov2", pretrained_path="pretrain.pt"):
    is_rae = encoder_type == "rae_dinov2"
    return SimpleNamespace(
        pretrained_path=pretrained_path,
        RGB_ENCODER=SimpleNamespace(
            type=encoder_type,
            output_size=768 if is_rae else 512,
        ),
        use_depth_embedding=True,
        use_sprels=True,
        fix_lang_embedding=False,
        fix_pano_embedding=False,
    )


def _pretrain_visual_checkpoint(encoder_type="rae_dinov2", prefix="module."):
    input_size = 768 if encoder_type == "rae_dinov2" else 512
    return {
        f"{prefix}bert.img_embeddings.img_linear.weight": torch.randn(
            768,
            input_size,
        ),
        f"{prefix}bert.img_embeddings.img_linear.bias": torch.randn(768),
    }


def _flatten(value, prefix=()):
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            flattened.update(_flatten(child, prefix + (key,)))
        return flattened
    return {prefix: value}


def test_online_image_embeddings_matches_pretrain_direct_interface():
    config = _image_config()
    online = ImageEmbeddings(config)
    offline = PretrainImageEmbeddings(config)

    assert not hasattr(online, "rgb_projection")
    assert not hasattr(offline, "rgb_projection")
    assert online.img_linear.in_features == offline.img_linear.in_features == 768
    assert online.img_linear.out_features == offline.img_linear.out_features == 768


def test_forward_panorama_consumes_raw_768_cls_directly():
    image_embeddings = ImageEmbeddings(_image_config())
    holder = SimpleNamespace(
        img_embeddings=image_embeddings,
        embeddings=SimpleNamespace(
            token_type_embeddings=torch.nn.Embedding(2, 768)
        ),
    )
    linear_inputs = []
    hook = image_embeddings.img_linear.register_forward_pre_hook(
        lambda _module, inputs: linear_inputs.append(inputs[0])
    )

    try:
        outputs = GlocalTextPathNavCMT.forward_panorama(
            holder,
            rgb_fts=torch.randn(2, 12, 768),
            dep_fts=torch.randn(2, 12, 128),
            loc_fts=torch.randn(2, 12, 4),
            nav_types=torch.zeros(2, 12, dtype=torch.long),
            view_lens=torch.tensor([12, 12]),
        )
    finally:
        hook.remove()

    assert linear_inputs[0].shape == (2, 12, 768)
    assert outputs[0].shape == (2, 12, 768)


class _FakeDepthEncoder(torch.nn.Module):
    is_blind = False

    def __init__(self, *_args, **_kwargs):
        super().__init__()

    def forward(self, observations):
        return torch.zeros(observations["depth"].shape[0], 128, 4, 4)


class _FakeClipEncoder(torch.nn.Module):
    is_blind = False

    def __init__(self, device):
        super().__init__()
        self.device_argument = device


class _FakeRaeEncoder(torch.nn.Module):
    is_blind = False
    calls = []

    def __init__(
        self,
        model_dir,
        device,
        precision,
        cls_residual_mlp_enabled,
        cls_residual_mlp_hidden_dim,
        cls_residual_mlp_zero_init,
    ):
        super().__init__()
        self.calls.append(
            (
                model_dir,
                device,
                precision,
                cls_residual_mlp_enabled,
                cls_residual_mlp_hidden_dim,
                cls_residual_mlp_zero_init,
            )
        )


class _ConcreteETP(policy_module.ETP):
    @property
    def perception_embedding_size(self):
        return 1

    @property
    def recurrent_hidden_size(self):
        return 1


@pytest.mark.parametrize(
    ("encoder_type", "expected_class", "expected_size"),
    (
        ("clip", _FakeClipEncoder, 512),
        ("rae_dinov2", _FakeRaeEncoder, 768),
    ),
)
def test_etp_builds_configured_rgb_encoder(
    monkeypatch,
    encoder_type,
    expected_class,
    expected_size,
):
    monkeypatch.setattr(
        policy_module, "get_vlnbert_models", lambda **_kwargs: torch.nn.Module()
    )
    monkeypatch.setattr(policy_module, "VlnResnetDepthEncoder", _FakeDepthEncoder)
    monkeypatch.setattr(policy_module, "CLIPEncoder", _FakeClipEncoder)
    monkeypatch.setattr(policy_module, "RaeDinov2RgbEncoder", _FakeRaeEncoder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _FakeRaeEncoder.calls.clear()

    policy = _ConcreteETP(
        observation_space=None,
        model_config=_model_config(encoder_type),
        num_actions=1,
        dropout_rate=0.0,
    )

    assert isinstance(policy.rgb_encoder, expected_class)
    assert policy.rgb_output_size == expected_size
    if encoder_type == "rae_dinov2":
        assert _FakeRaeEncoder.calls == [
            (
                "rae-model",
                torch.device("cpu"),
                "ambient",
                True,
                768,
                True,
            )
        ]


class _WaypointRgbEncoder(torch.nn.Module):
    is_blind = False

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, observations):
        return torch.ones(observations["rgb"].shape[0], self.output_size)


class _WaypointDepthEncoder(torch.nn.Module):
    is_blind = False

    def forward(self, observations):
        return torch.ones(observations["depth"].shape[0], 128, 4, 4)


class _WaypointPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_inputs = []

    def forward(self, rgb, depth):
        self.rgb_inputs.append(rgb)
        return torch.zeros(1, 120, 12)


def _waypoint_policy(output_size):
    policy = object.__new__(_ConcreteETP)
    torch.nn.Module.__init__(policy)
    policy.device = torch.device("cpu")
    policy.rgb_output_size = output_size
    policy.raenwm_enabled = False
    policy.rgb_encoder = _WaypointRgbEncoder(output_size)
    policy.depth_encoder = _WaypointDepthEncoder()
    policy.space_pool_rgb = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(start_dim=2)
    )
    policy.space_pool_depth = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(start_dim=2)
    )
    policy.pano_img_idxes = policy_module.np.arange(
        12,
        dtype=policy_module.np.int64,
    )
    policy.pano_angle_fts = torch.zeros(12, 4)
    return policy


@pytest.mark.parametrize("output_size", (768, 512))
def test_waypoint_preserves_encoder_dimension_without_projection(
    monkeypatch,
    output_size,
):
    def fake_nms(values, **_kwargs):
        output = torch.zeros_like(values)
        output[:, :, 1, 0] = 1.0
        return output

    monkeypatch.setattr(policy_module, "nms", fake_nms)
    policy = _waypoint_policy(output_size)
    predictor = _WaypointPredictor()
    observations = {
        "rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "depth": torch.zeros(1, 256, 256, 1),
    }
    for heading in range(30, 360, 30):
        observations[f"rgb_{heading}"] = torch.zeros(
            1, 224, 224, 3, dtype=torch.uint8
        )
        observations[f"depth_{heading}"] = torch.zeros(1, 256, 256, 1)

    outputs = policy(
        mode="waypoint",
        observations=observations,
        waypoint_predictor=predictor,
        in_train=False,
    )

    assert predictor.rgb_inputs[0].shape == (12, output_size)
    assert outputs["pano_rgb"].shape == (1, 12, output_size)
    assert all(candidate.shape[-1] == output_size for candidate in outputs["cand_rgb"])


def test_online_default_config_explicitly_preserves_clip():
    from vlnce_baselines.config.default import get_config

    rgb = get_config().MODEL.RGB_ENCODER

    assert rgb.type == "clip"
    assert rgb.output_size == 512
    assert rgb.cls_residual_mlp_enabled is False
    assert rgb.cls_residual_mlp_hidden_dim == 768
    assert rgb.cls_residual_mlp_zero_init is True
    assert "stat_path" not in rgb
    assert "raw_output_size" not in rgb
    assert "projection_hidden_size" not in rgb


def test_vlnbert_receives_direct_768_config_and_normalizes_module_checkpoint(
    monkeypatch,
):
    captured = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured.update(kwargs)
            return kwargs

    checkpoint = _pretrain_visual_checkpoint()
    monkeypatch.setattr(init_module.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        "vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt.GlocalTextPathNavCMT",
        FakeModel,
    )

    init_module.get_vlnbert_models(_init_config(), dropout_rate=0.2)

    config = captured["config"]
    assert config.rgb_encoder_type == "rae_dinov2"
    assert config.image_feat_size == 768
    assert not hasattr(config, "raw_image_feat_size")
    assert not hasattr(config, "projection_hidden_size")
    assert "bert.img_embeddings.img_linear.weight" in captured["state_dict"]


def test_module_checkpoint_loads_direct_img_linear_into_real_model(monkeypatch):
    checkpoint = _pretrain_visual_checkpoint()
    expected_weight = checkpoint[
        "module.bert.img_embeddings.img_linear.weight"
    ].clone()
    monkeypatch.setattr(init_module.torch, "load", lambda *_args, **_kwargs: checkpoint)

    model = init_module.get_vlnbert_models(_init_config())

    torch.testing.assert_close(model.img_embeddings.img_linear.weight, expected_weight)
    assert not hasattr(model.img_embeddings, "rgb_projection")


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    (
        (
            {
                "module.bert.img_embeddings.rgb_projection.0.weight": torch.ones(1),
                **_pretrain_visual_checkpoint(prefix="module."),
            },
            "retired 768->512 rgb_projection",
        ),
        (
            {
                "module.bert.img_embeddings.img_linear.weight": torch.ones(768, 512),
                "module.bert.img_embeddings.img_linear.bias": torch.ones(768),
            },
            "img_linear.weight.*\(768, 768\).*\(768, 512\)",
        ),
        ({}, "missing.*img_linear.weight"),
    ),
)
def test_rae_pretrained_checkpoint_guard_rejects_old_interface(
    monkeypatch,
    checkpoint,
    message,
):
    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return kwargs

    monkeypatch.setattr(init_module.torch, "load", lambda *_args, **_kwargs: checkpoint)
    monkeypatch.setattr(
        "vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt.GlocalTextPathNavCMT",
        FakeModel,
    )

    with pytest.raises(ValueError, match=message):
        init_module.get_vlnbert_models(_init_config())


@pytest.mark.parametrize(
    ("original_path", "rae_path", "task_name", "hfov"),
    (
        ("run_r2r/iter_train.yaml", "run_r2r/iter_train_rae_dino.yaml", "r2r", 90),
        ("run_rxr/iter_train.yaml", "run_rxr/iter_train_rae_dino.yaml", "rxr", 63),
    ),
)
def test_rae_yaml_selects_etpnav_768_pipeline(
    original_path,
    rae_path,
    task_name,
    hfov,
):
    original = yaml.safe_load((ROOT / original_path).read_text(encoding="utf-8"))
    rae = yaml.safe_load((ROOT / rae_path).read_text(encoding="utf-8"))
    changed = {
        key
        for key in set(_flatten(original)) | set(_flatten(rae))
        if _flatten(original).get(key) != _flatten(rae).get(key)
    }
    allowed = {
        ("TENSORBOARD_DIR",),
        ("CHECKPOINT_FOLDER",),
        ("RESULTS_DIR",),
        ("MODEL", "pretrained_path"),
        ("MODEL", "RGB_ENCODER", "type"),
        ("MODEL", "RGB_ENCODER", "precision"),
        ("MODEL", "RGB_ENCODER", "model_dir"),
        ("MODEL", "RGB_ENCODER", "output_size"),
        ("MODEL", "RGB_ENCODER", "cls_residual_mlp_enabled"),
        ("MODEL", "RGB_ENCODER", "cls_residual_mlp_hidden_dim"),
        ("MODEL", "RGB_ENCODER", "cls_residual_mlp_zero_init"),
    }
    if task_name == "r2r":
        allowed.update(
            {
                ("IL", "checkpoint_sync_enabled"),
                ("IL", "checkpoint_sync_destination"),
                ("EVAL", "checkpoint_order"),
            }
        )
    assert changed == allowed
    assert rae["MODEL"]["RGB_ENCODER"] == {
        "type": "rae_dinov2",
        "precision": "ambient",
        "model_dir": "pretrained/rae_dinov2_with_registers_base",
        "output_size": 768,
        "cls_residual_mlp_enabled": True,
        "cls_residual_mlp_hidden_dim": 768,
        "cls_residual_mlp_zero_init": True,
    }
    assert "rae_dinov2_etpnav_cls_768" in rae["MODEL"]["pretrained_path"]
    assert f"rae_dinov2_etpnav_cls_768/{task_name}" in rae["TENSORBOARD_DIR"]

    task_config = yaml.safe_load(
        (ROOT / rae["BASE_TASK_CONFIG_PATH"]).read_text(encoding="utf-8")
    )
    assert task_config["SIMULATOR"]["RGB_SENSOR"]["HFOV"] == hfov
    assert task_config["SIMULATOR"]["DEPTH_SENSOR"]["HFOV"] == hfov
