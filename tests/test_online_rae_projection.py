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
    is_rae = encoder_type == "rae_dinov2"
    return PretrainedConfig(
        rgb_encoder_type=encoder_type,
        raw_image_feat_size=768 if is_rae else 512,
        image_feat_size=512,
        projection_hidden_size=768,
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
            model_dir="rae-model",
            stat_path="rae-stat.pt",
            raw_output_size=768 if encoder_type == "rae_dinov2" else 512,
            output_size=512,
            projection_hidden_size=768,
        ),
    )


def _init_config(encoder_type="rae_dinov2", pretrained_path="pretrain.pt"):
    return SimpleNamespace(
        pretrained_path=pretrained_path,
        RGB_ENCODER=SimpleNamespace(
            type=encoder_type,
            raw_output_size=768 if encoder_type == "rae_dinov2" else 512,
            output_size=512,
            projection_hidden_size=768,
        ),
        use_depth_embedding=True,
        use_sprels=True,
        fix_lang_embedding=False,
        fix_pano_embedding=False,
    )


def _module_projection_checkpoint():
    projection = PretrainImageEmbeddings(_image_config()).rgb_projection
    return {
        f"module.bert.img_embeddings.rgb_projection.{name}": value
        for name, value in projection.state_dict().items()
    }


def _flatten(value, prefix=()):
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            flattened.update(_flatten(child, prefix + (key,)))
        return flattened
    return {prefix: value}


def test_online_image_embeddings_matches_pretrain_projection_structure():
    config = _image_config()
    online = ImageEmbeddings(config)
    offline = PretrainImageEmbeddings(config)

    assert list(online.rgb_projection.state_dict()) == list(
        offline.rgb_projection.state_dict()
    )
    assert [
        (layer.in_features, layer.out_features)
        for layer in online.rgb_projection
        if isinstance(layer, torch.nn.Linear)
    ] == [(768, 768), (768, 768), (768, 512)]

    projected = online.project_rgb(torch.randn(4, 768))
    assert projected.shape == (4, 512)
    assert torch.isfinite(projected).all()


def test_online_clip_projection_is_identity_for_legacy_dimensions():
    config = _image_config("clip")
    del config.rgb_encoder_type
    del config.raw_image_feat_size
    del config.projection_hidden_size
    module = ImageEmbeddings(config)
    features = torch.randn(4, 512)

    assert isinstance(module.rgb_projection, torch.nn.Identity)
    assert module.project_rgb(features) is features
    assert not list(module.rgb_projection.parameters())


def test_forward_panorama_consumes_512_without_projecting_again():
    image_embeddings = ImageEmbeddings(_image_config())
    holder = SimpleNamespace(
        img_embeddings=image_embeddings,
        embeddings=SimpleNamespace(
            token_type_embeddings=torch.nn.Embedding(2, 768)
        ),
    )
    projection_calls = []
    linear_inputs = []
    projection_hook = image_embeddings.rgb_projection.register_forward_hook(
        lambda *_args: projection_calls.append(True)
    )
    linear_hook = image_embeddings.img_linear.register_forward_pre_hook(
        lambda _module, inputs: linear_inputs.append(inputs[0])
    )

    try:
        outputs = GlocalTextPathNavCMT.forward_panorama(
            holder,
            rgb_fts=torch.randn(2, 12, 512),
            dep_fts=torch.randn(2, 12, 128),
            loc_fts=torch.randn(2, 12, 4),
            nav_types=torch.zeros(2, 12, dtype=torch.long),
            view_lens=torch.tensor([12, 12]),
        )
    finally:
        projection_hook.remove()
        linear_hook.remove()

    assert projection_calls == []
    assert len(linear_inputs) == 1
    assert linear_inputs[0].shape == (2, 12, 512)
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

    def __init__(self, model_dir, stat_path, device):
        super().__init__()
        self.backbone_weight = torch.nn.Parameter(
            torch.ones(1), requires_grad=False
        )
        self.calls.append((model_dir, stat_path, device))


class _ConcreteETP(policy_module.ETP):
    @property
    def perception_embedding_size(self):
        return 1

    @property
    def recurrent_hidden_size(self):
        return 1


@pytest.mark.parametrize(
    ("encoder_type", "expected_class"),
    (("clip", _FakeClipEncoder), ("rae_dinov2", _FakeRaeEncoder)),
)
def test_etp_builds_configured_rgb_encoder(
    monkeypatch, encoder_type, expected_class
):
    monkeypatch.setattr(
        policy_module, "get_vlnbert_models", lambda **_kwargs: torch.nn.Module()
    )
    monkeypatch.setattr(
        policy_module, "VlnResnetDepthEncoder", _FakeDepthEncoder
    )
    monkeypatch.setattr(policy_module, "CLIPEncoder", _FakeClipEncoder)
    monkeypatch.setattr(
        policy_module,
        "RaeDinov2ClsEncoder",
        _FakeRaeEncoder,
        raising=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _FakeRaeEncoder.calls.clear()

    policy = _ConcreteETP(
        observation_space=None,
        model_config=_model_config(encoder_type),
        num_actions=1,
        dropout_rate=0.0,
    )

    assert isinstance(policy.rgb_encoder, expected_class)
    if encoder_type == "rae_dinov2":
        assert _FakeRaeEncoder.calls == [
            ("rae-model", "rae-stat.pt", torch.device("cpu"))
        ]


def test_etp_rejects_unknown_rgb_encoder(monkeypatch):
    monkeypatch.setattr(
        policy_module, "get_vlnbert_models", lambda **_kwargs: torch.nn.Module()
    )
    monkeypatch.setattr(
        policy_module, "VlnResnetDepthEncoder", _FakeDepthEncoder
    )
    monkeypatch.setattr(policy_module, "CLIPEncoder", _FakeClipEncoder)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="Unsupported RGB encoder.*mystery"):
        _ConcreteETP(
            observation_space=None,
            model_config=_model_config("mystery"),
            num_actions=1,
            dropout_rate=0.0,
        )


class _WaypointRgbEncoder(torch.nn.Module):
    is_blind = False

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        self.frozen_weight = torch.nn.Parameter(
            torch.ones(1), requires_grad=False
        )

    def forward(self, observations):
        count = observations["rgb"].shape[0]
        return torch.ones(count, self.output_size)


class _WaypointDepthEncoder(torch.nn.Module):
    is_blind = False

    def forward(self, observations):
        count = observations["depth"].shape[0]
        return torch.ones(count, 128, 4, 4)


class _ProjectionHolder(torch.nn.Module):
    def __init__(self, image_embeddings):
        super().__init__()
        self.img_embeddings = image_embeddings


class _WaypointPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_inputs = []

    def forward(self, rgb, depth):
        self.rgb_inputs.append(rgb)
        assert depth.shape == (12, 128, 4, 4)
        return torch.zeros(1, 120, 12)


def _waypoint_policy(encoder_type):
    policy = object.__new__(_ConcreteETP)
    torch.nn.Module.__init__(policy)
    raw_size = 768 if encoder_type == "rae_dinov2" else 512
    policy.device = torch.device("cpu")
    policy.rgb_encoder = _WaypointRgbEncoder(raw_size)
    policy.depth_encoder = _WaypointDepthEncoder()
    policy.vln_bert = _ProjectionHolder(ImageEmbeddings(_image_config(encoder_type)))
    policy.space_pool_rgb = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(start_dim=2)
    )
    policy.space_pool_depth = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(start_dim=2)
    )
    policy.pano_img_idxes = policy_module.np.arange(12, dtype=policy_module.np.int64)
    policy.pano_angle_fts = torch.zeros(12, 4)
    return policy


@pytest.mark.parametrize("encoder_type", ("rae_dinov2", "clip"))
def test_waypoint_projects_once_and_returns_only_512_dim_features(
    monkeypatch, encoder_type
):
    def fake_nms(values, **_kwargs):
        output = torch.zeros_like(values)
        output[:, :, 1, 0] = 1.0
        return output

    monkeypatch.setattr(policy_module, "nms", fake_nms)
    policy = _waypoint_policy(encoder_type)
    predictor = _WaypointPredictor()
    projection_calls = []
    hook = policy.vln_bert.img_embeddings.rgb_projection.register_forward_hook(
        lambda *_args: projection_calls.append(True)
    )
    observations = {
        "rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "depth": torch.zeros(1, 256, 256, 1),
    }

    try:
        outputs = policy(
            mode="waypoint",
            observations=observations,
            waypoint_predictor=predictor,
            in_train=False,
        )
    finally:
        hook.remove()

    assert len(projection_calls) == 1
    assert len(predictor.rgb_inputs) == 1
    assert predictor.rgb_inputs[0].shape == (12, 512)
    assert outputs["pano_rgb"].shape == (1, 12, 512)
    assert all(candidate.shape[-1] == 512 for candidate in outputs["cand_rgb"])
    assert all(not p.requires_grad for p in policy.rgb_encoder.parameters())
    projection_parameters = list(
        policy.vln_bert.img_embeddings.rgb_projection.parameters()
    )
    if encoder_type == "rae_dinov2":
        assert projection_parameters
        assert all(p.requires_grad for p in projection_parameters)
    else:
        assert projection_parameters == []


def test_online_default_config_explicitly_preserves_clip():
    from vlnce_baselines.config.default import get_config

    rgb = get_config().MODEL.RGB_ENCODER

    assert rgb.cnn_type == "TorchVisionResNet50"
    assert rgb.type == "clip"
    assert rgb.model_dir == ""
    assert rgb.stat_path == ""
    assert rgb.raw_output_size == 512
    assert rgb.output_size == 512
    assert rgb.projection_hidden_size == 768


def test_vlnbert_receives_rgb_config_and_normalizes_module_checkpoint(
    monkeypatch
):
    captured = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured.update(kwargs)
            return kwargs

    monkeypatch.setattr(
        init_module.torch,
        "load",
        lambda *_args, **_kwargs: _module_projection_checkpoint(),
    )
    monkeypatch.setattr(
        "vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt.GlocalTextPathNavCMT",
        FakeModel,
    )

    init_module.get_vlnbert_models(_init_config(), dropout_rate=0.2)

    config = captured["config"]
    assert config.rgb_encoder_type == "rae_dinov2"
    assert config.raw_image_feat_size == 768
    assert config.image_feat_size == 512
    assert config.projection_hidden_size == 768
    assert "bert.img_embeddings.rgb_projection.0.weight" in captured["state_dict"]


def test_module_checkpoint_keys_are_canonicalized_once_and_load_real_model(
    monkeypatch
):
    real_model_class = GlocalTextPathNavCMT
    projection_weight = torch.full((768, 768), 0.125)
    sap_weight = torch.full((1, 1536), -0.25)
    sap_bias = torch.full((1,), -0.75)
    checkpoint = _module_projection_checkpoint()
    checkpoint.update({
        "module.bert.img_embeddings.rgb_projection.0.weight": projection_weight,
        "module.bert.global_sap_head.net.4.weight": sap_weight,
        "module.global_sap_head.net.4.bias": sap_bias,
    })
    captured = {}

    class InspectingModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured["state_dict"] = kwargs["state_dict"]
            model, loading_info = real_model_class.from_pretrained(
                output_loading_info=True,
                **kwargs,
            )
            captured["loading_info"] = loading_info
            return model

    monkeypatch.setattr(
        init_module.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt.GlocalTextPathNavCMT",
        InspectingModel,
    )

    model = init_module.get_vlnbert_models(_init_config())

    expected_keys = {
        key.removeprefix("module.")
        for key in _module_projection_checkpoint()
    } | {
        "bert.global_sap_head.net.4.weight",
        "bert.global_sap_head.net.4.bias",
    }
    assert set(captured["state_dict"]) == expected_keys
    assert not any(
        key.startswith("module.")
        or "bert.module." in key
        or key.startswith("bert.bert.")
        for key in captured["state_dict"]
    )
    torch.testing.assert_close(
        model.img_embeddings.rgb_projection[0].weight,
        projection_weight,
    )
    torch.testing.assert_close(model.global_sap_head.net[4].weight, sap_weight)
    torch.testing.assert_close(model.global_sap_head.net[4].bias, sap_bias)

    relevant_unexpected = [
        key
        for key in captured["loading_info"]["unexpected_keys"]
        if "rgb_projection" in key
        or "global_sap_head" in key
        or "module." in key
    ]
    assert relevant_unexpected == []


def test_clip_checkpoint_without_rae_projection_still_loads(monkeypatch):
    captured = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            captured.update(kwargs)
            return kwargs

    checkpoint = {
        "bert.embeddings.word_embeddings.weight": torch.ones(2, 2)
    }
    monkeypatch.setattr(
        init_module.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        "vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt.GlocalTextPathNavCMT",
        FakeModel,
    )

    init_module.get_vlnbert_models(_init_config("clip"))

    assert captured["state_dict"] == checkpoint
    assert captured["config"].rgb_encoder_type == "clip"


@pytest.mark.parametrize(
    ("encoder_type", "checkpoint", "message"),
    (
        ("rae_dinov2", {}, "RAE/DINOv2.*missing.*rgb_projection.0.weight"),
        (
            "clip",
            {"bert.img_embeddings.rgb_projection.0.weight": torch.ones(1)},
            "CLIP.*RAE/DINOv2.*rgb_projection.0.weight",
        ),
        (
            "clip",
            {"module.bert.img_embeddings.rgb_projection.0.weight": torch.ones(1)},
            "CLIP.*RAE/DINOv2.*rgb_projection.0.weight",
        ),
    ),
)
def test_pretrained_checkpoint_type_guard(
    monkeypatch, encoder_type, checkpoint, message
):
    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return kwargs

    monkeypatch.setattr(
        init_module.torch, "load", lambda *_args, **_kwargs: checkpoint
    )
    monkeypatch.setattr(
        "vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt.GlocalTextPathNavCMT",
        FakeModel,
    )

    with pytest.raises(ValueError, match=message):
        init_module.get_vlnbert_models(_init_config(encoder_type))


def test_rae_pretrained_projection_guard_rejects_extra_projection_key():
    checkpoint = _module_projection_checkpoint()
    checkpoint[
        "module.bert.img_embeddings.rgb_projection.extra.weight"
    ] = torch.ones(1)
    normalized = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }

    with pytest.raises(
        ValueError,
        match=r"missing=\[\].*extra=.*extra\.weight",
    ):
        init_module._validate_pretrain_rgb_projection(
            normalized,
            "rae_dinov2",
        )


@pytest.mark.parametrize(
    ("original_path", "rae_path", "task_name", "hfov"),
    (
        (
            "run_r2r/iter_train.yaml",
            "run_r2r/iter_train_rae_dino.yaml",
            "r2r",
            90,
        ),
        (
            "run_rxr/iter_train.yaml",
            "run_rxr/iter_train_rae_dino.yaml",
            "rxr",
            63,
        ),
    ),
)
def test_rae_yaml_only_changes_allowed_fields(
    original_path, rae_path, task_name, hfov
):
    original = yaml.safe_load((ROOT / original_path).read_text(encoding="utf-8"))
    rae = yaml.safe_load((ROOT / rae_path).read_text(encoding="utf-8"))
    original_flat = _flatten(original)
    rae_flat = _flatten(rae)

    allowed = {
        ("TENSORBOARD_DIR",),
        ("CHECKPOINT_FOLDER",),
        ("RESULTS_DIR",),
        ("MODEL", "pretrained_path"),
        ("MODEL", "RGB_ENCODER", "type"),
        ("MODEL", "RGB_ENCODER", "precision"),
        ("MODEL", "RGB_ENCODER", "model_dir"),
        ("MODEL", "RGB_ENCODER", "stat_path"),
        ("MODEL", "RGB_ENCODER", "raw_output_size"),
        ("MODEL", "RGB_ENCODER", "projection_hidden_size"),
    }
    changed = {
        key
        for key in set(original_flat) | set(rae_flat)
        if original_flat.get(key) != rae_flat.get(key)
    }
    assert changed == allowed
    assert rae["MODEL"]["RGB_ENCODER"] == {
        "type": "rae_dinov2",
        "precision": "float32",
        "model_dir": "pretrained/rae_dinov2_with_registers_base",
        "stat_path": "pretrained/rae_dinov2_with_registers_base/stat.pt",
        "raw_output_size": 768,
        "output_size": 512,
        "projection_hidden_size": 768,
    }
    assert rae["MODEL"]["pretrained_path"] == (
        "pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/best/"
        "model_best_step_452500.pt"
    )
    assert rae["TENSORBOARD_DIR"] == (
        f"data/logs/rae_dinov2/{task_name}/tensorboard/"
    )
    assert rae["CHECKPOINT_FOLDER"] == (
        f"data/logs/rae_dinov2/{task_name}/checkpoints/"
    )
    assert rae["RESULTS_DIR"] == (
        f"data/logs/rae_dinov2/{task_name}/results/"
    )

    task_config = yaml.safe_load(
        (ROOT / rae["BASE_TASK_CONFIG_PATH"]).read_text(encoding="utf-8")
    )
    assert task_config["SIMULATOR"]["RGB_SENSOR"]["HFOV"] == hfov
    assert task_config["SIMULATOR"]["DEPTH_SENSOR"]["HFOV"] == hfov
