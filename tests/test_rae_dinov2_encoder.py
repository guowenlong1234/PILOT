from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.models.encoders import rae_dinov2_encoder as encoder_module
from vlnce_baselines.models.encoders.rae_dinov2_encoder import (
    ClsResidualMlp,
    RaeDinov2RgbEncoder,
    prepare_rae_rgb_tensor,
    resolve_rae_compute_dtype,
)


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.projection = torch.nn.Linear(768, 768, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(768))
        self.layernorm = torch.nn.LayerNorm(768)
        self.last_pixels = None
        self.forward_grad_enabled = None
        self.last_hidden_dtype = None
        self.output_hidden_states = None

    def forward(self, pixel_values, output_hidden_states=False):
        self.last_pixels = pixel_values.detach().clone()
        self.forward_grad_enabled = torch.is_grad_enabled()
        self.output_hidden_states = output_hidden_states
        cls = self.scale * torch.ones(
            pixel_values.shape[0],
            768,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        cls = self.projection(cls)
        self.last_hidden_dtype = cls.dtype
        other_tokens = torch.zeros(
            pixel_values.shape[0],
            260,
            768,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat((cls.unsqueeze(1), other_tokens), dim=1)
        )


@pytest.fixture
def encoder_factory(monkeypatch, tmp_path):
    created = []

    def make(**kwargs):
        backbone = FakeBackbone()
        model_calls = []
        processor_calls = []

        def load_model(model_dir, **load_kwargs):
            model_calls.append((model_dir, load_kwargs))
            return backbone

        def load_processor(model_dir, **load_kwargs):
            processor_calls.append((model_dir, load_kwargs))
            return SimpleNamespace(
                image_mean=[0.5, 0.25, 0.0],
                image_std=[0.5, 0.25, 1.0],
            )

        monkeypatch.setattr(
            encoder_module.Dinov2WithRegistersModel,
            "from_pretrained",
            staticmethod(load_model),
        )
        monkeypatch.setattr(
            encoder_module.AutoImageProcessor,
            "from_pretrained",
            staticmethod(load_processor),
        )
        encoder = RaeDinov2RgbEncoder(
            model_dir=tmp_path / "model",
            device=torch.device("cpu"),
            **kwargs,
        )
        encoder._test_model_calls = model_calls
        encoder._test_processor_calls = processor_calls
        created.append(encoder)
        return encoder

    return make


def test_encoder_loads_only_local_assets_and_disables_layernorm_affine(
    encoder_factory,
):
    encoder = encoder_factory()

    assert encoder._test_model_calls[0][1] == {"local_files_only": True}
    assert encoder._test_processor_calls[0][1] == {"local_files_only": True}
    assert encoder.backbone.layernorm.elementwise_affine is False
    assert encoder.backbone.layernorm.weight is None
    assert encoder.backbone.layernorm.bias is None
    assert not hasattr(encoder, "latent_mean")
    assert not hasattr(encoder, "latent_var")


def test_encoder_returns_raw_cls_without_rae_statistics(encoder_factory):
    encoder = encoder_factory()
    rgb = torch.zeros(2, 224, 224, 3, dtype=torch.uint8)
    rgb[..., 0] = 255
    rgb[..., 1] = 128

    actual = encoder({"rgb": rgb})

    expected_pixels = torch.empty(2, 3, 224, 224)
    expected_pixels[:, 0] = 1.0
    expected_pixels[:, 1] = (128.0 / 255.0 - 0.25) / 0.25
    expected_pixels[:, 2] = 0.0
    torch.testing.assert_close(encoder.backbone.last_pixels, expected_pixels)
    torch.testing.assert_close(actual, torch.full((2, 768), 2.0))
    assert encoder.backbone.output_hidden_states is True
    assert actual.dtype == torch.float32


def test_encoder_returns_register_free_patch_map_without_changing_cls(
    encoder_factory,
):
    encoder = encoder_factory()
    observations = {
        "rgb": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
    }

    cls, patch_latents = encoder.forward_with_patch_latents(observations)

    torch.testing.assert_close(cls, torch.full((2, 768), 2.0))
    assert patch_latents.shape == (2, 768, 16, 16)
    assert patch_latents.dtype == torch.float32
    assert torch.count_nonzero(patch_latents) == 0
    assert not hasattr(encoder, "latent_mean")
    assert not hasattr(encoder, "latent_var")


def test_encoder_exposes_raw_cls_nav_cls_and_raw_patch_separately(
    encoder_factory,
):
    encoder = encoder_factory(
        cls_residual_mlp_enabled=True,
        cls_residual_mlp_hidden_dim=768,
        cls_residual_mlp_zero_init=True,
    )
    with torch.no_grad():
        encoder.cls_residual_mlp.layers[-1].bias.fill_(0.25)
    observations = {
        "rgb": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
    }

    raw_cls, nav_cls, raw_patch = (
        encoder.forward_with_raw_cls_and_patch_latents(observations)
    )

    torch.testing.assert_close(raw_cls, torch.full((2, 768), 2.0))
    torch.testing.assert_close(nav_cls, torch.full((2, 768), 2.25))
    assert raw_patch.shape == (2, 768, 16, 16)
    assert raw_cls.requires_grad is False
    assert nav_cls.requires_grad is True


def test_prepare_rgb_matches_etpnav_layout_range_and_resize_rules():
    bhwc = torch.full((2, 112, 112, 4), 255, dtype=torch.uint8)
    prepared = prepare_rae_rgb_tensor(bhwc, torch.device("cpu"), size=224)

    assert prepared.shape == (2, 3, 224, 224)
    assert prepared.dtype == torch.float32
    torch.testing.assert_close(prepared, torch.ones_like(prepared))

    chw_minus_one_to_one = torch.zeros(3, 224, 224).sub_(1)
    prepared = prepare_rae_rgb_tensor(
        chw_minus_one_to_one,
        torch.device("cpu"),
    )
    torch.testing.assert_close(prepared, torch.zeros_like(prepared))


def test_zero_initialized_residual_mlp_starts_as_identity(encoder_factory):
    encoder = encoder_factory(
        cls_residual_mlp_enabled=True,
        cls_residual_mlp_hidden_dim=768,
        cls_residual_mlp_zero_init=True,
    )
    features = torch.randn(3, 768)

    actual = encoder({"rgb_features": features})

    torch.testing.assert_close(actual, features)
    assert isinstance(encoder.cls_residual_mlp, ClsResidualMlp)
    layers = encoder.cls_residual_mlp.layers
    assert [
        (layer.in_features, layer.out_features)
        for layer in layers
        if isinstance(layer, torch.nn.Linear)
    ] == [(768, 768), (768, 768), (768, 768)]
    assert torch.count_nonzero(layers[-1].weight) == 0
    assert torch.count_nonzero(layers[-1].bias) == 0


def test_backbone_is_frozen_but_residual_mlp_is_trainable(encoder_factory):
    encoder = encoder_factory(cls_residual_mlp_enabled=True)
    encoder.train(True)

    output = encoder(
        {"rgb": torch.zeros(2, 224, 224, 3, dtype=torch.uint8)}
    )
    output.square().mean().backward()

    assert encoder.training
    assert not encoder.backbone.training
    assert encoder.backbone.forward_grad_enabled is False
    assert all(not parameter.requires_grad for parameter in encoder.backbone.parameters())
    assert all(parameter.grad is None for parameter in encoder.backbone.parameters())
    assert all(
        parameter.requires_grad
        for parameter in encoder.cls_residual_mlp.parameters()
    )
    assert encoder.cls_residual_mlp.layers[-1].weight.grad is not None
    assert torch.count_nonzero(
        encoder.cls_residual_mlp.layers[-1].weight.grad
    ) > 0


def test_disabled_residual_path_has_no_trainable_encoder_parameters(
    encoder_factory,
):
    encoder = encoder_factory()
    with torch.enable_grad():
        output = encoder(
            {"rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8)}
        )

    assert output.requires_grad is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_encoder_follows_etpnav_outer_autocast_by_default(encoder_factory):
    encoder = encoder_factory()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(
            {"rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8)}
        )

    assert encoder.backbone.last_hidden_dtype == torch.bfloat16
    assert output.dtype == torch.float32


def test_encoder_can_explicitly_disable_outer_autocast(encoder_factory):
    encoder = encoder_factory(precision="float32")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = encoder(
            {"rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8)}
        )

    assert encoder.backbone.last_hidden_dtype == torch.float32
    assert output.dtype == torch.float32


def test_encoder_can_run_backbone_in_bf16_and_return_float32(encoder_factory):
    encoder = encoder_factory(precision="bf16")

    output = encoder(
        {"rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8)}
    )

    assert encoder.backbone.last_hidden_dtype == torch.bfloat16
    assert encoder.backbone.scale.dtype == torch.bfloat16
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("observations", "message"),
    (
        ({}, "rgb.*rgb_features"),
        ({"rgb_features": torch.ones(2, 767)}, "last dimension.*768"),
        (
            {"rgb_features": torch.full((2, 768), float("nan"))},
            "NaN or infinity",
        ),
    ),
)
def test_encoder_rejects_invalid_observations(
    encoder_factory,
    observations,
    message,
):
    with pytest.raises((ValueError, FloatingPointError), match=message):
        encoder_factory()(observations)


def test_encoder_rejects_unknown_compute_precision():
    assert resolve_rae_compute_dtype("ambient") is None
    with pytest.raises(ValueError, match="Unsupported RAE/DINOv2 precision"):
        resolve_rae_compute_dtype("float8")


def test_encoder_is_never_blind(encoder_factory):
    assert encoder_factory().is_blind is False
