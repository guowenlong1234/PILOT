from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.models.encoders import rae_dinov2_encoder as encoder_module
from vlnce_baselines.models.encoders.rae_dinov2_encoder import (
    RaeDinov2ClsEncoder,
    normalize_rae_cls,
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

    def forward(self, pixel_values):
        self.last_pixels = pixel_values.detach().clone()
        self.forward_grad_enabled = torch.is_grad_enabled()
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
            5,
            768,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return SimpleNamespace(
            last_hidden_state=torch.cat((cls.unsqueeze(1), other_tokens), dim=1)
        )


@pytest.fixture
def fake_encoder(monkeypatch, tmp_path):
    backbone = FakeBackbone()
    model_calls = []
    processor_calls = []

    def load_model(model_dir, **kwargs):
        model_calls.append((model_dir, kwargs))
        return backbone

    def load_processor(model_dir, **kwargs):
        processor_calls.append((model_dir, kwargs))
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

    stat_path = tmp_path / "stat.pt"
    torch.save(
        {
            "mean": None,
            "var": torch.ones(768, 2, 2),
        },
        stat_path,
    )
    encoder = RaeDinov2ClsEncoder(
        model_dir=tmp_path / "model",
        stat_path=stat_path,
        device=torch.device("cpu"),
    )
    encoder._test_model_calls = model_calls
    encoder._test_processor_calls = processor_calls
    return encoder


def test_spatial_variance_is_averaged_for_cls():
    cls = torch.tensor([[2.0, 4.0]])
    var = torch.tensor([[[1.0, 3.0]], [[4.0, 12.0]]])

    actual = normalize_rae_cls(cls, mean=None, var=var, eps=0.0)

    expected = cls / torch.sqrt(torch.tensor([[2.0, 8.0]]))
    torch.testing.assert_close(actual, expected)


def test_spatial_mean_and_eps_are_applied_for_cls():
    cls = torch.tensor([[3.0, 7.0]])
    mean = torch.tensor([[[1.0, 3.0]], [[3.0, 5.0]]])
    var = torch.tensor([[[2.0, 2.0]], [[5.0, 7.0]]])

    actual = normalize_rae_cls(cls, mean=mean, var=var, eps=2.0)

    expected = (cls - torch.tensor([[2.0, 4.0]])) / torch.sqrt(
        torch.tensor([[4.0, 8.0]])
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("mean", "var", "message"),
    (
        (None, torch.ones(3, 2, 2), "var"),
        (torch.ones(3), torch.ones(2), "mean"),
        (None, torch.ones(2, 2), "var"),
    ),
)
def test_normalize_rae_cls_rejects_incompatible_stat_shape(mean, var, message):
    with pytest.raises(ValueError, match=rf"{message}.*shape"):
        normalize_rae_cls(torch.ones(1, 2), mean=mean, var=var)


def test_normalize_rae_cls_rejects_non_finite_output():
    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        normalize_rae_cls(
            torch.ones(1, 2),
            mean=None,
            var=torch.zeros(2),
            eps=0.0,
        )


def test_normalize_rae_cls_rejects_infinite_variance_before_normalizing():
    with pytest.raises(ValueError, match=r"var.*finite"):
        normalize_rae_cls(
            torch.ones(1, 2),
            mean=None,
            var=torch.tensor([float("inf"), 1.0]),
        )


def test_normalize_rae_cls_rejects_non_finite_mean_explicitly():
    with pytest.raises(ValueError, match=r"mean.*finite"):
        normalize_rae_cls(
            torch.ones(1, 2),
            mean=torch.tensor([float("nan"), 0.0]),
            var=torch.ones(2),
        )


def test_normalize_rae_cls_rejects_negative_variance_even_with_positive_eps():
    with pytest.raises(ValueError, match=r"var.*non-negative"):
        normalize_rae_cls(
            torch.ones(1, 2),
            mean=None,
            var=torch.tensor([-1e-6, 1.0]),
            eps=1e-5,
        )


def test_normalize_rae_cls_rejects_negative_variance_before_spatial_mean():
    var = torch.tensor([[[-1.0, 3.0]], [[1.0, 1.0]]])

    with pytest.raises(ValueError, match=r"var.*non-negative"):
        normalize_rae_cls(torch.ones(1, 2), mean=None, var=var)


def test_encoder_loads_local_assets_and_disables_final_layernorm_affine(
    fake_encoder,
):
    assert fake_encoder._test_model_calls[0][1] == {"local_files_only": True}
    assert fake_encoder._test_processor_calls[0][1] == {"local_files_only": True}
    assert fake_encoder.backbone.layernorm.elementwise_affine is False
    assert fake_encoder.backbone.layernorm.weight is None
    assert fake_encoder.backbone.layernorm.bias is None


def test_encoder_preprocesses_uint8_bhwc_and_returns_normalized_cls(fake_encoder):
    rgb = torch.zeros(2, 224, 224, 3, dtype=torch.uint8)
    rgb[..., 0] = 255
    rgb[..., 1] = 128

    actual = fake_encoder({"rgb": rgb})

    expected_pixels = torch.empty(2, 3, 224, 224)
    expected_pixels[:, 0] = 1.0
    expected_pixels[:, 1] = (128.0 / 255.0 - 0.25) / 0.25
    expected_pixels[:, 2] = 0.0
    torch.testing.assert_close(fake_encoder.backbone.last_pixels, expected_pixels)
    assert actual.shape == (2, 768)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.full((2, 768), 2.0))


@pytest.mark.parametrize(
    ("rgb", "message"),
    (
        (torch.zeros(224, 224, 3, dtype=torch.uint8), "BHWC"),
        (torch.zeros(1, 3, 224, 224, dtype=torch.uint8), "BHWC"),
        (torch.zeros(1, 224, 224, 4, dtype=torch.uint8), "3 channels"),
        (torch.zeros(1, 200, 224, 3, dtype=torch.uint8), "224x224"),
        (torch.zeros(1, 224, 200, 3, dtype=torch.uint8), "224x224"),
    ),
)
def test_encoder_rejects_wrong_rgb_layout_or_shape(fake_encoder, rgb, message):
    with pytest.raises(ValueError, match=message):
        fake_encoder({"rgb": rgb})


def test_encoder_rejects_non_uint8_rgb(fake_encoder):
    with pytest.raises(ValueError, match="uint8"):
        fake_encoder({"rgb": torch.zeros(1, 224, 224, 3)})


def test_encoder_stays_frozen_and_in_eval_mode_after_train(fake_encoder):
    assert not fake_encoder.training
    assert not fake_encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in fake_encoder.parameters())

    returned = fake_encoder.train(True)

    assert returned is fake_encoder
    assert not fake_encoder.training
    assert not fake_encoder.backbone.training
    assert all(not parameter.requires_grad for parameter in fake_encoder.parameters())


def test_encoder_forward_disables_gradient_tracking(fake_encoder):
    with torch.enable_grad():
        output = fake_encoder(
            {"rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8)}
        )

    assert fake_encoder.backbone.forward_grad_enabled is False
    assert output.requires_grad is False


def test_encoder_disables_outer_autocast_for_backbone(fake_encoder):
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = fake_encoder(
            {"rgb": torch.zeros(1, 224, 224, 3, dtype=torch.uint8)}
        )

    assert fake_encoder.backbone.last_hidden_dtype == torch.float32
    assert output.dtype == torch.float32


def test_encoder_is_never_blind(fake_encoder):
    assert fake_encoder.is_blind is False
