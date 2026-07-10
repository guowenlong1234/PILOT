import pytest
import torch

from model_components.rgb_projection import build_rgb_projection


def test_rae_projection_has_expected_layers_and_output_shape():
    projection = build_rgb_projection("rae_dinov2", 768, 512, 768)
    modules = list(projection)

    linear_layers = [
        module for module in modules if isinstance(module, torch.nn.Linear)
    ]

    assert [type(module) for module in modules] == [
        torch.nn.Linear,
        torch.nn.GELU,
        torch.nn.Linear,
        torch.nn.GELU,
        torch.nn.Linear,
    ]
    assert [
        (layer.in_features, layer.out_features) for layer in linear_layers
    ] == [(768, 768), (768, 768), (768, 512)]
    assert projection(torch.randn(2, 36, 768)).shape == (2, 36, 512)


def test_clip_projection_is_identity_and_returns_same_tensor():
    projection = build_rgb_projection("clip", 512, 512, 768)
    features = torch.randn(2, 12, 512)

    assert isinstance(projection, torch.nn.Identity)
    assert projection(features) is features


def test_rae_projection_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="768"):
        build_rgb_projection("rae_dinov2", 512, 512, 768)


def test_clip_projection_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="512"):
        build_rgb_projection("clip", 768, 512, 768)


def test_projection_rejects_unknown_encoder_type():
    with pytest.raises(ValueError, match="Unsupported RGB encoder type"):
        build_rgb_projection("unknown", 768, 512, 768)
