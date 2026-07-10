from pathlib import Path
import subprocess
import sys

import pytest
import torch

from model_components.rgb_projection import build_rgb_projection


ROOT = Path(__file__).resolve().parents[1]


def test_projection_module_imports_from_pretraining_workdir():
    result = subprocess.run(
        [sys.executable, "-c", "import model_components"],
        cwd=ROOT / "pretrain_src" / "pretrain_src",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout


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


@pytest.mark.parametrize(
    ("parameter_name", "bad_value"),
    (
        ("raw_size", 512),
        ("output_size", 513),
        ("hidden_size", 769),
    ),
)
def test_rae_projection_reports_wrong_dimension_values(parameter_name, bad_value):
    dimensions = {
        "raw_size": 768,
        "output_size": 512,
        "hidden_size": 768,
    }
    dimensions[parameter_name] = bad_value

    with pytest.raises(ValueError, match=rf"{parameter_name}={bad_value}"):
        build_rgb_projection("rae_dinov2", **dimensions)


def test_clip_projection_rejects_wrong_dimensions():
    with pytest.raises(ValueError, match="raw_size=768"):
        build_rgb_projection("clip", 768, 512, 768)


def test_projection_rejects_unknown_encoder_type():
    with pytest.raises(ValueError, match="Unsupported RGB encoder type"):
        build_rgb_projection("unknown", 768, 512, 768)


@pytest.mark.parametrize(
    ("parameter_name", "bad_value"),
    (
        ("raw_size", 768.9),
        ("output_size", 512.1),
        ("hidden_size", True),
    ),
)
def test_projection_rejects_dimensions_that_are_not_exact_integers(
    parameter_name,
    bad_value,
):
    dimensions = {
        "raw_size": 768,
        "output_size": 512,
        "hidden_size": 768,
    }
    dimensions[parameter_name] = bad_value

    with pytest.raises(
        ValueError,
        match=rf"{parameter_name}.*{bad_value}",
    ):
        build_rgb_projection("rae_dinov2", **dimensions)
