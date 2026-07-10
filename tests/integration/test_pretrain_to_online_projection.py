from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from pretrain_src.pretrain_src.model.vilmodel import (
    ImageEmbeddings as PretrainImageEmbeddings,
)
from vlnce_baselines.models.etp import ETP_R1_vlnbert_init as init_module


def _image_config():
    return PretrainedConfig(
        rgb_encoder_type="rae_dinov2",
        raw_image_feat_size=768,
        image_feat_size=512,
        projection_hidden_size=768,
        hidden_size=768,
        depth_feat_size=128,
        angle_feat_size=4,
        hidden_dropout_prob=0.0,
        num_pano_layers=0,
        layer_norm_eps=1e-5,
        obj_feat_size=0,
    )


def _online_config(checkpoint: Path):
    return SimpleNamespace(
        pretrained_path=str(checkpoint),
        RGB_ENCODER=SimpleNamespace(
            type="rae_dinov2",
            raw_output_size=768,
            output_size=512,
            projection_hidden_size=768,
        ),
        use_depth_embedding=True,
        use_sprels=True,
        fix_lang_embedding=False,
        fix_pano_embedding=False,
    )


def _recognizable_projection_state():
    projection = PretrainImageEmbeddings(_image_config()).rgb_projection
    expected = {}
    for index, (name, value) in enumerate(projection.state_dict().items(), 1):
        expected[name] = torch.full_like(value, index / 16)
    projection.load_state_dict(expected)
    return projection, expected


def _save_pretrain_checkpoint(path, projection_state, *, module_prefix=True):
    prefix = "module." if module_prefix else ""
    state = {
        f"{prefix}bert.img_embeddings.rgb_projection.{name}": value
        for name, value in projection_state.items()
    }
    state[f"{prefix}global_sap_head.net.4.bias"] = torch.full((1,), -0.75)
    torch.save(state, path)


def test_all_pretrain_projection_parameters_load_exactly_into_online_model(
    tmp_path,
):
    projection, expected = _recognizable_projection_state()
    checkpoint = tmp_path / "pretrain_projection.pt"
    _save_pretrain_checkpoint(checkpoint, projection.state_dict())

    online_model = init_module.get_vlnbert_models(_online_config(checkpoint))
    actual = online_model.img_embeddings.rgb_projection.state_dict()

    assert list(actual) == list(expected)
    for name, expected_value in expected.items():
        torch.testing.assert_close(
            actual[name], expected_value, rtol=0, atol=0
        )
    torch.testing.assert_close(
        online_model.global_sap_head.net[4].bias,
        torch.full((1,), -0.75),
        rtol=0,
        atol=0,
    )


def test_online_loader_rejects_partial_pretrain_projection(
    tmp_path, monkeypatch
):
    _, complete_state = _recognizable_projection_state()
    missing_name = "4.bias"
    partial_state = {
        name: value
        for name, value in complete_state.items()
        if name != missing_name
    }
    checkpoint = tmp_path / "partial_projection.pt"
    _save_pretrain_checkpoint(checkpoint, partial_state)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, **kwargs):
            return kwargs

    monkeypatch.setattr(
        "vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt."
        "GlocalTextPathNavCMT",
        FakeModel,
    )

    with pytest.raises(
        ValueError,
        match=r"RAE/DINOv2.*complete.*rgb_projection.*4\.bias",
    ):
        init_module.get_vlnbert_models(_online_config(checkpoint))
