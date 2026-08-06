from pathlib import Path

import torch
from transformers import PretrainedConfig

from pretrain_src.pretrain_src.model.vilmodel import (
    ImageEmbeddings as PretrainImageEmbeddings,
)
from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import (
    ImageEmbeddings as OnlineImageEmbeddings,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(encoder_type, image_feat_size):
    return PretrainedConfig(
        rgb_encoder_type=encoder_type,
        image_feat_size=image_feat_size,
        hidden_size=768,
        depth_feat_size=128,
        angle_feat_size=4,
        obj_feat_size=0,
        hidden_dropout_prob=0.0,
        num_pano_layers=0,
        layer_norm_eps=1e-5,
        use_depth_embedding=True,
    )


def test_legacy_rgb_projection_component_is_removed():
    assert not (ROOT / "model_components" / "rgb_projection.py").exists()


def test_rae_pretrain_and_online_interfaces_are_direct_768_to_768():
    config = _config("rae_dinov2", 768)
    pretrain = PretrainImageEmbeddings(config)
    online = OnlineImageEmbeddings(config)

    for module in (pretrain, online):
        assert not hasattr(module, "rgb_projection")
        assert not hasattr(module, "project_rgb")
        assert module.img_linear.in_features == 768
        assert module.img_linear.out_features == 768
        assert module.img_linear(torch.randn(2, 36, 768)).shape == (
            2,
            36,
            768,
        )


def test_clip_interface_remains_direct_512_to_768():
    config = _config("clip", 512)
    module = OnlineImageEmbeddings(config)

    assert module.img_linear.in_features == 512
    assert module.img_linear.out_features == 768
