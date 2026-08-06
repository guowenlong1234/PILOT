from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.models.etp import ETP_R1_vlnbert_init as init_module


def _online_config(checkpoint: Path):
    return SimpleNamespace(
        pretrained_path=str(checkpoint),
        RGB_ENCODER=SimpleNamespace(
            type="rae_dinov2",
            output_size=768,
        ),
        use_depth_embedding=True,
        use_sprels=True,
        fix_lang_embedding=False,
        fix_pano_embedding=False,
    )


def _recognizable_img_linear_state():
    return {
        "weight": torch.arange(768 * 768, dtype=torch.float32).reshape(768, 768)
        / (768 * 768),
        "bias": torch.arange(768, dtype=torch.float32) / 768,
    }


def _save_pretrain_checkpoint(path, img_linear_state, *, module_prefix=True):
    prefix = "module." if module_prefix else ""
    state = {
        f"{prefix}bert.img_embeddings.img_linear.{name}": value
        for name, value in img_linear_state.items()
    }
    state[f"{prefix}global_sap_head.net.4.bias"] = torch.full((1,), -0.75)
    torch.save(state, path)


def test_pretrain_img_linear_loads_exactly_into_online_model(tmp_path):
    expected = _recognizable_img_linear_state()
    checkpoint = tmp_path / "pretrain_direct_768.pt"
    _save_pretrain_checkpoint(checkpoint, expected)

    online_model = init_module.get_vlnbert_models(_online_config(checkpoint))
    actual = online_model.img_embeddings.img_linear.state_dict()

    assert list(actual) == list(expected)
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name], expected_value, rtol=0, atol=0)
    assert not hasattr(online_model.img_embeddings, "rgb_projection")
    torch.testing.assert_close(
        online_model.global_sap_head.net[4].bias,
        torch.full((1,), -0.75),
        rtol=0,
        atol=0,
    )


def test_online_loader_rejects_old_768_to_512_projection_checkpoint(tmp_path):
    checkpoint = tmp_path / "old_projection.pt"
    state = _recognizable_img_linear_state()
    _save_pretrain_checkpoint(checkpoint, state)
    loaded = torch.load(checkpoint, map_location="cpu")
    loaded["module.bert.img_embeddings.rgb_projection.0.weight"] = torch.ones(1)
    torch.save(loaded, checkpoint)

    with pytest.raises(ValueError, match="retired 768->512 rgb_projection"):
        init_module.get_vlnbert_models(_online_config(checkpoint))


def test_online_loader_rejects_512_dim_img_linear(tmp_path):
    checkpoint = tmp_path / "old_512_interface.pt"
    _save_pretrain_checkpoint(
        checkpoint,
        {
            "weight": torch.ones(768, 512),
            "bias": torch.ones(768),
        },
    )

    with pytest.raises(ValueError, match=r"img_linear.weight.*\(768, 768\)"):
        init_module.get_vlnbert_models(_online_config(checkpoint))
