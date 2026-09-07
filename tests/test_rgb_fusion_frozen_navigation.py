"""Behavioral checks for the frozen RGB-only navigation experiment."""
from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.nwm.rgb_fusion import (
    RaeNwmRgbFusionAdapter, apply_rgb_fusion_to_current_candidates,
)
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer


def _trainer(*, frozen=True, aligned=False):
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        IL=SimpleNamespace(freeze_navigation_backbone=frozen, is_requeue=False),
        MODEL=SimpleNamespace(
            RAENWM=SimpleNamespace(enabled=True, rgb_fusion_enabled=True,
                                  rgb_fusion_align_navigation_cls=aligned),
            ACTIVE_LOOKAHEAD=SimpleNamespace(enabled=False),
        ),
    )
    trainer.policy = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.8))
    return trainer


def test_freeze_keeps_navigation_input_gradients_and_disables_dropout():
    trainer = _trainer()
    before = {k: v.clone() for k, v in trainer.policy.state_dict().items()}
    trainer._configure_navigation_backbone_training()
    adapter = RaeNwmRgbFusionAdapter(4, 4, gate_bias_init=0.0)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.01)
    observed, predicted = torch.randn(3, 4), torch.randn(3, 4)
    fused, _ = adapter(observed, predicted, [0.5] * 3, [1.0] * 3)
    output = trainer.policy(fused)
    torch.testing.assert_close(output, trainer.policy(fused), rtol=0, atol=0)
    output.square().mean().backward()
    assert adapter.residual.layers[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert all(not p.requires_grad and p.grad is None for p in trainer.policy.parameters())
    for key, value in trainer.policy.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_alignment_rejects_trainable_navigation_space():
    trainer = _trainer(frozen=False, aligned=True)
    with pytest.raises(ValueError, match="frozen navigation"):
        trainer._configure_navigation_backbone_training()


def test_frozen_cls_gradient_contract_does_not_require_updates():
    trainer = _trainer()
    residual = torch.nn.Linear(4, 4).requires_grad_(False)
    trainer.policy = SimpleNamespace(net=SimpleNamespace(
        rgb_encoder=SimpleNamespace(cls_residual_mlp=residual)))
    trainer._assert_rgb_cls_residual_navigation_gradient()


def test_prediction_alignment_applies_same_residual_before_agreement():
    raw = torch.tensor([[1.0, 0.0]])
    predicted = torch.tensor([[0.0, 1.0]])
    seen = {}

    def fusion(obs, pred, agreement, distance):
        seen.update(pred=pred, agreement=agreement)
        return obs + pred, {}

    outputs = {"cand_rgb": [raw.clone()]}
    previews = [[SimpleNamespace(target_kind="new_ghost", target_vp="g1")]]
    prediction = SimpleNamespace(pred_cls_raw=predicted, meta={"records": [
        SimpleNamespace(env_index=0, ghost_vp="g1", distance_m=1.0)]})
    apply_rgb_fusion_to_current_candidates(
        outputs, previews, prediction, fusion,
        prediction_transform=lambda x: x + torch.tensor([[2.0, -1.0]]),
    )
    torch.testing.assert_close(seen["pred"], torch.tensor([[2.0, 0.0]]))
    torch.testing.assert_close(seen["agreement"], torch.ones(1))
    torch.testing.assert_close(outputs["cand_rgb"][0], torch.tensor([[3.0, 0.0]]))


def test_new_aligned_run_allows_clean_base_but_rejects_unaligned_adapter():
    trainer = _trainer(aligned=True)
    trainer._validate_rgb_fusion_navigation_contract({})
    with pytest.raises(ValueError, match="alignment mismatch"):
        trainer._validate_rgb_fusion_navigation_contract(
            {"raenwm_rgb_fusion_adapter_state_dict": {}})


def test_resume_refuses_changed_freeze_contract():
    trainer = _trainer()
    trainer.config.IL.is_requeue = True
    with pytest.raises(ValueError, match="training contract mismatch"):
        trainer._validate_rgb_fusion_navigation_contract({})
    trainer._validate_rgb_fusion_navigation_contract({
        "rgb_fusion_navigation_contract": trainer._rgb_fusion_navigation_contract()
    })
