from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.nwm.rgb_fusion import (
    RaeNwmRgbFusionAdapter,
    apply_rgb_fusion_to_current_candidates,
    clone_wp_outputs_candidate_rgb,
)


@dataclass(frozen=True)
class _Preview:
    candidate_vp: str
    target_kind: str
    target_vp: str
    position: object = None
    front_vp: str = "0"


def test_zero_initialized_adapter_is_identity():
    adapter = RaeNwmRgbFusionAdapter(4, 6, zero_init=True)
    raw = torch.randn(2, 4)
    fused, diagnostics = adapter(
        raw, torch.randn(2, 4), torch.tensor([0.2, 0.8]), [1.0, 2.0]
    )
    torch.testing.assert_close(fused, raw, rtol=0, atol=0)
    assert diagnostics["gate"].shape == (2, 1)


def test_adapter_uses_nonzero_residual_confidence_and_distance_inputs():
    adapter = RaeNwmRgbFusionAdapter(4, 6, zero_init=True, alpha=0.5)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.zero_()
        adapter.residual.layers[-1].bias.fill_(2.0)
        adapter.gate[0].weight[0, -2] = 1.0
        adapter.gate[0].weight[1, -1] = 1.0
        adapter.gate[2].weight[0, :2] = 1.0
    raw = torch.zeros(2, 4)
    fused, _ = adapter(
        raw,
        torch.ones(2, 4),
        confidence=torch.tensor([0.0, 1.0]),
        distance=torch.tensor([0.0, 2.0]),
    )
    assert torch.all(fused[1] > fused[0])


@pytest.mark.parametrize(
    "raw,wm,match",
    (
        (torch.zeros(4), torch.zeros(4), "shape"),
        (torch.zeros(1, 3), torch.zeros(1, 3), "input_dim"),
        (torch.zeros(1, 4), torch.zeros(2, 4), "same shape"),
    ),
)
def test_adapter_rejects_invalid_dimensions(raw, wm, match):
    with pytest.raises(ValueError, match=match):
        RaeNwmRgbFusionAdapter(4, 4)(raw, wm, [1.0], [1.0])


class _BiasFusion(torch.nn.Module):
    def forward(self, raw, wm, confidence, distance):
        return raw + wm + confidence[:, None] + distance[:, None], {}


def _prediction(records):
    return SimpleNamespace(
        pred_cls=torch.tensor([[10.0] * 4 for _ in records]),
        confidence=torch.tensor([0.5 for _ in records]),
        meta={"records": records},
    )


def test_candidate_fusion_changes_only_ghosts_and_reuses_ghost_prediction():
    raw = torch.eye(4)[:3]
    wp_outputs = {"cand_rgb": [raw.clone()]}
    previews = [[
        _Preview("0_0", "node", "0"),
        _Preview("0_1", "new_ghost", "g0"),
        _Preview("0_2", "new_ghost", "g0"),
    ]]
    diagnostics = apply_rgb_fusion_to_current_candidates(
        wp_outputs,
        previews,
        _prediction([
            SimpleNamespace(env_index=0, ghost_vp="g0", distance_m=1.0)
        ]),
        _BiasFusion(),
    )
    torch.testing.assert_close(wp_outputs["cand_rgb"][0][0], raw[0])
    torch.testing.assert_close(
        wp_outputs["cand_rgb"][0][1:], raw[1:] + 11.5
    )
    assert diagnostics == [{"fused_candidate_count": 2}]


def test_candidate_clone_keeps_raw_node_panorama_features():
    wp_outputs = {"cand_rgb": [torch.eye(2)]}
    cloned = clone_wp_outputs_candidate_rgb(wp_outputs)
    wp_outputs["cand_rgb"][0][0].add_(10)
    torch.testing.assert_close(cloned["cand_rgb"][0], torch.eye(2))


def test_duplicate_prediction_record_is_rejected():
    record = SimpleNamespace(env_index=0, ghost_vp="g0", distance_m=1.0)
    with pytest.raises(ValueError, match="Duplicate"):
        apply_rgb_fusion_to_current_candidates(
            {"cand_rgb": [torch.zeros(1, 4)]},
            [[_Preview("0_0", "new_ghost", "g0")]],
            _prediction([record, record]),
            _BiasFusion(),
        )
