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
    adapter = RaeNwmRgbFusionAdapter(
        4, 6, zero_init=True, gate_bias_init=0.0
    )
    raw = torch.randn(2, 4)
    fused, diagnostics = adapter(
        raw, torch.randn(2, 4), torch.tensor([0.2, 0.8]), [1.0, 2.0]
    )
    torch.testing.assert_close(fused, raw, rtol=0, atol=0)
    assert diagnostics["gate"].shape == (2, 1)
    torch.testing.assert_close(
        diagnostics["gate"], torch.full((2, 1), 0.5)
    )


def test_adapter_uses_nonzero_residual_confidence_and_distance_inputs():
    adapter = RaeNwmRgbFusionAdapter(
        4, 6, zero_init=True, alpha=0.5, gate_bias_init=0.0
    )
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
        agreement=torch.tensor([0.0, 1.0]),
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


def _native_prediction(records, value):
    return SimpleNamespace(
        pred_cls=None,
        pred_cls_raw=torch.as_tensor(value, dtype=torch.float32),
        confidence=None,
        meta={"records": records},
    )


class _AgreementFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.agreements = []

    def forward(self, raw, wm, agreement, distance):
        self.agreements.append(agreement.detach().clone())
        return raw, {}


def test_native_prediction_uses_cosine_agreement_in_zero_to_one_range():
    raw = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    wp_outputs = {"cand_rgb": [raw.clone()]}
    previews = [[
        _Preview("0_0", "new_ghost", "g0"),
        _Preview("0_1", "existing_ghost", "g1"),
    ]]
    records = [
        SimpleNamespace(env_index=0, ghost_vp="g0", distance_m=1.0),
        SimpleNamespace(env_index=0, ghost_vp="g1", distance_m=2.0),
    ]
    fusion = _AgreementFusion()

    apply_rgb_fusion_to_current_candidates(
        wp_outputs,
        previews,
        _native_prediction(records, [[1.0, 0.0], [0.0, -1.0]]),
        fusion,
    )

    torch.testing.assert_close(fusion.agreements[0], torch.tensor([1.0]))
    torch.testing.assert_close(fusion.agreements[1], torch.tensor([0.0]))


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
