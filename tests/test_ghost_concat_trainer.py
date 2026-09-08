from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer
from vlnce_baselines import ss_trainer_ETP_R1 as trainer_module


def trainer(kind="ghost_concat"):
    obj = object.__new__(RLTrainer)
    obj.device = torch.device("cpu")
    obj.raenwm_rgb_fusion_adapter = None
    obj.config = SimpleNamespace(
        IL=SimpleNamespace(freeze_navigation_backbone=True, is_requeue=False),
        MODEL=SimpleNamespace(
            RGB_ENCODER=SimpleNamespace(type="rae_dinov2", output_size=768),
            ACTIVE_LOOKAHEAD=SimpleNamespace(enabled=False),
            RAENWM=SimpleNamespace(
                enabled=True, rgb_fusion_enabled=True, rgb_fusion_trainable=True,
                rgb_fusion_type=kind, ghost_concat_hidden_dim=1536,
                rgb_fusion_zero_init=True, rgb_fusion_alpha=1.0,
                rgb_fusion_align_navigation_cls=False, predict_cls_token=True,
                condition_source_pose="context_last",
            ),
        ),
    )
    return obj


def test_factory_builds_exact_three_layer_no_bottleneck_mlp():
    obj = trainer()
    module = obj._initialize_raenwm_rgb_fusion_adapter()
    linear = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [
        (1536, 1536), (1536, 1536), (1536, 768)
    ]


@pytest.mark.parametrize("field,value", [
    ("active", True), ("native", False), ("align", True),
])
def test_factory_rejects_unsupported_training_and_prediction_contract(field, value):
    obj = trainer()
    if field == "frozen": obj.config.IL.freeze_navigation_backbone = value
    if field == "active": obj.config.MODEL.ACTIVE_LOOKAHEAD.enabled = value
    if field == "native": obj.config.MODEL.RAENWM.predict_cls_token = value
    if field == "align": obj.config.MODEL.RAENWM.rgb_fusion_align_navigation_cls = value
    with pytest.raises(ValueError):
        obj._initialize_raenwm_rgb_fusion_adapter()


def test_prediction_stage_does_not_mutate_candidate_rgb_or_run_legacy_fusion(monkeypatch):
    obj = trainer()
    prediction = SimpleNamespace(meta={"records": []})
    obj.raenwm_runtime = SimpleNamespace(predict=lambda queries: prediction)
    obj._raenwm_low_level_context_enabled = lambda: True
    obj._build_raenwm_preview_queries = lambda *args: []
    obj._accumulate_rgb_fusion_diagnostics = lambda *args: pytest.fail("fusion recorded before graph construction")
    monkeypatch.setattr(trainer_module, "heading_from_quaternion", lambda value: 0.0)
    monkeypatch.setattr(trainer_module, "apply_rgb_fusion_to_current_candidates",
                        lambda *args, **kwargs: pytest.fail("legacy RGB mutation was used"))
    observed = {"cand_rgb": [torch.randn(3, 768)]}
    before = observed["cand_rgb"][0].clone()
    result = obj._run_raenwm_rgb_fusion_prediction(None, [[0, 0, 0]], [None], [[]], observed)
    assert result is prediction
    torch.testing.assert_close(observed["cand_rgb"][0], before, rtol=0, atol=0)
    assert obj.last_raenwm_rgb_fusion_diagnostics is None


@pytest.mark.parametrize("current,saved", [("ghost_concat", "residual_gate"), ("residual_gate", "ghost_concat")])
def test_checkpoint_rejects_cross_structure_adapter_loading(current, saved):
    obj = trainer(current)
    source = trainer(saved)
    checkpoint = {"raenwm_rgb_fusion_adapter_state_dict": {},
                  "rgb_fusion_navigation_contract": source._rgb_fusion_navigation_contract()}
    with pytest.raises(ValueError, match="structure mismatch"):
        obj._validate_rgb_fusion_navigation_contract(checkpoint)


def test_fresh_baseline_and_matching_new_checkpoint_are_accepted():
    obj = trainer()
    obj._validate_rgb_fusion_navigation_contract({})
    checkpoint = {"raenwm_rgb_fusion_adapter_state_dict": {},
                  "rgb_fusion_navigation_contract": obj._rgb_fusion_navigation_contract()}
    obj._validate_rgb_fusion_navigation_contract(checkpoint)
    bad = deepcopy(checkpoint)
    bad["rgb_fusion_navigation_contract"]["ghost_concat"]["layer_dims"][1] = 512
    with pytest.raises(ValueError, match="architecture"):
        obj._validate_rgb_fusion_navigation_contract(bad)
    obj.config.IL.is_requeue = True
    obj.config.MODEL.RAENWM.rgb_fusion_alpha = 0.5
    with pytest.raises(ValueError, match="training contract"):
        obj._validate_rgb_fusion_navigation_contract(checkpoint)
    obj.config.IL.is_requeue = False
    obj._validate_rgb_fusion_navigation_contract(checkpoint)  # Explicit eval alpha ablation.


def test_legacy_checkpoint_without_type_field_still_resumes():
    obj = trainer("residual_gate")
    saved = obj._rgb_fusion_navigation_contract()
    saved.pop("fusion_type")
    obj.config.IL.is_requeue = True
    obj._validate_rgb_fusion_navigation_contract({
        "raenwm_rgb_fusion_adapter_state_dict": {}, "rgb_fusion_navigation_contract": saved,
    })


def test_coverage_includes_queries_without_ready_prediction(monkeypatch):
    obj = trainer()
    obj.last_candidate_q0_prediction_diagnostics = {"q0_first_stage_requested": 7}
    nav_inputs = {"example": True}
    monkeypatch.setattr(trainer_module, "apply_ghost_concat_to_graph",
                        lambda *args: (nav_inputs, {"eligible_candidate_count": 2, "fused_candidate_count": 2}))
    recorded = []
    obj._accumulate_rgb_fusion_diagnostics = lambda query, rows: recorded.extend(rows)
    assert obj._apply_ghost_concat_prediction(nav_inputs, None) is nav_inputs
    assert recorded[0]["eligible_candidate_count"] == 7
    assert recorded[0]["fused_candidate_count"] == 2
