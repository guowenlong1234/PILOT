from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.nwm.ghost_concat_fusion import (
    GhostConcatFusionAdapter,
    apply_ghost_concat_to_graph,
)


def _prediction(rows, values):
    return SimpleNamespace(
        pred_cls_raw=values,
        meta={"records": [SimpleNamespace(env_index=e, ghost_vp=g) for e, g in rows]},
    )


def _graph():
    return {
        "gmap_img_fts": torch.arange(40, dtype=torch.float32).reshape(2, 5, 4) / 10,
        "gmap_vp_ids": [[None, "0", "g0", "g1"], [None, "0", "g0", "g2", "gpad"]],
        "gmap_masks": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.bool),
        "gmap_visited_masks": torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0]], dtype=torch.bool),
    }


def test_default_architecture_has_three_linear_layers_and_no_bottleneck():
    module = GhostConcatFusionAdapter()
    linear = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear] == [
        (1536, 1536), (1536, 1536), (1536, 768)
    ]
    with pytest.raises(ValueError, match="no bottleneck"):
        GhostConcatFusionAdapter(4, 7)


def test_zero_initialization_preserves_graph_and_receives_gradient_through_frozen_head():
    graph = _graph()
    saved = graph["gmap_img_fts"].clone()
    adapter = GhostConcatFusionAdapter(4, 8)
    head = torch.nn.Linear(4, 2)
    head.requires_grad_(False)
    pred = _prediction([(0, "g0"), (1, "g0")], torch.randn(2, 4))
    out, stats = apply_ghost_concat_to_graph(graph, pred, adapter)
    torch.testing.assert_close(out["gmap_img_fts"], saved, rtol=0, atol=0)
    assert out is not graph
    assert out["gmap_img_fts"].data_ptr() != graph["gmap_img_fts"].data_ptr()
    torch.testing.assert_close(graph["gmap_img_fts"], saved, rtol=0, atol=0)
    loss = head(out["gmap_img_fts"])[..., 0].sum()
    loss.backward()
    assert adapter.residual.layers[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in head.parameters())
    assert stats["fused_candidate_count"] == 2


def test_multi_env_single_forward_matches_serial_and_preserves_other_rows():
    torch.manual_seed(12)
    graph = _graph()
    original = graph["gmap_img_fts"].clone()
    adapter = GhostConcatFusionAdapter(4, 8, zero_init=False, alpha=0.3)
    pred = _prediction([(1, "g0"), (0, "g1"), (0, "g0")], torch.randn(3, 4))
    calls = []
    hook = adapter.register_forward_hook(lambda module, args, result: calls.append(args[0].shape))
    output, stats = apply_ghost_concat_to_graph(graph, pred, adapter)
    hook.remove()
    assert calls == [torch.Size([3, 4])]
    expected = original.clone()
    for row, (env, col) in enumerate([(1, 2), (0, 3), (0, 2)]):
        expected[env, col] = adapter(original[env, col:col + 1], pred.pred_cls_raw[row:row + 1])[0][0]
    torch.testing.assert_close(output["gmap_img_fts"], expected)
    torch.testing.assert_close(graph["gmap_img_fts"], original, rtol=0, atol=0)
    assert stats["fused_candidate_count"] == 3
    # An unrelated ghost's prediction cannot change the selected ghost.
    changed_pred = pred.pred_cls_raw.clone()
    changed_pred[0] += 100
    other, _ = apply_ghost_concat_to_graph(graph, _prediction([(1, "g0"), (0, "g1"), (0, "g0")], changed_pred), adapter)
    torch.testing.assert_close(other["gmap_img_fts"][0], output["gmap_img_fts"][0], rtol=0, atol=0)


def test_visited_padding_missing_and_nonfinite_predictions_are_ignored():
    graph = _graph()
    adapter = GhostConcatFusionAdapter(4, 8)
    with torch.no_grad():
        adapter.residual.layers[-1].bias.fill_(2)
    rows = [(0, "g0"), (0, "g1"), (1, "g0"), (1, "g2"), (1, "gpad"), (2, "g0"), (0, "0")]
    values = torch.ones(7, 4)
    values[1, 0] = float("nan")
    values[2, 2] = float("inf")
    result, stats = apply_ghost_concat_to_graph(graph, _prediction(rows, values), adapter)
    expected = graph["gmap_img_fts"].clone()
    expected[0, 2] += 2
    torch.testing.assert_close(result["gmap_img_fts"], expected, rtol=0, atol=0)
    assert stats["eligible_candidate_count"] == 3
    assert stats["fused_candidate_count"] == 1
    assert stats["invalid_prediction_count"] == 2
    result["gmap_img_fts"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters())


def test_alpha_zero_is_exact_identity_with_nonfinite_prediction():
    graph = _graph()
    adapter = GhostConcatFusionAdapter(4, 8, zero_init=False, alpha=0)
    result, stats = apply_ghost_concat_to_graph(
        graph, _prediction([(0, "g0")], torch.full((1, 4), float("nan"))), adapter
    )
    torch.testing.assert_close(result["gmap_img_fts"], graph["gmap_img_fts"], rtol=0, atol=0)
    assert stats["fused_candidate_count"] == 0


def test_graph_gradients_match_serial_reference_and_prediction_rows_are_independent():
    torch.manual_seed(4)
    graph = _graph()
    graph["gmap_img_fts"].requires_grad_(True)
    adapter = GhostConcatFusionAdapter(4, 8, zero_init=False)
    serial = GhostConcatFusionAdapter(4, 8, zero_init=False)
    serial.load_state_dict(adapter.state_dict())
    values = torch.randn(2, 4, requires_grad=True)
    pred = _prediction([(0, "g0"), (1, "g0")], values)
    result, _ = apply_ghost_concat_to_graph(graph, pred, adapter)
    result["gmap_img_fts"][0, 2].square().sum().backward()
    raw_ref = graph["gmap_img_fts"][0, 2:3].detach().requires_grad_(True)
    pred_ref = values[0:1].detach().requires_grad_(True)
    serial(raw_ref, pred_ref)[0].square().sum().backward()
    torch.testing.assert_close(values.grad[0], pred_ref.grad[0])
    torch.testing.assert_close(values.grad[1], torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(graph["gmap_img_fts"].grad[0, 2], raw_ref.grad[0])
    for batch_param, serial_param in zip(adapter.parameters(), serial.parameters()):
        torch.testing.assert_close(batch_param.grad, serial_param.grad)


@pytest.mark.parametrize("prediction", [None, SimpleNamespace(pred_cls_raw=None), _prediction([], torch.empty(0, 4)), _prediction([(5, "g0")], torch.ones(1, 4))])
def test_no_matches_returns_original_without_forward(prediction):
    graph = _graph()
    adapter = GhostConcatFusionAdapter(4, 8)
    calls = []
    hook = adapter.register_forward_hook(lambda *args: calls.append(1))
    result, stats = apply_ghost_concat_to_graph(graph, prediction, adapter)
    hook.remove()
    assert result is graph
    assert not calls
    assert stats["fused_candidate_count"] == 0


def test_duplicate_and_mismatched_records_fail_loudly():
    graph = _graph()
    adapter = GhostConcatFusionAdapter(4, 8)
    with pytest.raises(ValueError, match="Duplicate"):
        apply_ghost_concat_to_graph(graph, _prediction([(0, "g0"), (0, "g0")], torch.ones(2, 4)), adapter)
    with pytest.raises(ValueError, match="row count"):
        apply_ghost_concat_to_graph(graph, _prediction([(0, "g0")], torch.ones(2, 4)), adapter)
