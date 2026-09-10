"""Persistent fusion state follows real observation counts and rollout autograd."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_graph_map_candidate_preview import GraphMap
from vlnce_baselines.nwm.ghost_concat_fusion import (
    GhostConcatFusionAdapter, apply_ghost_concat_to_graph,
)


def _graph(mode="persistent_node_state"):
    return GraphMap(False, 0.5, True, 0, ghost_concat_memory_mode=mode)


def _observe(graph, values):
    values = torch.as_tensor(values, dtype=torch.float32)
    count = len(values)
    graph.update_graph(
        None, 1, "0", np.zeros(3), torch.zeros(4),
        ["0_%d" % i for i in range(count)],
        [np.asarray([1., 0., 0.])] * count,
        values, None,
    )


def _inputs(graph):
    return {
        "gmap_img_fts": torch.stack([torch.zeros(4), graph.get_node_embeds("g0")])[None],
        "gmap_vp_ids": [[None, "g0"]],
        "gmap_masks": torch.tensor([[True, True]]),
        "gmap_visited_masks": torch.tensor([[False, False]]),
    }


def _pred(values=None):
    return SimpleNamespace(
        pred_cls_raw=torch.ones(1, 4) if values is None else values,
        meta={"records": [SimpleNamespace(env_index=0, ghost_vp="g0")]},
    )


def _adapter():
    adapter = GhostConcatFusionAdapter(4, 8)
    with torch.no_grad():
        adapter.residual.layers[-1].bias.fill_(2.)
    return adapter


def test_persistent_fusion_preserves_count_then_merges_real_observation():
    graph = _graph()
    _observe(graph, [[1.] * 4, [3.] * 4])
    output, stats = apply_ghost_concat_to_graph(_inputs(graph), _pred(), _adapter(), [graph])
    torch.testing.assert_close(graph.get_node_embeds("g0"), torch.full((4,), 4.))
    torch.testing.assert_close(output["gmap_img_fts"][0, 1], graph.get_node_embeds("g0"))
    assert graph.ghost_embeds["g0"][1] == 2
    torch.testing.assert_close(graph.ghost_embeds["g0"][0], torch.full((4,), 8.))
    _observe(graph, [[10.] * 4])
    assert graph.ghost_embeds["g0"][1] == 3
    torch.testing.assert_close(graph.get_node_embeds("g0"), torch.full((4,), 6.))
    assert stats["persistent_writeback_count"] == 1
    assert all(not value.requires_grad for value in stats.values())


def test_no_prediction_retains_state_and_deletion_and_episode_reset_clear_it():
    graph = _graph()
    _observe(graph, [[1.] * 4])
    adapter = _adapter()
    apply_ghost_concat_to_graph(_inputs(graph), _pred(), adapter, [graph])
    saved = graph.ghost_embeds["g0"]
    inputs = _inputs(graph)
    output, stats = apply_ghost_concat_to_graph(inputs, None, adapter, [graph])
    assert output is inputs and graph.ghost_embeds["g0"] is saved
    assert stats["persistent_retained_state_count"] == 1
    graph.delete_ghost("g0")
    assert not graph.ghost_embeds and not graph.ghost_concat_state_vps
    assert not _graph().ghost_concat_state_vps
    with pytest.raises(KeyError):
        graph.write_ghost_concat_state("g0", torch.zeros(4))


@pytest.mark.parametrize("mode", ["current_step_only", "persistent_node_state"])
def test_missing_graph_argument_and_legacy_mode_never_write(mode):
    graph = _graph(mode)
    _observe(graph, [[1.] * 4])
    saved = graph.ghost_embeds["g0"]
    adapter = _adapter()
    apply_ghost_concat_to_graph(_inputs(graph), _pred(), adapter)
    assert graph.ghost_embeds["g0"] is saved
    if mode == "current_step_only":
        apply_ghost_concat_to_graph(_inputs(graph), _pred(), adapter, [graph])
        assert graph.ghost_embeds["g0"] is saved
        with pytest.raises(ValueError, match="requires"):
            graph.write_ghost_concat_state("g0", torch.zeros(4))


@pytest.mark.parametrize("case", ["nan", "visited", "padding", "alpha_zero", "overflow"])
def test_invalid_or_disabled_rows_preserve_exact_stored_state(case):
    graph = _graph()
    _observe(graph, [[1.] * 4, [3.] * 4])
    adapter = _adapter()
    apply_ghost_concat_to_graph(_inputs(graph), _pred(), adapter, [graph])
    saved = graph.ghost_embeds["g0"]
    inputs, prediction = _inputs(graph), _pred()
    if case == "nan":
        prediction.pred_cls_raw[0, 0] = float("nan")
    elif case == "visited":
        inputs["gmap_visited_masks"][0, 1] = True
    elif case == "padding":
        inputs["gmap_masks"][0, 1] = False
    elif case == "alpha_zero":
        adapter.alpha = 0
    else:
        with torch.no_grad():
            adapter.residual.layers[-1].bias.fill_(torch.finfo(torch.float32).max * .75)
    output, stats = apply_ghost_concat_to_graph(inputs, prediction, adapter, [graph])
    assert graph.ghost_embeds["g0"] is saved
    torch.testing.assert_close(output["gmap_img_fts"], inputs["gmap_img_fts"], rtol=0, atol=0)
    assert stats["persistent_writeback_count"] == 0
    assert stats["persistent_retained_state_count"] == 1


def test_cross_step_gradient_matches_explicit_recurrence():
    torch.manual_seed(20)
    graph = _graph()
    observations = torch.randn(2, 4, requires_grad=True)
    _observe(graph, observations)
    adapter = GhostConcatFusionAdapter(4, 8, zero_init=False)
    reference = GhostConcatFusionAdapter(4, 8, zero_init=False)
    reference.load_state_dict(adapter.state_dict())
    predictions = torch.randn(2, 4, requires_grad=True)
    apply_ghost_concat_to_graph(_inputs(graph), _pred(predictions[:1]), adapter, [graph])
    new_observation = torch.randn(1, 4, requires_grad=True)
    _observe(graph, new_observation)
    output, _ = apply_ghost_concat_to_graph(_inputs(graph), _pred(predictions[1:]), adapter, [graph])
    output["gmap_img_fts"][0, 1].square().sum().backward()
    ref_obs = observations.detach().requires_grad_(True)
    ref_new = new_observation.detach().requires_grad_(True)
    ref_pred = predictions.detach().requires_grad_(True)
    first, _ = reference(ref_obs.mean(0, keepdim=True), ref_pred[:1])
    second, _ = reference((2 * first + ref_new) / 3, ref_pred[1:])
    second.square().sum().backward()
    assert predictions.grad[0].abs().sum() > 0
    torch.testing.assert_close(predictions.grad, ref_pred.grad)
    torch.testing.assert_close(observations.grad, ref_obs.grad)
    torch.testing.assert_close(new_observation.grad, ref_new.grad)
    for parameter, ref_parameter in zip(adapter.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad)


def test_default_mode_and_invalid_mode_and_nonfinite_direct_write():
    assert GraphMap(False, .5, True, 0).ghost_concat_memory_mode == "current_step_only"
    with pytest.raises(ValueError, match="ghost_concat_memory_mode"):
        _graph("unknown")
    graph = _graph()
    _observe(graph, [[1.] * 4])
    saved = graph.ghost_embeds["g0"]
    assert not graph.write_ghost_concat_state("g0", torch.full((4,), float("nan")))
    assert graph.ghost_embeds["g0"] is saved


def test_multi_environment_writeback_is_local_to_valid_live_ghost():
    graphs = [_graph(), _graph()]
    for graph in graphs:
        _observe(graph, [[1.] * 4])
    inputs = [_inputs(graph) for graph in graphs]
    batch = {key: (sum([item[key] for item in inputs], []) if key == "gmap_vp_ids"
                   else torch.cat([item[key] for item in inputs], dim=0))
             for key in inputs[0]}
    prediction = SimpleNamespace(
        pred_cls_raw=torch.tensor([[float("nan")] * 4, [1.] * 4]),
        meta={"records": [SimpleNamespace(env_index=1, ghost_vp="g0"),
                            SimpleNamespace(env_index=0, ghost_vp="g0")]},
    )
    saved = graphs[1].ghost_embeds["g0"]
    output, stats = apply_ghost_concat_to_graph(batch, prediction, _adapter(), graphs)
    assert graphs[1].ghost_embeds["g0"] is saved
    torch.testing.assert_close(graphs[0].get_node_embeds("g0"), torch.full((4,), 3.))
    torch.testing.assert_close(output["gmap_img_fts"][1], batch["gmap_img_fts"][1])
    assert stats["persistent_writeback_count"] == 1
