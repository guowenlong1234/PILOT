import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import torch


def _load_graph_utils():
    names = (
        "habitat", "habitat.tasks", "habitat.tasks.utils",
        "habitat.utils", "habitat.utils.geometry_utils",
    )
    old = {name: sys.modules.get(name) for name in names}
    modules = {name: types.ModuleType(name) for name in names}
    modules["habitat.tasks.utils"].cartesian_to_polar = lambda a, b: (a, b)
    geometry = modules["habitat.utils.geometry_utils"]
    geometry.quaternion_rotate_vector = lambda quat, vector: vector
    geometry.quaternion_from_coeff = lambda quat: quat
    sys.modules.update(modules)
    path = Path(__file__).parents[1] / "vlnce_baselines/models/graph_utils.py"
    spec = importlib.util.spec_from_file_location("preview_graph_utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return module


GraphMap = _load_graph_utils().GraphMap


def _graph(has_real_pos=False):
    return GraphMap(has_real_pos, 0.5, True, 0.0)


def test_preview_is_side_effect_free_and_merges_same_step_ghosts():
    graph = _graph()
    positions = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([1.2, 0.0, 0.0], dtype=np.float32),
    ]
    preview = graph.preview_candidate_mapping(
        "0", np.zeros(3, dtype=np.float32), ["0_0", "0_1"], positions
    )
    assert [(item.target_kind, item.target_vp) for item in preview] == [
        ("new_ghost", "g0"), ("new_ghost", "g0")
    ]
    assert graph.node_pos == {} and graph.ghost_pos == {}
    mapping = graph.update_graph(
        None, 1, "0", np.zeros(3, dtype=np.float32), torch.zeros(2),
        ["0_0", "0_1"], positions,
        torch.tensor([[1.0, 1.0], [3.0, 5.0]]), None,
        candidate_preview=preview,
    )
    assert mapping == [("0_0", "g0"), ("0_1", "g0")]
    torch.testing.assert_close(graph.get_node_embeds("g0"), torch.tensor([2.0, 3.0]))


def test_preview_classifies_node_and_existing_ghost_and_preserves_goal_cache():
    graph = _graph(has_real_pos=True)
    zero = np.zeros(3, dtype=np.float32)
    ghost_pos = np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
    graph.update_graph(
        None, 1, "0", zero, torch.zeros(2), ["0_0"], [ghost_pos],
        torch.ones(1, 2), [ghost_pos.copy()], [4.0],
    )
    preview = graph.preview_candidate_mapping(
        "1", np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        ["1_0", "1_1"],
        [zero, np.asarray([2.1, 0.0, 0.0], dtype=np.float32)],
    )
    assert [(item.target_kind, item.target_vp) for item in preview] == [
        ("node", "0"), ("existing_ghost", "g0")
    ]
    graph.update_graph(
        "0", 2, "1", np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        torch.zeros(2), ["1_0", "1_1"],
        [zero, np.asarray([2.1, 0.0, 0.0], dtype=np.float32)],
        torch.ones(2, 2), [zero, ghost_pos.copy()], [None, 2.0],
        candidate_preview=preview,
    )
    assert graph.ghost_goal_dists["g0"] == [4.0, 2.0]
    graph.delete_ghost("g0")
    assert "g0" not in graph.ghost_goal_dists


def test_persistent_q0_and_source_context_follow_ghost_lifecycle():
    graph = _graph()
    zero = np.zeros(3, dtype=np.float32)
    ghost_pos = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    mapping = graph.update_graph(
        None, 1, "0", zero, torch.zeros(2), ["0_0"], [ghost_pos],
        torch.ones(1, 2), [ghost_pos.copy()],
    )
    records = graph.record_persistent_q0_candidates(
        mapping,
        [ghost_pos],
        [ghost_pos],
        [{
            "valid": True,
            "raw_position": ghost_pos,
            "position": ghost_pos,
            "navmesh_island": 2,
        }],
        [3],
        [1.0],
        source_front_vp="0",
        source_high_level_step=0,
    )
    assert len(records) == 1
    assert graph.select_persistent_q0("g0") is records[0]
    snapshot = SimpleNamespace(
        source_front_vp="0", source_high_level_step=0
    )
    graph.record_raenwm_source_context(snapshot)
    assert graph.get_raenwm_source_context(records[0]) is snapshot

    graph.delete_ghost("g0")
    assert "g0" not in graph.ghost_persistent_q0
    assert graph.raenwm_source_contexts == {}
