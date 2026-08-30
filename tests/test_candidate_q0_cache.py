from types import SimpleNamespace

import numpy as np
import torch

from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.nwm.active_lookahead.candidate_q0 import (
    Q0_CONTRACT,
    Q0_POSITION_SOURCE,
    commit_candidate_q0_cache,
)
from vlnce_baselines.nwm.etp_adapter import RaeSourceContextSnapshot
from vlnce_baselines.nwm.types import NwmCondition


def _graph():
    return GraphMap(False, 0.5, True, 0.0)


def _snapshot():
    return RaeSourceContextSnapshot(
        source_front_vp="0",
        source_high_level_step=0,
        context_latents=torch.zeros(4, 257, 768),
        source_position=np.zeros(3, dtype=np.float32),
        source_yaw=0.25,
    )


def test_candidate_q0_cache_uses_committed_mean_and_cpu_fp16_patch():
    graph = _graph()
    positions = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([1.2, 0.0, 0.0], dtype=np.float32),
    ]
    previews = graph.preview_candidate_mapping(
        "0", np.zeros(3, dtype=np.float32), ["0_0", "0_1"], positions
    )
    graph.update_graph(
        None,
        1,
        "0",
        np.zeros(3, dtype=np.float32),
        torch.zeros(2),
        ["0_0", "0_1"],
        positions,
        torch.ones(2, 2),
        None,
        candidate_preview=previews,
    )
    prediction = SimpleNamespace(
        pred_latent=torch.ones(1, 768, 16, 16),
        meta={
            "records": [
                SimpleNamespace(
                    env_index=0,
                    ghost_vp="g0",
                    condition=NwmCondition(0.1, 0.2, 0.3, 0.04),
                    horizon=5.12,
                )
            ]
        },
    )
    runtime = SimpleNamespace(
        source_context_snapshot=lambda *args, **kwargs: _snapshot()
    )

    records = commit_candidate_q0_cache(
        graph,
        env_index=0,
        candidate_previews=previews,
        candidate_view_indices=[1, 2],
        candidate_forward_distances=[1.0, 1.2],
        prediction=prediction,
        runtime=runtime,
        source_front_vp="0",
        source_high_level_step=0,
    )

    assert len(records) == 1
    record = graph.select_candidate_q0("g0")
    assert record.contract_version == Q0_CONTRACT
    assert record.position_source == Q0_POSITION_SOURCE
    np.testing.assert_allclose(record.target_position, graph.ghost_mean_pos["g0"])
    assert record.predicted_patch_cpu_fp16.device.type == "cpu"
    assert record.predicted_patch_cpu_fp16.dtype == torch.float16
    assert record.current_view_index == 1


def test_observed_ghost_loses_old_cache_when_new_prediction_is_missing():
    graph = _graph()
    position = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    previews = graph.preview_candidate_mapping("0", np.zeros(3), ["0_0"], [position])
    graph.update_graph(
        None, 1, "0", np.zeros(3), torch.zeros(2), ["0_0"], [position],
        torch.ones(1, 2), None, candidate_preview=previews,
    )
    prediction = SimpleNamespace(
        pred_latent=torch.ones(1, 768, 16, 16),
        meta={"records": [SimpleNamespace(
            env_index=0,
            ghost_vp="g0",
            condition=NwmCondition(0.1, 0.0, 0.0, 0.01),
            horizon=1.0,
        )]},
    )
    runtime = SimpleNamespace(source_context_snapshot=lambda *a, **k: _snapshot())
    kwargs = dict(
        graph=graph,
        env_index=0,
        candidate_previews=previews,
        candidate_view_indices=[0],
        candidate_forward_distances=[1.0],
        runtime=runtime,
        source_front_vp="0",
        source_high_level_step=0,
    )
    commit_candidate_q0_cache(prediction=prediction, **kwargs)
    assert graph.select_candidate_q0("g0") is not None

    next_previews = graph.preview_candidate_mapping(
        "1", np.asarray([0.2, 0.0, 0.0]), ["1_0"],
        [np.asarray([1.1, 0.0, 0.0], dtype=np.float32)],
    )
    graph.update_graph(
        "0", 2, "1", np.asarray([0.2, 0.0, 0.0]), torch.zeros(2),
        ["1_0"], [np.asarray([1.1, 0.0, 0.0], dtype=np.float32)],
        torch.ones(1, 2), None, candidate_preview=next_previews,
    )
    commit_candidate_q0_cache(
        graph,
        env_index=0,
        candidate_previews=next_previews,
        candidate_view_indices=[1],
        candidate_forward_distances=[0.9],
        prediction=SimpleNamespace(pred_latent=None, meta={"records": []}),
        runtime=runtime,
        source_front_vp="1",
        source_high_level_step=1,
    )
    assert graph.select_candidate_q0("g0") is None
