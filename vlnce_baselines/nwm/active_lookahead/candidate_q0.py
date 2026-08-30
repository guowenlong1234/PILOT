"""Shared first-stage Q0 query and cache contract for SFT and frozen GRPO."""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Sequence

import numpy as np
import torch

from .types import CandidateQ0


Q0_CONTRACT = "r1_post_update_ghost_mean_cached_v1"
Q0_POSITION_SOURCE = "r1_post_update_ghost_mean"


def build_candidate_q0_queries(
    cur_pos,
    cur_ori,
    candidate_previews,
    *,
    heading_from_orientation=None,
):
    """Build exactly one query per ghost at its final post-update R1 mean."""

    from vlnce_baselines.nwm.runtime import NwmQuery

    if heading_from_orientation is None:
        from vlnce_baselines.models.graph_utils import heading_from_quaternion

        heading_from_orientation = heading_from_quaternion

    queries = []
    for env_index, previews in enumerate(candidate_previews):
        grouped = OrderedDict()
        for preview in previews:
            if preview.target_kind not in ("new_ghost", "existing_ghost"):
                continue
            ghost_vp = str(preview.target_vp)
            target = np.asarray(
                preview.post_update_ghost_mean, dtype=np.float32
            )
            previous = grouped.setdefault(ghost_vp, target)
            if not np.allclose(previous, target, atol=1.0e-6, rtol=0.0):
                raise ValueError(
                    f"candidate previews disagree on final mean for {ghost_vp}"
                )
        for ghost_vp, target in grouped.items():
            queries.append(
                NwmQuery(
                    env_index=int(env_index),
                    query_id=ghost_vp,
                    current_position=np.asarray(cur_pos[env_index], dtype=np.float32),
                    current_yaw=float(
                        heading_from_orientation(cur_ori[env_index])
                    ),
                    target_position=target.copy(),
                )
            )
    return queries


def _prediction_rows(prediction: Any):
    if prediction is None:
        return {}
    records = list((getattr(prediction, "meta", None) or {}).get("records", ()))
    latent = getattr(prediction, "pred_latent", None)
    if not records or latent is None:
        return {}
    if not torch.is_tensor(latent) or tuple(latent.shape[1:]) != (768, 16, 16):
        shape = tuple(latent.shape) if torch.is_tensor(latent) else None
        raise ValueError(f"candidate q0 patch shape is invalid: {shape}")
    if int(latent.shape[0]) != len(records):
        raise ValueError("candidate q0 prediction rows do not match metadata")
    rows = {}
    for index, record in enumerate(records):
        patch = latent[index]
        if not bool(torch.isfinite(patch).all()):
            continue
        key = (int(record.env_index), str(record.ghost_vp))
        if key in rows:
            raise ValueError(f"duplicate candidate q0 prediction row: {key}")
        rows[key] = (record, patch)
    return rows


def _representative_candidate(previews, view_indices, forward_distances, ghost_vp):
    candidates = []
    for index, preview in enumerate(previews):
        if str(preview.target_vp) != str(ghost_vp):
            continue
        if preview.target_kind not in ("new_ghost", "existing_ghost"):
            continue
        mean = np.asarray(preview.post_update_ghost_mean, dtype=np.float32)
        position = np.asarray(preview.position, dtype=np.float32)
        candidates.append(
            (
                float(np.linalg.norm(position - mean)),
                index,
                int(view_indices[index]),
                float(forward_distances[index]),
                mean,
            )
        )
    if not candidates:
        raise KeyError(f"no candidate maps to observed ghost {ghost_vp}")
    _, _, view_index, forward, target = min(candidates)
    if view_index < 0 or not math.isfinite(forward) or forward < 0:
        raise ValueError("candidate q0 geometry is invalid")
    return view_index, forward, target.copy()


def commit_candidate_q0_cache(
    graph,
    *,
    env_index: int,
    candidate_previews: Sequence[Any],
    candidate_view_indices: Sequence[int],
    candidate_forward_distances: Sequence[float],
    prediction: Any,
    runtime: Any,
    source_front_vp: str,
    source_high_level_step: int,
):
    """Publish successful Q0 rows after graph update and invalidate failures."""

    if not (
        len(candidate_previews)
        == len(candidate_view_indices)
        == len(candidate_forward_distances)
    ):
        raise ValueError("candidate q0 cache inputs must be aligned")
    observed = list(OrderedDict.fromkeys(
        str(preview.target_vp)
        for preview in candidate_previews
        if preview.target_kind in ("new_ghost", "existing_ghost")
    ))
    rows = _prediction_rows(prediction)
    snapshot = runtime.source_context_snapshot(
        int(env_index),
        source_front_vp=str(source_front_vp),
        source_high_level_step=int(source_high_level_step),
    )
    cached = []
    if snapshot is not None:
        for ghost_vp in observed:
            row = rows.get((int(env_index), ghost_vp))
            if row is None:
                continue
            record, patch = row
            view_index, forward, target = _representative_candidate(
                candidate_previews,
                candidate_view_indices,
                candidate_forward_distances,
                ghost_vp,
            )
            condition = record.condition
            target_yaw = float(snapshot.source_yaw) + float(condition.dtheta)
            patch_cpu = patch.detach().to(device="cpu", dtype=torch.float32)
            patch_fp16 = patch_cpu.to(dtype=torch.float16).contiguous()
            quantization_max_abs = float(
                (patch_cpu - patch_fp16.float()).abs().max().item()
            )
            cached.append(
                CandidateQ0(
                    contract_version=Q0_CONTRACT,
                    position_source=Q0_POSITION_SOURCE,
                    ghost_vp=ghost_vp,
                    source_front_vp=str(source_front_vp),
                    source_high_level_step=int(source_high_level_step),
                    source_position=np.asarray(
                        snapshot.source_position, dtype=np.float32
                    ).copy(),
                    source_yaw=float(snapshot.source_yaw),
                    target_position=target,
                    target_yaw=target_yaw,
                    condition=(
                        float(condition.dx),
                        float(condition.dy),
                        float(condition.dtheta),
                        float(condition.rel_t),
                    ),
                    horizon=float(record.horizon),
                    source_context=snapshot,
                    predicted_patch_cpu_fp16=patch_fp16,
                    patch_quantization_max_abs=quantization_max_abs,
                    candidate_forward_m=forward,
                    current_view_index=view_index,
                )
            )
    return graph.replace_candidate_q0_cache(observed, cached)
