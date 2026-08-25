"""Small online decision helpers retained by predicted-q1 joint SFT."""

from typing import Any, Sequence

import torch

from .topk_query import executable_ghost_indices


E24_FACTORY = (
    "vlnce_baselines.nwm.active_lookahead.residual_head:"
    "InterleavedCrossModalTopKFutureLogitResidualHead"
)
E24_AVG3_SOURCE_STEPS = (17000, 17250, 17500)


def quantize_like_offline_cache(values: torch.Tensor) -> torch.Tensor:
    """Reproduce fp16-on-disk followed by fp32-on-load future features."""

    return values.to(torch.float16).to(torch.float32)


def stop_isolated_e24_actions(
    base_logits: torch.Tensor,
    deltas: torch.Tensor,
    gmap_vp_ids: Sequence[Sequence[Any]],
) -> torch.Tensor:
    """Preserve base STOP; otherwise argmax over every adjusted ghost."""

    if deltas.shape != base_logits.shape:
        raise ValueError("E24 global deltas must match base logits")
    actions = base_logits.detach().argmax(dim=-1)
    for row, ids in enumerate(gmap_vp_ids):
        if int(actions[row]) == 0:
            continue
        ghost_indices = executable_ghost_indices(ids)
        if not ghost_indices:
            raise RuntimeError("E24 base MOVE row has no executable ghost")
        scores = base_logits[row, list(ghost_indices)].detach() + deltas[
            row, list(ghost_indices)
        ]
        actions[row] = ghost_indices[int(scores.argmax())]
    return actions
