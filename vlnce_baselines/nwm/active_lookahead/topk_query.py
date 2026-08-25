"""Stable frozen-base topK selection and bounded action-invariance proof."""

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .types import BoundedInvariantResult, TopKQueryPlan


def _scores(values: Any, *, count: Optional[int] = None) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError("base_logits must be one-dimensional")
    if count is not None:
        if result.shape[0] < count:
            raise ValueError("base_logits must cover every gmap_vp_id")
        result = result[:count]
    if np.isnan(result).any():
        raise ValueError("base_logits must not contain NaN")
    return result


def executable_ghost_indices(gmap_vp_ids: Sequence[Any]) -> Tuple[int, ...]:
    return tuple(
        index
        for index, vp in enumerate(gmap_vp_ids)
        if isinstance(vp, str) and vp.startswith("g")
    )


def stable_topk_ghost_indices(
    gmap_vp_ids: Sequence[Any],
    base_logits: Any,
    *,
    k: int = 5,
) -> Tuple[int, ...]:
    """Rank executable ghosts by frozen logits; ties keep gmap order."""

    topk = int(k)
    if topk <= 0:
        raise ValueError("k must be positive")
    logits = _scores(base_logits, count=len(gmap_vp_ids))
    ghost_indices = executable_ghost_indices(gmap_vp_ids)
    if any(not np.isfinite(logits[index]) for index in ghost_indices):
        raise ValueError("executable ghost logits must be finite")
    ranked = sorted(ghost_indices, key=lambda index: (-float(logits[index]), index))
    return tuple(ranked[: min(topk, len(ranked))])


def bounded_action_invariant(
    base_logits: Any,
    executable_indices: Sequence[int],
    query_indices: Sequence[int],
    *,
    delta_max: float,
    winner_index: Optional[int] = None,
) -> BoundedInvariantResult:
    """Prove whether every legal queried delta preserves the ghost winner."""

    logits = _scores(base_logits)
    executable = tuple(int(index) for index in executable_indices)
    if not executable:
        raise ValueError("at least one executable ghost is required")
    if len(set(executable)) != len(executable):
        raise ValueError("executable_indices must be unique")
    if min(executable) < 0 or max(executable) >= len(logits):
        raise IndexError("executable ghost index is outside base_logits")
    if any(not np.isfinite(logits[index]) for index in executable):
        raise ValueError("executable ghost logits must be finite")
    queried = {int(index) for index in query_indices}
    if not queried.issubset(set(executable)):
        raise ValueError("query_indices must be executable ghost indices")
    bound = float(delta_max)
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("delta_max must be finite and non-negative")

    if winner_index is None:
        winner = max(executable, key=lambda index: (float(logits[index]), -index))
    else:
        winner = int(winner_index)
        if winner not in executable:
            raise ValueError("winner_index must be an executable ghost index")
        expected = max(executable, key=lambda index: (float(logits[index]), -index))
        if winner != expected:
            raise ValueError("winner_index must match the frozen base ghost winner")

    winner_lower = float(logits[winner]) - (bound if winner in queried else 0.0)
    competitors = [index for index in executable if index != winner]
    if competitors:
        competitor = max(
            competitors,
            key=lambda index: (
                float(logits[index]) + (bound if index in queried else 0.0),
                -index,
            ),
        )
        competitor_upper = float(logits[competitor]) + (
            bound if competitor in queried else 0.0
        )
    else:
        competitor = None
        competitor_upper = -float("inf")
    return BoundedInvariantResult(
        invariant=winner_lower > competitor_upper,
        winner_index=winner,
        winner_lower_bound=winner_lower,
        max_competitor_index=competitor,
        max_competitor_upper_bound=competitor_upper,
    )


def build_topk_query_plan(
    gmap_vp_ids: Sequence[Any],
    base_logits: Any,
    *,
    k: int = 5,
    delta_max: float = 1.0,
    stop_index: int = 0,
    enable_bounded_skip: bool = True,
) -> TopKQueryPlan:
    """Build a teacher-free query plan with STOP and hard bounded skips."""

    topk = int(k)
    if topk <= 0:
        raise ValueError("k must be positive")
    bound = float(delta_max)
    if not np.isfinite(bound) or bound < 0:
        raise ValueError("delta_max must be finite and non-negative")
    logits = _scores(base_logits, count=len(gmap_vp_ids))
    stop = int(stop_index)
    if stop < 0 or stop >= len(gmap_vp_ids):
        raise IndexError("stop_index is outside gmap_vp_ids")
    if not np.isfinite(logits[stop]):
        raise ValueError("STOP logit must be finite")
    ghosts = executable_ghost_indices(gmap_vp_ids)

    # np.argmax is stable and therefore follows the planner's first-index tie
    # behavior. Masked non-executable entries may remain -inf.
    base_action = int(np.argmax(logits))
    if base_action == stop:
        return TopKQueryPlan(
            base_action_index=base_action,
            base_stop=True,
            executable_ghost_indices=ghosts,
            ranked_ghost_indices=(),
            ranked_ghost_ids=(),
            query_indices=(),
            query_ghost_ids=(),
            skip_reason="base_stop",
            bounded_result=None,
        )
    if base_action not in ghosts:
        raise ValueError("frozen base MOVE winner must be an executable ghost")
    ranked = stable_topk_ghost_indices(gmap_vp_ids, logits, k=topk)
    if not ranked:
        raise ValueError("frozen base MOVE decision has no executable ghost")

    proof = bounded_action_invariant(
        logits,
        ghosts,
        ranked,
        delta_max=bound,
        winner_index=base_action,
    )
    if enable_bounded_skip and proof.invariant:
        query_indices = ()
        query_ids = ()
        reason = "bounded_invariant"
    else:
        query_indices = ranked
        query_ids = tuple(str(gmap_vp_ids[index]) for index in ranked)
        reason = None
    return TopKQueryPlan(
        base_action_index=base_action,
        base_stop=False,
        executable_ghost_indices=ghosts,
        ranked_ghost_indices=ranked,
        ranked_ghost_ids=tuple(str(gmap_vp_ids[index]) for index in ranked),
        query_indices=query_indices,
        query_ghost_ids=query_ids,
        skip_reason=reason,
        bounded_result=proof,
    )
