"""Whole-decision pruning for bounded, greedy, STOP-isolated deployment."""
import math

import torch

from .topk_query import build_topk_query_plan


def deployment_residual_bound(gain, delta_max, clip=1.0):
    values = (float(gain), float(delta_max), float(clip))
    if not all(math.isfinite(x) for x in values) or min(values[1:]) < 0:
        raise ValueError('residual bounds must be finite and nonnegative')
    return min(values[2], abs(values[0]) * values[1])


def inference_query_plans(ids_by_env, logits, no_vp_left, step, max_len, *, bound):
    """Return certificates without predicting futures or touching any RNG.

    The arithmetic guard covers conversion of a bounded residual and the final
    score addition. Near ties deliberately retain the full computation. This
    certifies argmax only, never a sampled probability distribution.
    """
    scores = logits.detach().cpu()
    plans = []
    for i, ids in enumerate(ids_by_env):
        if step == max_len - 1 or bool(no_vp_left[i]):
            plans.append(dict(skip_reason='forced_stop', margin=None, numerical_guard=None))
            continue
        row = scores[i, :len(ids)]
        # The legacy numpy selector cannot consume bfloat16 directly.
        plan = build_topk_query_plan(ids, row.double(), k=5, delta_max=bound)
        proof = plan.bounded_result
        if proof is None:
            plans.append(dict(skip_reason=plan.skip_reason, margin=None, numerical_guard=None))
            continue
        margin = proof.winner_lower_bound - proof.max_competitor_upper_bound
        scale = float(row[list(plan.executable_ghost_indices)].double().abs().max())
        guard = 4 * torch.finfo(row.dtype).eps * (1 + scale + bound)
        reason = None
        if len(plan.executable_ghost_indices) == 1:
            reason = 'single_candidate'
        elif proof.invariant and margin > guard:
            reason = 'bounded_invariant'
        plans.append(dict(skip_reason=reason, margin=margin if math.isfinite(margin) else None,
                          numerical_guard=guard))
    return plans
