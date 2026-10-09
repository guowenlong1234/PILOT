"""Certified and configurable margin pruning for greedy STOP-isolated deployment."""
import math

import torch

from .topk_query import build_topk_query_plan


def deployment_residual_bound(gain, delta_max, clip=1.0):
    values = (float(gain), float(delta_max), float(clip))
    if not all(math.isfinite(x) for x in values) or min(values[1:]) < 0:
        raise ValueError('residual bounds must be finite and nonnegative')
    return min(values[2], abs(values[0]) * values[1])


def resolve_margin_threshold(value):
    """-1/None retain certificate-only behavior; otherwise use a logit gap."""
    if value is None or float(value)==-1:
        return None
    threshold=float(value)
    if not math.isfinite(threshold) or threshold<0:
        raise ValueError('margin threshold must be finite and nonnegative, or -1')
    return threshold


def inference_query_plans(ids_by_env, logits, no_vp_left, step, max_len, *, bound,
                          margin_threshold=None):
    """Return a query gate and a separate certificate without consuming RNG.

    The arithmetic guard covers conversion of a bounded residual and the final
    score addition. Near ties deliberately retain the full computation. This
    certifies argmax only, never a sampled probability distribution. The
    optional tighter margin threshold is empirical, not a safety certificate.
    """
    threshold=resolve_margin_threshold(margin_threshold)
    scores = logits.detach().cpu()
    plans = []
    for i, ids in enumerate(ids_by_env):
        if step == max_len - 1 or bool(no_vp_left[i]):
            plans.append(dict(skip_reason='forced_stop', certified_skip_reason='forced_stop',
                              margin=None, base_margin=None, numerical_guard=None,margin_threshold=threshold))
            continue
        row = scores[i, :len(ids)]
        # The legacy numpy selector cannot consume bfloat16 directly.
        plan = build_topk_query_plan(ids, row.double(), k=5, delta_max=bound)
        proof = plan.bounded_result
        if proof is None:
            plans.append(dict(skip_reason=plan.skip_reason,certified_skip_reason=plan.skip_reason,
                              margin=None,base_margin=None,numerical_guard=None,margin_threshold=threshold))
            continue
        margin = proof.winner_lower_bound - proof.max_competitor_upper_bound
        scale = float(row[list(plan.executable_ghost_indices)].double().abs().max())
        guard = 4 * torch.finfo(row.dtype).eps * (1 + scale + bound)
        reason = None
        if len(plan.executable_ghost_indices) == 1:
            reason = 'single_candidate'
        elif proof.invariant and margin > guard:
            reason = 'bounded_invariant'
        ordered=sorted((float(row[j]) for j in plan.executable_ghost_indices),reverse=True)
        base_margin=ordered[0]-ordered[1] if len(ordered)>1 else None
        certified_reason=reason
        if reason is None and threshold is not None and base_margin>threshold+guard:
            reason='margin_threshold'
        plans.append(dict(skip_reason=reason,certified_skip_reason=certified_reason,
                          margin=margin if math.isfinite(margin) else None,
                          base_margin=base_margin,margin_threshold=threshold,numerical_guard=guard))
    return plans
