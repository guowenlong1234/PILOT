"""Frozen active-lookahead policy helpers for GRPO."""

from __future__ import annotations

from collections.abc import Sequence

import torch


FROZEN_LOOKAHEAD_DISTRIBUTION_VERSION = (
    "etpr1-frozen-lookahead-stop-mass-v1"
)


def frozen_lookahead_probs(
    base_logits: torch.Tensor,
    frozen_delta: torch.Tensor,
    valid_action_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply frozen MOVE residuals while preserving base STOP probability.

    Index zero is STOP.  The residual only changes the conditional distribution
    over valid MOVE actions.  This makes a zero residual exactly recover the
    ordinary full softmax and prevents lookahead from creating or deleting STOP
    probability mass.
    """

    if base_logits.ndim != 2:
        raise ValueError("base_logits must be [batch, actions]")
    if frozen_delta.shape != base_logits.shape:
        raise ValueError("frozen_delta must match base_logits")
    if valid_action_mask.shape != base_logits.shape:
        raise ValueError("valid_action_mask must match base_logits")
    if base_logits.shape[1] < 1:
        raise ValueError("the action dimension must include STOP")

    valid = valid_action_mask.to(device=base_logits.device, dtype=torch.bool)
    if not bool(valid[:, 0].all()):
        raise ValueError("STOP must be valid for every row")
    if not torch.isfinite(frozen_delta[valid]).all():
        raise ValueError("valid frozen lookahead deltas must be finite")

    masked_base = base_logits.masked_fill(~valid, -torch.inf)
    base_probs = torch.softmax(masked_base, dim=-1)
    if not torch.isfinite(base_probs).all():
        raise ValueError("base policy produced a non-finite distribution")

    if base_logits.shape[1] == 1:
        return torch.ones_like(base_logits)

    move_valid = valid[:, 1:]
    has_move = move_valid.any(dim=-1)
    adjusted_move_logits = (
        base_logits[:, 1:] + frozen_delta.detach()[:, 1:]
    ).masked_fill(~move_valid, -torch.inf)
    safe_move_logits = torch.where(
        has_move[:, None],
        adjusted_move_logits,
        torch.zeros_like(adjusted_move_logits),
    )
    move_conditional = torch.softmax(safe_move_logits, dim=-1)
    move_conditional = move_conditional.masked_fill(~move_valid, 0.0)

    stop_prob = torch.where(
        has_move,
        base_probs[:, 0],
        torch.ones_like(base_probs[:, 0]),
    )
    move_probs = move_conditional * (1.0 - stop_prob)[:, None]
    result = torch.cat((stop_prob[:, None], move_probs), dim=-1)
    result = result.masked_fill(~valid, 0.0)
    if not torch.isfinite(result).all():
        raise ValueError("adjusted policy produced a non-finite distribution")
    return result


def resolve_executed_actions(
    sampled_actions: torch.Tensor,
    *,
    no_vp_left: Sequence[bool],
    final_step: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return actual high-level actions and rows valid for policy learning.

    Habitat forces STOP at the trajectory limit or when no ghost remains.  A
    sampled MOVE changed by that external rule was not drawn from the behavior
    distribution as STOP, so it must not enter the GRPO ratio calculation.
    """

    if sampled_actions.ndim != 1:
        raise ValueError("sampled_actions must be one-dimensional")
    if len(no_vp_left) != int(sampled_actions.shape[0]):
        raise ValueError("no_vp_left must match sampled_actions")
    forced = torch.tensor(
        [bool(final_step) or bool(value) for value in no_vp_left],
        dtype=torch.bool,
        device=sampled_actions.device,
    )
    sampled_stop = sampled_actions == 0
    externally_changed = forced & ~sampled_stop
    executed = sampled_actions.masked_fill(externally_changed, 0)
    return executed, ~externally_changed
