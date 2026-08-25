import pytest
import torch

from vlnce_baselines.nwm.active_lookahead.topk_query import (
    bounded_action_invariant,
    build_topk_query_plan,
    stable_topk_ghost_indices,
)


def test_stable_topk_uses_frozen_logits_and_gmap_order_for_ragged_ghosts():
    ids = [None, "visited-node", "g-history", "g-visible-a", "g-visible-b"]
    logits = torch.tensor([0.0, -torch.inf, 1.0, 1.0, 0.5, -99.0])

    assert stable_topk_ghost_indices(ids, logits, k=5) == (2, 3, 4)
    plan = build_topk_query_plan(
        ids, logits, k=2, delta_max=1.0, enable_bounded_skip=False
    )

    assert plan.base_action_index == 2
    assert plan.base_stop is False
    assert plan.executable_ghost_indices == (2, 3, 4)
    assert plan.ranked_ghost_indices == (2, 3)
    assert plan.ranked_ghost_ids == ("g-history", "g-visible-a")
    assert plan.query_indices == (2, 3)
    assert plan.query_ghost_ids == ("g-history", "g-visible-a")
    assert plan.skip_reason is None


def test_base_stop_skips_every_query_before_oracle_expansion():
    plan = build_topk_query_plan(
        [None, "g0", "g1"], [3.0, -float("inf"), 1.0], k=5, delta_max=1.0
    )

    assert plan.base_stop is True
    assert plan.base_action_index == 0
    assert plan.ranked_ghost_indices == ()
    assert plan.query_count == 0
    assert plan.skip_reason == "base_stop"
    assert plan.bounded_result is None


def test_margin_strictly_greater_than_two_delta_is_bounded_invariant():
    plan = build_topk_query_plan(
        [None, "g0", "g1", "g2"],
        [0.0, 3.0, 0.9, -4.0],
        k=2,
        delta_max=1.0,
    )

    assert plan.skip_reason == "bounded_invariant"
    assert plan.query_count == 0
    assert plan.ranked_ghost_indices == (1, 2)
    assert plan.ranked_ghost_ids == ("g0", "g1")
    assert plan.bounded_result.invariant is True
    assert plan.bounded_result.winner_lower_bound == pytest.approx(2.0)
    assert plan.bounded_result.max_competitor_upper_bound == pytest.approx(1.9)


def test_margin_equal_to_two_delta_is_not_a_safe_strict_proof():
    plan = build_topk_query_plan(
        [None, "g0", "g1"], [0.0, 3.0, 1.0], k=2, delta_max=1.0
    )

    assert plan.skip_reason is None
    assert plan.query_indices == (1, 2)
    assert plan.bounded_result.invariant is False
    assert plan.bounded_result.winner_lower_bound == pytest.approx(2.0)
    assert plan.bounded_result.max_competitor_upper_bound == pytest.approx(2.0)


def test_general_lower_upper_proof_accounts_for_unqueried_competitors():
    proof = bounded_action_invariant(
        [0.0, 3.0, 2.1, 0.0],
        executable_indices=[1, 2, 3],
        query_indices=[1],
        delta_max=1.0,
    )

    assert proof.winner_index == 1
    assert proof.winner_lower_bound == pytest.approx(2.0)
    assert proof.max_competitor_index == 2
    assert proof.max_competitor_upper_bound == pytest.approx(2.1)
    assert proof.invariant is False
    assert bool(proof) is False


def test_single_executable_ghost_is_always_bounded_invariant():
    proof = bounded_action_invariant(
        [0.0, 0.5], [1], [1], delta_max=1.0
    )

    assert proof.invariant is True
    assert proof.max_competitor_index is None
    assert proof.max_competitor_upper_bound == -float("inf")


def test_topk_rejects_invalid_budget_and_non_finite_executable_logits():
    with pytest.raises(ValueError, match="k must be positive"):
        stable_topk_ghost_indices([None, "g0"], [0.0, 1.0], k=0)
    with pytest.raises(ValueError, match="executable ghost logits must be finite"):
        stable_topk_ghost_indices([None, "g0"], [0.0, -float("inf")], k=1)
