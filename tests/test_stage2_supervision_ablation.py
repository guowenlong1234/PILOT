"""B ablation: expand action supervision without changing proxy penalties."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from vlnce_baselines.nwm.active_lookahead.offline_objective import (
    ALL_MOVE_WITH_FUTURE_ROWS,
    OfflineDecisionLossConfig,
    offline_decision_aware_loss,
)


def batch():
    # Row 0: teacher has future. Row 1: teacher outside TopK.
    # Row 2: teacher inside TopK, but its future is invalid.
    return dict(
        teacher_rank_in_topk=torch.tensor([0, -1, 1]),
        topk_valid_mask=torch.tensor([[True, True], [True, False], [True, False]]),
        teacher_valid=torch.ones(3, dtype=torch.bool),
        teacher_stop=torch.zeros(3, dtype=torch.bool),
        no_vp_left=torch.zeros(3, dtype=torch.bool),
        base_stop=torch.zeros(3, dtype=torch.bool),
        base_logits=torch.tensor([[0., 1., -1.], [1., 0., -1.], [1., 0., -1.]]),
        ghost_valid_mask=torch.ones(3, 3, dtype=torch.bool),
        topk_base_indices=torch.tensor([[1, 0], [0, -1], [0, 1]]),
        teacher_base_index=torch.tensor([1, 1, 1]),
    )


def configs():
    baseline = OfflineDecisionLossConfig(
        correct_row_weight=2., wrong_row_weight=1.,
        regularization_weight=.001, absent_noop_weight=.05,
    )
    return baseline, replace(baseline, decision_row_policy=ALL_MOVE_WITH_FUTURE_ROWS)


def test_expansion_matches_full_ghost_ce_and_pair_reference():
    delta = torch.tensor([[.2, -.1], [.3, 0.], [.4, 0.]], requires_grad=True)
    b = batch()
    baseline, expanded = configs()
    old = offline_decision_aware_loss(delta, **b, config=baseline)
    new = offline_decision_aware_loss(delta, **b, config=expanded)
    assert old.decision_sample_count == 1
    assert new.decision_sample_count == 3
    assert new.decision_base_correct_count == 1
    assert new.decision_base_wrong_count == 2
    adjusted = torch.stack((
        b['base_logits'][0] + torch.stack((delta[0, 1], delta[0, 0], delta[0, 0]*0)),
        b['base_logits'][1] + torch.stack((delta[1, 0], delta[1, 0]*0, delta[1, 0]*0)),
        b['base_logits'][2] + torch.stack((delta[2, 0], delta[2, 0]*0, delta[2, 0]*0)),
    ))
    weights = torch.tensor([2., 1., 1.])
    expected_ce = (F.cross_entropy(adjusted, b['teacher_base_index'], reduction='none') * weights).sum()/4
    expected_pair = (F.softplus((.25-adjusted[:, 1]+adjusted[:, [0, 2]].max(1).values)/.25)*weights).sum()/4
    torch.testing.assert_close(new.final_loss, expected_ce)
    torch.testing.assert_close(new.pair_loss, expected_pair)
    for name in ('signed_loss', 'positive_loss', 'negative_loss', 'absent_loss',
                 'regularization_loss', 'absent_noop_loss'):
        assert torch.equal(getattr(old, name), getattr(new, name)), name
    assert new.regularization_candidate_count == old.regularization_candidate_count == 2
    assert new.absent_noop_candidate_count == old.absent_noop_candidate_count == 2
    actual_grad, = torch.autograd.grad(new.final_loss + new.pair_loss, delta, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected_ce + expected_pair, delta)
    torch.testing.assert_close(actual_grad, expected_grad)
    # Minimizing either action loss suppresses each adjustable wrong move;
    # invalid futures remain exactly untouched.
    assert (actual_grad[1:, 0] > 0).all()
    assert torch.equal(actual_grad[1:, 1], torch.zeros(2))


@pytest.mark.parametrize('field,value', [
    ('teacher_valid', False), ('teacher_stop', True), ('base_stop', True),
    ('no_vp_left', True), ('topk_valid_mask', False), ('teacher_base_index', -1),
])
def test_expanded_supervision_still_excludes_non_move_or_unmodifiable_rows(field, value):
    b = batch()
    b[field][1:] = value
    delta = torch.tensor([[.2, -.1], [.3, 0.], [.4, 0.]], requires_grad=True)
    result = offline_decision_aware_loss(delta, **b, config=configs()[1])
    assert result.decision_sample_count == 1
    gradient, = torch.autograd.grad(result.final_loss + result.pair_loss, delta)
    assert torch.equal(gradient[1:], torch.zeros(2, 2))


def test_missing_teacher_future_only_batch_has_finite_zero_original_regularizer():
    b = {k: v[1:] for k, v in batch().items()}
    delta = torch.tensor([[.3, 0.], [.4, 0.]], requires_grad=True)
    result = offline_decision_aware_loss(delta, **b, config=configs()[1])
    assert result.decision_sample_count == 2
    assert result.regularization_candidate_count == 0
    assert result.regularization_loss.item() == 0
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(delta.grad).all()


def test_expanded_supervision_rejects_padded_teacher():
    b = batch()
    b['ghost_valid_mask'][1, 1] = False
    with pytest.raises(ValueError, match='teacher selects a padded ghost'):
        offline_decision_aware_loss(torch.zeros(3, 2), **b, config=configs()[1])


def test_present_teacher_population_is_exact_baseline():
    b = {k: v[:1] for k, v in batch().items()}
    delta = torch.tensor([[.2, -.1]], requires_grad=True)
    old, new = [offline_decision_aware_loss(delta, **b, config=c) for c in configs()]
    for key, value in vars(old).items():
        other = getattr(new, key)
        assert torch.equal(value, other) if torch.is_tensor(value) else value == other


def test_coverage_policy_roundtrip_and_unknown_rejection():
    _, expanded = configs()
    assert OfflineDecisionLossConfig.from_mapping(expanded.to_dict()) == expanded
    with pytest.raises(ValueError, match='decision_row_policy'):
        replace(expanded, decision_row_policy='all_rows')
