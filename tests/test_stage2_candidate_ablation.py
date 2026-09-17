"""Separate real TopK context from permission to apply a future residual."""
import copy

import pytest
import torch

from vlnce_baselines.nwm.active_lookahead.residual_head import (
    InterleavedCrossModalTopKFutureLogitResidualHead as Head,
)


def make_head(**kwargs):
    torch.manual_seed(17)
    head = Head(input_dim=8, hidden_dim=8, num_queries=2,
                num_attention_heads=2, ffn_dim=16, fusion_layers=2,
                dropout=0, **kwargs).eval()
    with torch.no_grad():
        head.score_mlp[-1].weight.normal_(std=0.2)
    return head


def inputs():
    torch.manual_seed(5)
    return dict(
        owner_embeddings=torch.randn(1, 4, 8),
        text_tokens=torch.randn(1, 4, 3, 8),
        future_tokens=torch.randn(1, 4, 5, 8),
        # The frozen winner (slot 0) has no valid future; slot 3 is padding.
        base_ghost_log_probs=torch.tensor([[-0.1, -0.4, -0.9, 0.0]]),
        candidate_valid_mask=torch.tensor([[False, True, True, False]]),
        candidate_present_mask=torch.tensor([[True, True, True, False]]),
        candidate_geometry=torch.randn(1, 4, 3),
    )


def test_default_preserves_checkpoint_schema_and_ignores_presence_mask():
    head = make_head()
    explicit = make_head(candidate_context_mode='future_valid')
    variant = make_head(candidate_context_mode='all_present')
    explicit.load_state_dict(head.state_dict(), strict=True)
    variant.load_state_dict(head.state_dict(), strict=True)
    assert list(head.state_dict()) == list(variant.state_dict())
    data = inputs()
    expected = head.forward_topk_from_log_probs(**data)
    data.pop('candidate_present_mask')
    actual = explicit.forward_topk_from_log_probs(**data)
    for key in ('raw_delta', 'delta', 'future_summary', 'base_ghost_log_prob'):
        assert torch.equal(getattr(expected, key), getattr(actual, key))


def test_all_present_uses_real_frozen_winner_but_does_not_modify_it():
    head = make_head(candidate_context_mode='all_present')
    captured = []
    handle = head.score_mlp.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone()))
    data = inputs()
    result = head.forward_topk_from_log_probs(**data)
    handle.remove()
    score = captured[0]
    assert score.shape[0] == 3
    torch.testing.assert_close(score[:, -3], torch.tensor([0., -0.3, -0.8]))
    torch.testing.assert_close(score[:, -2], torch.full((3,), 0.3 / 1.3))
    assert torch.equal(score[:, -1], torch.tensor([1., 0., 0.]))
    invalid = ~data['candidate_valid_mask']
    assert torch.count_nonzero(result.raw_delta[invalid]) == 0
    assert torch.count_nonzero(result.delta[invalid]) == 0
    assert torch.count_nonzero(result.future_summary[0, 0]) > 0


@pytest.mark.parametrize('field', ['owner_embeddings', 'text_tokens', 'candidate_geometry',
                                 'base_ghost_log_probs'])
def test_nonfuture_candidate_context_changes_valid_candidate_scores(field):
    head = make_head(candidate_context_mode='all_present')
    data = inputs()
    before = head.forward_topk_from_log_probs(**data).delta
    changed = copy.deepcopy(data)
    # Avoid uniform owner offsets, which LayerNorm intentionally removes.
    changed[field][0, 0] *= -3
    after = head.forward_topk_from_log_probs(**changed).delta
    assert not torch.allclose(before, after)


def test_invalid_future_and_padding_cannot_affect_scores_or_future_gradients():
    head = make_head(candidate_context_mode='all_present')
    data = inputs()
    mask = torch.ones(1, 4, 5, dtype=torch.bool)
    mask[~data['candidate_valid_mask']] = False
    data['future_token_mask'] = mask
    before = head.forward_topk_from_log_probs(**data).delta
    changed = copy.deepcopy(data)
    changed['future_tokens'][~data['candidate_valid_mask']] = torch.nan
    for key in ('owner_embeddings', 'text_tokens', 'candidate_geometry', 'base_ghost_log_probs'):
        changed[key][0, 3] = 10000
    after = head.forward_topk_from_log_probs(**changed).delta
    assert torch.equal(before, after)

    data['future_tokens'].requires_grad_(True)
    output = head.forward_topk_from_log_probs(**data, detach_future_tokens=False)
    output.delta.sum().backward()
    grad = data['future_tokens'].grad
    assert torch.count_nonzero(grad[~data['candidate_valid_mask']]) == 0
    assert torch.count_nonzero(grad[data['candidate_valid_mask']]) > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


def test_no_future_keeps_all_outputs_zero_and_context_finite():
    head = make_head(candidate_context_mode='all_present')
    data = inputs()
    data['candidate_valid_mask'].zero_()
    data['future_tokens'].fill_(torch.nan)
    data['future_token_mask'] = torch.zeros(1, 4, 5, dtype=torch.bool)
    output = head.forward_topk_from_log_probs(**data)
    assert torch.count_nonzero(output.delta) == 0
    assert torch.count_nonzero(output.raw_delta) == 0
    assert torch.isfinite(output.future_summary).all()
    data['candidate_present_mask'].zero_()
    empty = head.forward_topk_from_log_probs(**data)
    assert torch.count_nonzero(empty.delta) == 0
    assert torch.count_nonzero(empty.future_summary) == 0


def test_presence_contract_rejects_missing_mask_and_invalid_future_membership():
    head = make_head(candidate_context_mode='all_present')
    data = inputs()
    data.pop('candidate_present_mask')
    with pytest.raises(ValueError, match='requires candidate_present_mask'):
        head.forward_topk_from_log_probs(**data)
    data['candidate_present_mask'] = torch.zeros(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match='must be present'):
        head.forward_topk_from_log_probs(**data)
