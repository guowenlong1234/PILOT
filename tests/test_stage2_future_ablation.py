"""Future-content ablation keeps every non-content part of Stage-2 fixed."""
import copy

import pytest
import torch

from vlnce_baselines.nwm.active_lookahead.residual_head import (
    InterleavedCrossModalTopKFutureLogitResidualHead as Head,
)
from vlnce_baselines.nwm.active_lookahead.stage2_training import (
    MODEL_CONFIG, experiment_configs,
)


def make_head(mode):
    torch.manual_seed(23)
    head = Head(input_dim=8, hidden_dim=8, num_queries=2,
                num_attention_heads=2, ffn_dim=16, fusion_layers=2,
                dropout=0, future_mode=mode).eval()
    with torch.no_grad():
        head.score_mlp[-1].weight.normal_(std=.2)
    return head


def inputs():
    torch.manual_seed(7)
    return dict(owner_embeddings=torch.randn(2, 3, 8),
        text_tokens=torch.randn(2, 4, 8), future_tokens=torch.randn(2, 3, 5, 8),
        base_ghost_log_probs=torch.randn(2, 3),
        candidate_valid_mask=torch.tensor([[True, False, True], [False, True, True]]),
        candidate_geometry=torch.randn(2, 3, 3))


def test_none_has_identical_schema_and_seeded_initialization():
    full, none = make_head('full'), make_head('none')
    assert full.future_mode == 'full' and none.future_mode == 'none'
    assert list(full.state_dict()) == list(none.state_dict())
    assert all(torch.equal(value, none.state_dict()[name])
               for name, value in full.state_dict().items())


def test_none_is_invariant_to_future_content_but_keeps_valid_mask():
    head = make_head('none')
    data = inputs()
    expected = head.forward_topk_from_log_probs(**data).delta
    changed = copy.deepcopy(data)
    changed['future_tokens'].normal_(mean=1000, std=500)
    actual = head.forward_topk_from_log_probs(**changed).delta
    assert torch.equal(expected, actual)

    changed_mask = copy.deepcopy(data)
    changed_mask['candidate_valid_mask'][0, 0] = False
    with_different_support = head.forward_topk_from_log_probs(**changed_mask).delta
    assert torch.count_nonzero(with_different_support[0, 0]) == 0
    assert not torch.equal(expected, with_different_support)


def test_none_blocks_future_gradients_and_accepts_nan_content():
    head = make_head('none')
    data = inputs()
    data['future_tokens'].requires_grad_(True)
    data['future_tokens'].data[~data['candidate_valid_mask']] = torch.nan
    result = head.forward_topk_from_log_probs(**data, detach_future_tokens=False)
    assert torch.isfinite(result.delta).all()
    result.delta.sum().backward()
    assert data['future_tokens'].grad is None or not bool(data['future_tokens'].grad.any())


def test_experiment_and_checkpoint_config_explicitly_distinguish_modes():
    full, full_loss = experiment_configs(future_mode='full')
    none, none_loss = experiment_configs(future_mode='none')
    assert MODEL_CONFIG['future_mode'] == full['future_mode'] == 'full'
    assert none['future_mode'] == 'none'
    assert {k: v for k, v in full.items() if k != 'future_mode'} == {
        k: v for k, v in none.items() if k != 'future_mode'}
    assert full_loss == none_loss
    with pytest.raises(ValueError, match='future_mode'):
        experiment_configs(future_mode='invalid')
    with pytest.raises(ValueError, match='future_mode'):
        make_head('invalid')
