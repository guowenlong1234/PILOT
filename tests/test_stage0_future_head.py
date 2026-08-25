import pytest
import torch

from vlnce_baselines.nwm.active_lookahead.residual_head import (
    InterleavedCrossModalTopKFutureLogitResidualHead,
)


def _inputs(dim=8):
    torch.manual_seed(9)
    return (
        torch.randn(2, 3, dim, requires_grad=True),
        torch.randn(2, 5, dim, requires_grad=True),
        torch.randn(2, 3, 4, dim, requires_grad=True),
        torch.tensor(
            [[-0.2, -0.4, -1.2], [-0.1, -2.0, -3.0]],
            requires_grad=True,
        ),
        torch.tensor([[True, True, True], [True, True, False]]),
        torch.tensor(
            [
                [[0.4, 0.0, 1.0], [0.5, 1.0, 0.0], [0.6, 0.0, -1.0]],
                [[0.7, -1.0, 0.0], [0.8, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ]
        ),
    )


def _head(**overrides):
    values = dict(
        input_dim=8,
        hidden_dim=8,
        num_queries=2,
        num_attention_heads=2,
        ffn_dim=16,
        dropout=0,
        fusion_layers=3,
        score_context="base_bounded_margin_relative",
        round_weight_sharing="independent",
    )
    values.update(overrides)
    return InterleavedCrossModalTopKFutureLogitResidualHead(**values)


def test_e24_head_shape_zero_initialization_and_frozen_inputs():
    head = _head()
    owner, text, future, log_probs, valid, geometry = _inputs()
    output = head.forward_topk_from_log_probs(
        owner,
        text,
        future,
        log_probs,
        valid,
        candidate_geometry=geometry,
    )

    assert output.delta.shape == valid.shape
    assert output.future_summary.shape == (2, 3, 8)
    assert torch.count_nonzero(output.delta) == 0
    assert output.delta[1, 2] == 0
    output.delta.sum().backward()
    assert owner.grad is None
    assert text.grad is None
    assert future.grad is None
    assert log_probs.grad is None


def test_e24_head_uses_bounded_winner_margin_context():
    head = _head().eval()
    with torch.no_grad():
        head.score_mlp[0].weight.zero_()
        head.score_mlp[0].bias.zero_()
        head.score_mlp[0].weight[0, -2] = 1.0
        head.score_mlp[-1].weight.zero_()
        head.score_mlp[-1].bias.zero_()
        head.score_mlp[-1].weight[0, 0] = 1.0

    owner, text, future, log_probs, valid, geometry = _inputs()
    output = head.forward_topk_from_log_probs(
        owner,
        text,
        future,
        log_probs,
        valid,
        candidate_geometry=geometry,
    )
    valid_lp = log_probs.detach().masked_fill(~valid, -torch.inf)
    top_two = valid_lp.topk(k=2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    bounded = margin / (1.0 + margin)
    expected_input = bounded[:, None].expand_as(log_probs).masked_fill(~valid, 0.0)
    expected_raw = torch.nn.functional.gelu(expected_input)
    torch.testing.assert_close(output.raw_delta, expected_raw)
    torch.testing.assert_close(output.delta, torch.tanh(expected_raw))


def test_e24_head_keeps_masked_tokens_and_candidates_isolated():
    head = _head().eval()
    with torch.no_grad():
        head.score_mlp[-1].weight[0, 0] = 1.0
    owner, text, future, log_probs, valid, geometry = _inputs()
    text_mask = torch.ones(2, 5, dtype=torch.bool)
    text_mask[:, -1] = False
    future_mask = torch.ones(2, 3, 4, dtype=torch.bool)
    future_mask[:, :, -1] = False
    first = head.forward_topk_from_log_probs(
        owner,
        text,
        future,
        log_probs,
        valid,
        text_token_mask=text_mask,
        future_token_mask=future_mask,
        candidate_geometry=geometry,
    ).delta

    changed_owner = owner.detach().clone()
    changed_text = text.detach().clone()
    changed_future = future.detach().clone()
    changed_geometry = geometry.clone()
    changed_owner[~valid] = 1e4
    changed_text[:, -1] = 1e4
    changed_future[~valid] = -1e4
    changed_future[:, :, -1] = -1e4
    changed_geometry[~valid] = 1e4
    second = head.forward_topk_from_log_probs(
        changed_owner,
        changed_text,
        changed_future,
        log_probs.detach(),
        valid,
        text_token_mask=text_mask,
        future_token_mask=future_mask,
        candidate_geometry=changed_geometry,
    ).delta
    torch.testing.assert_close(second[valid], first[valid])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("delta_centering", "valid_raw_mean"),
        ("score_context", "base_relative"),
        ("round_weight_sharing", "shared"),
        ("residual_confidence_gate", "full"),
    ],
)
def test_e24_head_rejects_removed_experimental_variants(name, value):
    with pytest.raises(ValueError, match="E24 scorer requires"):
        _head(**{name: value})


def test_e24_head_requires_geometry_and_valid_dimensions():
    owner, text, future, log_probs, valid, _ = _inputs()
    with pytest.raises(ValueError, match="requires candidate q0 geometry"):
        _head().forward_topk_from_log_probs(
            owner, text, future, log_probs, valid
        )
    with pytest.raises(ValueError, match="fusion_layers"):
        _head(fusion_layers=0)
