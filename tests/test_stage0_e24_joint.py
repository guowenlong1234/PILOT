from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest
from torch.nn.parallel import DistributedDataParallel as DDP

from vlnce_baselines.nwm.active_lookahead.joint_e24 import (
    E24JointDecisionPack,
    E24JointTrainModule,
    _forward_head,
    attach_e24_joint_targets,
    build_e24_joint_step,
    e24_joint_denominators,
    forward_e24_joint_batch,
    forward_native_adjusted_batch,
    joint_action_scale,
    load_e24_joint_head,
    make_e24_joint_dummy_batch,
    normalized_e24_joint_loss,
    normalized_native_adjusted_loss,
    slice_e24_joint_batch,
)
from vlnce_baselines.nwm.active_lookahead.offline_objective import (
    OfflineDecisionLossConfig,
)
from vlnce_baselines.nwm.active_lookahead.native_cls_adapter import (
    Top5NativeClsAdapter,
    expand_native_cls_condition,
)
from vlnce_baselines.nwm.active_lookahead.online_e24 import (
    stop_isolated_e24_actions,
)
from vlnce_baselines.nwm.active_lookahead.residual_head import (
    InterleavedCrossModalTopKFutureLogitResidualHead,
)


def _head():
    return InterleavedCrossModalTopKFutureLogitResidualHead(
        input_dim=8,
        hidden_dim=8,
        num_queries=2,
        num_attention_heads=2,
        ffn_dim=16,
        dropout=0.0,
        fusion_layers=1,
    )


def _loss_config():
    return OfflineDecisionLossConfig(
        signed_weight=0.25,
        final_weight=1.0,
        pair_weight=0.5,
        regularization_weight=0.001,
        absent_noop_weight=0.05,
        correct_row_weight=2.0,
        wrong_row_weight=1.0,
    )


def _batch():
    torch.manual_seed(24)
    batch = 4
    topk = 3
    return {
        "owner_embeddings": torch.randn(batch, topk, 8, requires_grad=True),
        "text_tokens": torch.randn(batch, 5, 8, requires_grad=True),
        "text_token_mask": torch.ones(batch, 5, dtype=torch.bool),
        "future_tokens": torch.randn(batch, topk, 4, 8, requires_grad=True),
        "topk_slot_mask": torch.ones(batch, topk, dtype=torch.bool),
        "topk_valid_mask": torch.tensor(
            [[True, True, True], [True, True, True], [True, False, True], [True, True, True]]
        ),
        "candidate_q0_geometry": torch.randn(batch, topk, 3, requires_grad=True),
        "base_logits": torch.tensor(
            [[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0],
             [3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]],
            requires_grad=True,
        ),
        "ghost_valid_mask": torch.ones(batch, 4, dtype=torch.bool),
        "topk_base_indices": torch.tensor([[0, 1, 2]] * batch),
        # present, absent, present-but-invalid-future, STOP
        "teacher_rank_in_topk": torch.tensor([1, -1, 1, -1]),
        "teacher_valid": torch.ones(batch, dtype=torch.bool),
        "teacher_stop": torch.tensor([False, False, False, True]),
        "no_vp_left": torch.zeros(batch, dtype=torch.bool),
        "base_stop": torch.zeros(batch, dtype=torch.bool),
        "teacher_base_index": torch.tensor([1, 3, 1, -1]),
    }


def test_native_cls_condition_uses_normalized_xy_and_trigonometric_yaw():
    condition = torch.tensor([[0.25, -0.5, torch.pi / 2.0, 0.125]])

    expanded = expand_native_cls_condition(condition)

    torch.testing.assert_close(
        expanded,
        torch.tensor([[0.25, -0.5, 1.0, 0.0, 0.125]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_native_cls_adapter_starts_identity_and_only_changes_token_zero():
    adapter = Top5NativeClsAdapter(feature_dim=8, condition_hidden_dim=4)
    tokens = torch.randn(2, 3, 257, 8)
    condition = torch.randn(2, 3, 4)

    initial, diagnostics = adapter.adapt_tokens(tokens, condition)

    assert torch.equal(initial, tokens)
    assert diagnostics["cls_delta_norm"].shape == (2, 3)

    with torch.no_grad():
        adapter.fusion[-1].bias.fill_(0.5)
    changed, _ = adapter.adapt_tokens(tokens, condition)

    torch.testing.assert_close(changed[..., 0, :], tokens[..., 0, :] + 0.5)
    assert torch.equal(changed[..., 1:, :], tokens[..., 1:, :])


def test_native_cls_adapter_has_nonzero_gradient_after_identity_start():
    adapter = Top5NativeClsAdapter(feature_dim=8, condition_hidden_dim=4)
    cls = torch.randn(4, 8)
    condition = torch.randn(4, 4)

    output, _ = adapter(cls, condition)
    output.square().mean().backward()

    assert adapter.fusion[-1].weight.grad is not None
    assert torch.count_nonzero(adapter.fusion[-1].weight.grad) > 0


def test_joint_action_warmup_uses_checkpoint_iteration_boundaries():
    assert joint_action_scale(13800, 13800, 400) == 0.0
    assert joint_action_scale(14000, 13800, 400) == 0.5
    assert joint_action_scale(14200, 13800, 400) == 1.0
    assert joint_action_scale(15000, 13800, 400) == 1.0


def test_joint_targets_cover_teacher_present_absent_and_stop():
    pack = E24JointDecisionPack(
        env_indices=torch.tensor([0, 1, 2]),
        owner_embeddings=torch.zeros(3, 2, 8),
        text_tokens=torch.zeros(3, 1, 8),
        text_token_mask=torch.ones(3, 1, dtype=torch.bool),
        future_tokens=torch.zeros(3, 2, 4, 8),
        future_valid_mask=torch.ones(3, 2, dtype=torch.bool),
        topk_slot_mask=torch.ones(3, 2, dtype=torch.bool),
        candidate_geometry=torch.zeros(3, 2, 3),
        base_logits=torch.zeros(3, 3),
        ghost_valid_mask=torch.ones(3, 3, dtype=torch.bool),
        ghost_global_indices=torch.tensor([[2, 4, 7]] * 3),
        topk_base_indices=torch.tensor([[0, 1]] * 3),
        topk_global_indices=torch.tensor([[2, 4]] * 3),
    )
    labeled = attach_e24_joint_targets(
        pack, torch.tensor([4, 7, 0]), [False, False, False]
    )
    assert labeled.teacher_rank_in_topk.tolist() == [1, -1, -1]
    assert labeled.teacher_base_index.tolist() == [1, 2, -1]
    assert labeled.teacher_valid.tolist() == [True, True, True]
    assert labeled.teacher_stop.tolist() == [False, False, True]


def test_predicted_future_source_never_calls_oracle_environment(monkeypatch):
    import vlnce_baselines.nwm.active_lookahead.joint_e24 as joint

    record = SimpleNamespace(candidate_forward_m=1.0)
    cfg = SimpleNamespace(
        source="dino_cwp_nwm",
        offline_topk=5,
        e24_train_delta_scale=1.0,
    )

    class _Env:
        num_envs = 1

        def call(self, *_args, **_kwargs):
            raise AssertionError("predicted future source must not call Oracle env")

    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        envs=_Env(),
        gmaps=[SimpleNamespace(select_candidate_q0=lambda _ids: record)],
        _active_lookahead_config=lambda: cfg,
        _e24_joint_head_state_module=lambda: _head().eval(),
    )
    monkeypatch.setattr(
        joint,
        "build_dino_cwp_nwm_future_tokens",
        lambda _trainer, **kwargs: (
            torch.ones(1, 5, 4, 8),
            torch.zeros(1, 5, 4),
            torch.tensor([[True, False, False, False, False]]),
            {"topk_slots": 1.0, "future_valid": 1.0},
        ),
    )
    monkeypatch.setattr(
        joint, "candidate_q0_geometry", lambda records: torch.zeros(len(records), 3)
    )

    _deltas, counts, pack, diagnostics = build_e24_joint_step(
        trainer,
        nav_inputs={"gmap_vp_ids": [[None, "g1"]]},
        nav_outs={
            "global_logits": torch.tensor([[0.0, 1.0]]),
            "gmap_embeds": torch.randn(1, 2, 8),
        },
        txt_embeds=torch.randn(1, 2, 8),
        txt_masks=torch.ones(1, 2, dtype=torch.bool),
    )
    assert counts == [1]
    assert pack is not None and pack.future_valid_mask[0, 0]
    assert diagnostics["future_valid"] == 1.0


def test_native_cls_base_stop_without_ghosts_skips_lookahead():
    cfg = SimpleNamespace(
        source="dino_cwp_nwm",
        offline_topk=5,
    )
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        envs=SimpleNamespace(num_envs=1),
        raenwm_runtime=SimpleNamespace(predict_cls_token=True),
        _active_lookahead_config=lambda: cfg,
        _e24_joint_head_state_module=lambda: _head().eval(),
    )

    deltas, counts, pack, diagnostics = build_e24_joint_step(
        trainer,
        nav_inputs={"gmap_vp_ids": [[None, "visited-node"]]},
        nav_outs={
            "global_logits": torch.tensor([[0.0, -torch.inf]]),
            "gmap_embeds": torch.randn(1, 2, 8),
        },
        txt_embeds=torch.randn(1, 2, 8),
        txt_masks=torch.ones(1, 2, dtype=torch.bool),
    )

    torch.testing.assert_close(deltas, torch.zeros_like(deltas))
    assert counts == [0]
    assert pack is None
    assert all(value == 0.0 for value in diagnostics.values())


def test_native_cls_base_move_without_executable_ghost_still_fails():
    cfg = SimpleNamespace(
        source="dino_cwp_nwm",
        offline_topk=5,
    )
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        envs=SimpleNamespace(num_envs=1),
        raenwm_runtime=SimpleNamespace(predict_cls_token=True),
        _active_lookahead_config=lambda: cfg,
        _e24_joint_head_state_module=lambda: _head().eval(),
    )

    with pytest.raises(
        RuntimeError, match="base MOVE row has no executable ghost"
    ):
        build_e24_joint_step(
            trainer,
            nav_inputs={"gmap_vp_ids": [[None, "visited-node"]]},
            nav_outs={
                "global_logits": torch.tensor([[0.0, 1.0]]),
                "gmap_embeds": torch.randn(1, 2, 8),
            },
            txt_embeds=torch.randn(1, 2, 8),
            txt_masks=torch.ones(1, 2, dtype=torch.bool),
        )


def test_e24_loss_only_updates_head_and_navigation_loss_does_not():
    head = E24JointTrainModule(_head())
    batch = _batch()
    result, _ = forward_e24_joint_batch(head, batch, loss_config=_loss_config())
    result.loss.backward()
    assert any(parameter.grad is not None for parameter in head.parameters())
    for name in (
        "owner_embeddings",
        "text_tokens",
        "future_tokens",
        "candidate_q0_geometry",
        "base_logits",
    ):
        assert batch[name].grad is None

    head.zero_grad(set_to_none=True)
    policy_logits = torch.randn(3, 4, requires_grad=True)
    torch.nn.functional.cross_entropy(policy_logits, torch.tensor([0, 1, 2])).backward()
    assert policy_logits.grad is not None
    assert all(parameter.grad is None for parameter in head.parameters())


def test_rollout_head_forward_promotes_fp16_pack_to_head_dtype():
    pack = E24JointDecisionPack(
        env_indices=torch.tensor([0]),
        owner_embeddings=torch.randn(1, 2, 8, dtype=torch.float16),
        text_tokens=torch.randn(1, 3, 8, dtype=torch.float16),
        text_token_mask=torch.ones(1, 3, dtype=torch.bool),
        future_tokens=torch.randn(1, 2, 4, 8, dtype=torch.float16),
        future_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        topk_slot_mask=torch.ones(1, 2, dtype=torch.bool),
        candidate_geometry=torch.randn(1, 2, 3, dtype=torch.float16),
        base_logits=torch.tensor([[2.0, 1.0]], dtype=torch.float16),
        ghost_valid_mask=torch.ones(1, 2, dtype=torch.bool),
        ghost_global_indices=torch.tensor([[1, 2]]),
        topk_base_indices=torch.tensor([[0, 1]]),
        topk_global_indices=torch.tensor([[1, 2]]),
    )
    with torch.inference_mode():
        deltas = _forward_head(_head().eval(), pack)
    assert deltas.dtype == torch.float32
    assert deltas.shape == (1, 2)


def test_delayed_microbatch_loss_and_gradient_match_direct_batch():
    config = _loss_config()
    batch = _batch()
    direct_head = E24JointTrainModule(_head())
    replay_head = deepcopy(direct_head)
    denominators = e24_joint_denominators(batch, config)

    direct_result, _ = forward_e24_joint_batch(
        direct_head, batch, loss_config=config
    )
    direct_loss = normalized_e24_joint_loss(
        direct_result,
        global_denominators=denominators,
        config=config,
        world_size=1,
        loss_weight=1.0,
    )
    direct_loss.backward()

    replay_loss = torch.zeros(())
    for start in range(0, 4, 2):
        micro = slice_e24_joint_batch(batch, start, start + 2)
        result, _ = forward_e24_joint_batch(
            replay_head, micro, loss_config=config
        )
        loss = normalized_e24_joint_loss(
            result,
            global_denominators=denominators,
            config=config,
            world_size=1,
            loss_weight=1.0,
        )
        replay_loss = replay_loss + loss.detach()
        loss.backward()

    torch.testing.assert_close(replay_loss, direct_loss.detach(), rtol=2e-5, atol=2e-6)
    for direct, replay in zip(direct_head.parameters(), replay_head.parameters()):
        torch.testing.assert_close(replay.grad, direct.grad, rtol=2e-5, atol=2e-6)


def test_dummy_replay_is_parameter_zero_but_backward_safe():
    head = E24JointTrainModule(_head())
    dummy = make_e24_joint_dummy_batch(topk=3, feature_dim=8, token_count=4)
    result, _ = forward_e24_joint_batch(head, dummy, loss_config=_loss_config())
    result.loss.backward()
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in head.parameters()
    )


def test_native_dummy_replay_covers_adapter_and_e24_parameters():
    module = E24JointTrainModule(
        _head(),
        cls_adapter=Top5NativeClsAdapter(
            feature_dim=8,
            condition_hidden_dim=4,
        ),
    )
    dummy = make_e24_joint_dummy_batch(
        topk=3,
        feature_dim=8,
        token_count=257,
        native_cls=True,
    )

    result, _ = forward_e24_joint_batch(
        module,
        dummy,
        loss_config=_loss_config(),
    )
    result.loss.backward()

    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in module.parameters()
    )


class _FixedNativeModule(torch.nn.Module):
    delta_max = 1.0

    def __init__(self):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor([0.5, -0.25]))

    def forward(self, owner, *_args):
        return self.delta.expand(owner.shape[0], -1)


def test_native_adjusted_ce_covers_stop_top5_outside_and_padding():
    module = _FixedNativeModule()
    full_base = torch.tensor(
        [
            [3.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 0.0],
            [0.0, 2.0, 1.0, 3.0],
            [0.0, 2.0, 1.0, -torch.inf],
        ],
        requires_grad=True,
    )
    batch = {
        "owner_embeddings": torch.zeros(4, 2, 8),
        "text_tokens": torch.zeros(4, 1, 8),
        "text_token_mask": torch.ones(4, 1, dtype=torch.bool),
        "future_tokens": torch.zeros(4, 2, 257, 8),
        "topk_valid_mask": torch.ones(4, 2, dtype=torch.bool),
        "candidate_q0_geometry": torch.zeros(4, 2, 3),
        "q1_conditions": torch.zeros(4, 2, 4),
        "full_base_logits": full_base,
        "full_valid_mask": torch.tensor(
            [
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, True],
                [True, True, True, False],
            ]
        ),
        "topk_global_indices": torch.tensor([[1, 2]] * 4),
        "teacher_actions": torch.tensor([0, 1, 3, -100]),
    }

    result = forward_native_adjusted_batch(module, batch)
    expected = full_base.detach().clone()
    expected[:, 1] += 0.5
    expected[:, 2] -= 0.25
    expected[3, 3] = -torch.inf

    assert result.row_count == 3
    torch.testing.assert_close(result.adjusted_logits, expected)
    torch.testing.assert_close(
        result.loss_sum,
        torch.nn.functional.cross_entropy(
            expected[:3], torch.tensor([0, 1, 3]), reduction="sum"
        ),
    )
    normalized = normalized_native_adjusted_loss(
        result,
        global_row_count=3,
        world_size=1,
        loss_weight=1.0,
    )
    normalized.backward()
    assert module.delta.grad is not None
    assert torch.count_nonzero(module.delta.grad) > 0
    assert full_base.grad is None


def test_native_adjusted_loss_only_updates_adapter_and_e24():
    module = E24JointTrainModule(
        _head(),
        cls_adapter=Top5NativeClsAdapter(
            feature_dim=8,
            condition_hidden_dim=4,
        ),
    )
    with torch.no_grad():
        module.head.score_mlp[-1].weight.normal_(mean=0.0, std=0.1)
    owner = torch.randn(2, 2, 8, requires_grad=True)
    text = torch.randn(2, 3, 8, requires_grad=True)
    future = torch.randn(2, 2, 257, 8, requires_grad=True)
    geometry = torch.randn(2, 2, 3, requires_grad=True)
    full_base = torch.randn(2, 4, requires_grad=True)
    batch = {
        "owner_embeddings": owner,
        "text_tokens": text,
        "text_token_mask": torch.ones(2, 3, dtype=torch.bool),
        "future_tokens": future,
        "topk_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "candidate_q0_geometry": geometry,
        "q1_conditions": torch.randn(2, 2, 4, requires_grad=True),
        "full_base_logits": full_base,
        "full_valid_mask": torch.ones(2, 4, dtype=torch.bool),
        "topk_global_indices": torch.tensor([[1, 2], [1, 2]]),
        "teacher_actions": torch.tensor([1, 3]),
    }

    result = forward_native_adjusted_batch(module, batch)
    normalized_native_adjusted_loss(
        result,
        global_row_count=2,
        world_size=1,
        loss_weight=1.0,
    ).backward()

    assert any(parameter.grad is not None for parameter in module.head.parameters())
    assert module.cls_adapter.fusion[-1].weight.grad is not None
    assert torch.count_nonzero(module.cls_adapter.fusion[-1].weight.grad) > 0
    for value in (owner, text, future, geometry, full_base, batch["q1_conditions"]):
        assert value.grad is None


def test_stop_isolation_preserves_base_stop_and_cannot_create_stop():
    logits = torch.tensor([[5.0, 4.0, 3.0], [0.0, 2.0, 1.0]])
    deltas = torch.tensor([[0.0, 100.0, 100.0], [100.0, -5.0, 5.0]])
    actions = stop_isolated_e24_actions(
        logits, deltas, [[None, "g1", "g2"], [None, "g1", "g2"]]
    )
    assert actions.tolist() == [0, 2]


def test_zero_delta_matches_base_and_non_top5_can_still_win_global_comparison():
    ids = [[None, "g1", "g2", "g3", "g4", "g5", "g6"]]
    logits = torch.tensor([[-10.0, 5.0, 4.9, 4.8, 4.7, 4.6, 4.55]])
    zeros = torch.zeros_like(logits)
    assert stop_isolated_e24_actions(logits, zeros, ids).tolist() == [1]

    top5_penalty = torch.tensor([[0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]])
    assert stop_isolated_e24_actions(logits, top5_penalty, ids).tolist() == [6]


def test_avg3_is_loaded_as_trainable_weights_only(tmp_path):
    frozen = _head().eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    checkpoint = tmp_path / "avg3.pth"
    torch.save(
        {
            "format_version": "stage0-topk-offline-checkpoint-v2",
            "global_step": 17250,
            "model_config": {
                "factory": (
                    "vlnce_baselines.nwm.active_lookahead.residual_head:"
                    "InterleavedCrossModalTopKFutureLogitResidualHead"
                ),
                "kwargs": {
                    "input_dim": 8,
                    "hidden_dim": 8,
                    "num_queries": 2,
                    "num_attention_heads": 2,
                    "ffn_dim": 16,
                    "dropout": 0.0,
                    "fusion_layers": 1,
                },
            },
            "dataset_provenance": {"base_manifest_sha256": "b" * 64},
            "posthoc_weight_average": {
                "resume_forbidden": True,
                "sources": [
                    {"global_step": step} for step in (17000, 17250, 17500)
                ],
            },
            "future_head_state_dict": frozen.state_dict(),
            "future_head_optimizer_state_dict": {"must_not_be_loaded": True},
        },
        checkpoint,
    )
    import hashlib

    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded, metadata = load_e24_joint_head(
        str(checkpoint),
        device=torch.device("cpu"),
        expected_checkpoint_sha256=checkpoint_sha,
        expected_source_base_manifest_sha256="b" * 64,
    )
    assert loaded.training is True
    assert all(parameter.requires_grad for parameter in loaded.parameters())
    assert metadata["weights_only_initialization"] is True
    assert metadata["resume_forbidden_ignored_for_weights_only"] is True


def _sparse_ddp_worker(rank, world_size, init_method):
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method
    )
    try:
        torch.manual_seed(91)
        module = DDP(E24JointTrainModule(_head()), find_unused_parameters=True)
        reference = deepcopy(module.module) if rank == 0 else None
        full_batch = _batch()
        local_batch = (
            slice_e24_joint_batch(full_batch, 0, 2) if rank == 0 else None
        )
        config = _loss_config()
        local_denominators = (
            e24_joint_denominators(local_batch, config)
            if local_batch is not None
            else {name: 0.0 for name in (
                "signed", "decision_weight", "regularization", "absent_noop"
            )}
        )
        names = tuple(local_denominators)
        values = torch.tensor([local_denominators[name] for name in names])
        dist.all_reduce(values)
        denominators = dict(zip(names, values.tolist()))
        dummy = make_e24_joint_dummy_batch(topk=3, feature_dim=8, token_count=4)

        for replay_index in range(2):
            micro = (
                slice_e24_joint_batch(local_batch, replay_index, replay_index + 1)
                if local_batch is not None
                else dummy
            )
            context = module.no_sync() if replay_index == 0 else nullcontext()
            with context:
                result, _ = forward_e24_joint_batch(
                    module, micro, loss_config=config
                )
                loss = normalized_e24_joint_loss(
                    result,
                    global_denominators=denominators,
                    config=config,
                    world_size=world_size,
                    loss_weight=1.0,
                )
                loss.backward()

        failed = torch.zeros((), dtype=torch.int32)
        if rank == 0:
            result, _ = forward_e24_joint_batch(
                reference, local_batch, loss_config=config
            )
            normalized_e24_joint_loss(
                result,
                global_denominators=denominators,
                config=config,
                world_size=1,
                loss_weight=1.0,
            ).backward()
            for distributed, expected in zip(
                module.module.parameters(), reference.parameters()
            ):
                if distributed.grad is None or not torch.allclose(
                    distributed.grad, expected.grad, rtol=3e-5, atol=3e-6
                ):
                    failed.fill_(1)
                    break
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        assert failed.item() == 0

        for parameter in module.parameters():
            reference_parameter = parameter.detach().clone()
            dist.broadcast(reference_parameter, src=0)
            assert torch.equal(parameter.detach(), reference_parameter)
    finally:
        dist.destroy_process_group()


def test_sparse_two_rank_replay_has_exact_global_gradient_and_no_hang(tmp_path):
    init_method = f"file://{tmp_path / 'e24-ddp-init'}"
    mp.spawn(_sparse_ddp_worker, args=(2, init_method), nprocs=2, join=True)
