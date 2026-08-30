from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlnce_baselines.nwm.active_lookahead.dino_cwp_future import (
    SingleViewDinoCwpPredictor,
    build_dino_cwp_nwm_future_tokens,
    decode_dino_cwp_top1,
    latent_to_patch_tokens,
    load_dino_cwp_predictor,
    summarize_predicted_future_diagnostics,
    waypoint_to_world_position,
)
from vlnce_baselines.nwm.etp_adapter import (
    NwmEtpAdapter,
    RaeEtpAdapterConfig,
    RaeSourceContextSnapshot,
)


def _snapshot(marker, front, step, *, native=False):
    shape = (4, 257, 768) if native else (4, 768, 16, 16)
    context = torch.full(shape, float(marker))
    return RaeSourceContextSnapshot(
        source_front_vp=front,
        source_high_level_step=step,
        context_latents=context,
        source_position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        source_yaw=float(np.pi),
    )


def _record(ghost, front, step, x, *, marker=None, native=False):
    marker = float(step if step else 1) if marker is None else float(marker)
    snapshot = _snapshot(marker, front, step, native=native)
    return SimpleNamespace(
        contract_version="r1_post_update_ghost_mean_cached_v1",
        ghost_vp=ghost,
        source_front_vp=front,
        source_high_level_step=step,
        canonical_q0_position=np.asarray([x, 0.0, 1.0], dtype=np.float32),
        source_context=snapshot,
        predicted_patch_cpu_fp16=torch.full(
            (768, 16, 16), marker, dtype=torch.float16
        ),
    )


class _FakeNwm:
    def __init__(self, *, native=False):
        self.context_markers = []
        self.native = native

    def predict_time_with_heads_from_etp_batch(self, batch, return_rgb=False):
        del return_rgb
        markers = (
            batch.context_latent[:, 0]
            .reshape(batch.context_latent.shape[0], -1)[:, 0]
            .detach()
            .cpu()
            .tolist()
        )
        self.context_markers.append(markers)
        values = torch.as_tensor(markers, device=batch.context_latent.device).float()
        latent = values[:, None, None, None].expand(-1, 768, 16, 16).clone()
        cls = (values + 100.0)[:, None].expand(-1, 768).clone()
        pred_tokens = None
        if self.native:
            patch_tokens = latent.permute(0, 2, 3, 1).reshape(-1, 256, 768)
            normalized_cls = (values + 999.0)[:, None].expand(-1, 768)
            pred_tokens = torch.cat(
                (normalized_cls[:, None], patch_tokens), dim=1
            )
        return SimpleNamespace(
            pred_latent=latent,
            pred_cls=cls,
            pred_tokens=pred_tokens,
        )


class _FakeRuntime:
    def __init__(self, adapter, *, native=False):
        self.adapter = adapter
        self.predictor = _FakeNwm(native=native)
        self.predict_cls_token = native
        self.requests = []

    def predict_latent_targets(self, requests):
        self.requests.append(list(requests))
        batch = self.adapter.build_raenwm_latent_batch(requests, device="cpu")
        return self.predictor.predict_time_with_heads_from_etp_batch(batch)


class _FakeCwp:
    def __init__(self, none_rows=()):
        self.none_rows = set(none_rows)

    def __call__(self, patch_tokens):
        batch = int(patch_tokens.shape[0])
        heatmap = torch.zeros(batch, 30, 12, device=patch_tokens.device)
        heatmap[:, 15, 3] = 10.0
        none = torch.full((batch,), -10.0, device=patch_tokens.device)
        for row in self.none_rows:
            none[row] = 10.0
        return {"heatmap_logits": heatmap, "none_logit": none}


def _trainer(snapshots, *, none_rows=(), native=False):
    adapter = NwmEtpAdapter(RaeEtpAdapterConfig())
    adapter.reset(1)
    runtime = _FakeRuntime(adapter, native=native)
    return SimpleNamespace(
        device=torch.device("cpu"),
        raenwm_runtime=runtime,
        dino_cwp_future_predictor=_FakeCwp(none_rows),
        gmaps=[SimpleNamespace()],
        _active_lookahead_config=lambda: SimpleNamespace(dino_cwp_none_threshold=0.3),
    )


def test_predicted_future_batches_distinct_historical_contexts_and_reuses_them_for_q1():
    snapshots = {
        ("front0", 0): _snapshot(1.0, "front0", 0),
        ("front4", 4): _snapshot(4.0, "front4", 4),
    }
    trainer = _trainer(snapshots)
    records = [[
        _record("g0", "front0", 0, 0.0),
        _record("g1", "front4", 4, 1.0),
    ]]

    future, q1_conditions, valid, diagnostics = build_dino_cwp_nwm_future_tokens(
        trainer,
        active_envs=[0],
        records_by_env=records,
        topk=2,
        feature_dim=768,
        reference=torch.zeros(1),
    )

    assert valid.tolist() == [[True, True]]
    assert tuple(future.shape) == (1, 2, 257, 768)
    assert tuple(q1_conditions.shape) == (1, 2, 4)
    assert trainer.raenwm_runtime.predictor.context_markers == [[1.0, 4.0]]
    assert future[0, 0, 0, 0].item() == pytest.approx(101.0)
    assert future[0, 1, 0, 0].item() == pytest.approx(104.0)
    assert future[0, 0, 1, 0].item() == pytest.approx(1.0)
    assert diagnostics["q0_context_present"] == 2.0
    assert diagnostics["q1_nwm_success"] == 2.0
    assert diagnostics["future_valid"] == 2.0
    assert len(trainer.raenwm_runtime.requests) == 1
    q1_requests = trainer.raenwm_runtime.requests[0]
    assert all(request.horizon_override is not None for request in q1_requests)
    assert all(request.target_yaw is not None for request in q1_requests)
    assert diagnostics["q0_requested"] == 0.0
    assert diagnostics["q0_cache_present"] == 2.0


def test_missing_context_and_cwp_none_invalidate_only_their_slots():
    snapshots = {("front0", 0): _snapshot(2.0, "front0", 0)}
    trainer = _trainer(snapshots, none_rows=(0,))
    missing = _record("g1", "missing", 1, 1.0)
    missing.source_context = None
    records = [[_record("g0", "front0", 0, 0.0), missing]]

    future, q1_conditions, valid, diagnostics = build_dino_cwp_nwm_future_tokens(
        trainer,
        active_envs=[0],
        records_by_env=records,
        topk=2,
        feature_dim=768,
        reference=torch.zeros(1),
    )

    assert valid.tolist() == [[False, False]]
    assert torch.count_nonzero(future) == 0
    assert torch.count_nonzero(q1_conditions) == 0
    assert diagnostics["q0_record_present"] == 2.0
    assert diagnostics["q0_context_present"] == 1.0
    assert diagnostics["cwp_none"] == 1.0
    assert diagnostics["q1_requested"] == 0.0


def test_native_q1_uses_raw_cls_and_preserves_native_patch_tokens():
    snapshots = {
        ("front0", 0): _snapshot(3.0, "front0", 0, native=True),
    }
    trainer = _trainer(snapshots, native=True)

    future, q1_conditions, valid, diagnostics = (
        build_dino_cwp_nwm_future_tokens(
            trainer,
            active_envs=[0],
            records_by_env=[[
                _record("g0", "front0", 0, 0.0, marker=3.0, native=True)
            ]],
            topk=1,
            feature_dim=768,
            reference=torch.zeros(1),
        )
    )

    assert valid.tolist() == [[True]]
    assert future[0, 0, 0, 0].item() == pytest.approx(103.0)
    assert future[0, 0, 1, 0].item() == pytest.approx(3.0)
    assert q1_conditions.shape == (1, 1, 4)
    assert torch.isfinite(q1_conditions).all()
    assert diagnostics["future_valid"] == 1.0


def test_cwp_top1_decode_and_geometry_match_fixed_contract():
    heatmap = torch.zeros(1, 30, 12)
    heatmap[0, 15, 3] = 9.0
    prediction = decode_dino_cwp_top1(
        {"heatmap_logits": heatmap, "none_logit": torch.tensor([-5.0])},
        none_threshold=0.3,
    )[0]
    assert prediction.pred_none is False
    assert prediction.local_angle_deg == pytest.approx(1.5)
    assert prediction.distance_m == pytest.approx(1.0)
    assert waypoint_to_world_position(
        [0.0, 0.0, 0.0],
        heading_deg=0.0,
        local_angle_deg=0.0,
        distance_m=1.0,
    ).tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_small_cwp_model_and_latent_shape_contracts():
    model = SingleViewDinoCwpPredictor(
        hidden_dim=64, num_heads=4, encoder_layers=1, ffn_dim=128
    ).eval()
    output = model(torch.randn(2, 256, 768))
    assert tuple(output["heatmap_logits"].shape) == (2, 30, 12)
    assert tuple(output["none_logit"].shape) == (2,)
    assert tuple(latent_to_patch_tokens(torch.randn(2, 768, 16, 16)).shape) == (
        2, 256, 768
    )


def test_checkpoint_sha_is_verified_before_checkpoint_deserialization(tmp_path):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"not-a-checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_dino_cwp_predictor(
            checkpoint,
            expected_sha256="0" * 64,
            device=torch.device("cpu"),
        )


def test_fixed_initial_cumulative_horizon_is_clipped_and_explicit_yaw_is_used():
    adapter = NwmEtpAdapter(RaeEtpAdapterConfig())
    record = adapter.build_target_record(
        env_index=0,
        ghost_vp="g0",
        source_position=[0.0, 0.0, 0.0],
        source_yaw=np.pi,
        target_position=[1.0, 0.0, 1.0],
        target_yaw=np.pi / 2.0,
        horizon_override=100.0,
    )
    assert record.horizon == pytest.approx(64.0)
    assert record.condition.rel_t == pytest.approx(0.5)
    assert record.condition.dtheta == pytest.approx(-np.pi / 2.0)


def test_source_context_snapshot_copies_four_latents_to_cpu():
    adapter = NwmEtpAdapter(RaeEtpAdapterConfig(static_run_k=10))
    adapter.reset(1)
    for index in range(4):
        adapter.update_context(
            0,
            rgb=None,
            position=[float(index), 0.0, 0.0],
            yaw=0.1 * index,
            latent=torch.full((768, 16, 16), float(index)),
        )
    snapshot = adapter.source_context_snapshot(
        0, source_front_vp="front3", source_high_level_step=3
    )
    assert snapshot is not None
    assert snapshot.context_latents.device.type == "cpu"
    assert snapshot.context_latents.dtype == torch.float32
    assert tuple(snapshot.context_latents.shape) == (4, 768, 16, 16)
    assert snapshot.context_latents[:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert snapshot.source_position.tolist() == pytest.approx([3.0, 0.0, 0.0])


def test_predicted_future_diagnostic_summary_preserves_counts_and_rates():
    summary = summarize_predicted_future_diagnostics({
        "topk_slots": 10.0,
        "q0_record_present": 10.0,
        "q0_context_present": 6.0,
        "q0_cache_present": 6.0,
        "q0_requested": 6.0,
        "q0_nwm_success": 6.0,
        "cwp_requested": 6.0,
        "cwp_none": 1.0,
        "cwp_top1": 5.0,
        "q1_requested": 5.0,
        "q1_nwm_success": 5.0,
        "future_valid": 5.0,
        "q0_latent_rows": 2.0,
        "q0_latent_norm_sum": 8.0,
        "q1_latent_rows": 5.0,
        "q1_latent_norm_sum": 15.0,
        "action_rows": 4.0,
        "action_flips": 1.0,
    })

    assert summary["topk_slots"] == 10.0
    assert summary["oracle_q1_requested"] == 0.0
    assert summary["q0_context_coverage"] == pytest.approx(0.6)
    assert summary["q0_success_rate"] == pytest.approx(1.0)
    assert summary["cwp_none_rate"] == pytest.approx(1.0 / 6.0)
    assert summary["valid_rate"] == pytest.approx(0.5)
    assert summary["action_flip_rate"] == pytest.approx(0.25)
    assert summary["q0_latent_norm"] == pytest.approx(4.0)
    assert summary["q1_latent_norm"] == pytest.approx(3.0)
