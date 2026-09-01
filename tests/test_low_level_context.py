from types import SimpleNamespace

import numpy as np
import pytest
import torch

from habitat_extensions.habitat_simulator import Simulator
from vlnce_baselines.nwm.etp_adapter import NwmEtpAdapter, RaeEtpAdapterConfig
from vlnce_baselines.nwm.low_level_context import (
    LOW_LEVEL_CONTEXT_DIAGNOSTIC_FORMAT,
    LOW_LEVEL_CONTEXT_CONTRACT,
    LOW_LEVEL_CONTEXT_SOURCE,
    LOW_LEVEL_EVENT_PAYLOAD_FORMAT,
    LowLevelContextEventBuffer,
    LowLevelContextSynchronizer,
    summarize_low_level_context_diagnostics,
)
from vlnce_baselines.nwm.runtime import NwmPredictionRuntime
from vlnce_baselines.nwm.types import NwmPrediction
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer as SftTrainer
from vlnce_baselines.nwm.frozen_grpo import FrozenLookaheadController
from vlnce_baselines import ss_trainer_ETP_R1 as sft_module
from vlnce_baselines.nwm import frozen_grpo as frozen_grpo_module


def _frame_event(value, position, yaw=0.0):
    return {
        "type": "frame",
        "rgb": np.full((4, 4, 3), value, dtype=np.uint8),
        "position": np.asarray(position, dtype=np.float32),
        "yaw": float(yaw),
    }


def _payload(events, dropped=0):
    return {
        "format": LOW_LEVEL_EVENT_PAYLOAD_FORMAT,
        "contract": LOW_LEVEL_CONTEXT_CONTRACT,
        "events": events,
        "stats": {
            "reset_events": sum(event["type"] == "reset" for event in events),
            "frame_events": sum(event["type"] == "frame" for event in events),
            "dropped_static_frames": dropped,
        },
    }


def test_event_buffer_records_anchors_and_moves_but_drops_collision_frames():
    buffer = LowLevelContextEventBuffer(pos_eps=1.0e-3, yaw_eps=1.0e-3)
    rgb = lambda value: {"rgb": np.full((4, 4, 3), value, dtype=np.uint8)}

    buffer.reset_trace()
    assert buffer.append_observation(rgb(0), [0.0, 0.0, 0.0], 0.0)
    assert buffer.append_observation(
        rgb(1), [0.25, 0.0, 0.0], 0.0, movement_frame=True
    )
    # The yaw changed after turns, but MOVE_FORWARD did not translate: collision.
    assert not buffer.append_observation(
        rgb(2), [0.25, 0.0, 0.0], 1.0, movement_frame=True
    )
    # Teleport starts a new trace and records its already-rendered landing anchor.
    buffer.reset_trace()
    assert buffer.append_observation(rgb(3), [3.0, 0.0, 1.0], 0.0)

    payload = buffer.pop_payload()

    assert payload["contract"] == LOW_LEVEL_CONTEXT_CONTRACT
    assert [event["type"] for event in payload["events"]] == [
        "reset",
        "frame",
        "frame",
        "reset",
        "frame",
    ]
    assert payload["stats"]["dropped_static_frames"] == 1
    np.testing.assert_allclose(
        payload["events"][-1]["position"], [3.0, 0.0, 1.0]
    )


class _FakeEncoder:
    def __init__(self):
        self.batch_sizes = []

    def forward_with_raw_cls_and_patch_latents(self, observations):
        rgb = torch.as_tensor(observations["rgb"])
        self.batch_sizes.append(int(rgb.shape[0]))
        values = rgb[:, 0, 0, 0].float()
        raw_cls = values[:, None].expand(-1, 768).contiguous()
        patch = values[:, None, None, None].expand(
            -1, 768, 16, 16
        ).contiguous()
        return raw_cls, raw_cls, patch


class _IdentityNormalizer:
    @staticmethod
    def normalize_patch(value):
        return value

    @staticmethod
    def normalize_cls(value):
        return value


class _FakeVectorEnv:
    def __init__(self, payloads):
        self.payloads = payloads
        self.num_envs = len(payloads)

    def call(self, names):
        assert names == ["pop_raenwm_context_events"] * self.num_envs
        return self.payloads


def _runtime(num_envs):
    runtime = object.__new__(NwmPredictionRuntime)
    runtime.context_source = LOW_LEVEL_CONTEXT_SOURCE
    runtime.context_metadata = {
        "context_source": LOW_LEVEL_CONTEXT_SOURCE,
        "context_contract": LOW_LEVEL_CONTEXT_CONTRACT,
        "sampling_action": "MOVE_FORWARD",
        "teleport_anchor_policy": "clear_then_record_landing_rgb",
        "encode_batch_size": 64,
    }
    runtime.normalizer = _IdentityNormalizer()
    runtime.adapter = NwmEtpAdapter(
        RaeEtpAdapterConfig(
            context_size=4,
            max_buffer_size=8,
            static_run_k=3,
        )
    )
    runtime.adapter.reset(num_envs)
    return runtime


def _drain(payloads, batch_size):
    runtime = _runtime(len(payloads))
    encoder = _FakeEncoder()
    synchronizer = LowLevelContextSynchronizer(
        runtime=runtime,
        encoder=encoder,
        device="cpu",
        batch_size=batch_size,
    )
    diagnostics = synchronizer.drain(_FakeVectorEnv(payloads))
    return runtime, encoder, diagnostics


def test_low_level_batch_encoding_builds_native_four_by_257_context():
    payload = _payload(
        [{"type": "reset"}]
        + [_frame_event(index + 1, [index * 0.25, 0.0, 0.0]) for index in range(4)]
    )

    runtime, encoder, diagnostics = _drain([payload], batch_size=2)
    context = torch.stack(
        [frame.latent for frame in runtime.adapter.buffers[0].get_context()]
    )

    assert context.shape == (4, 257, 768)
    assert encoder.batch_sizes == [2, 2]
    assert diagnostics["context_ready_ratio"] == 1.0
    assert context[:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert context[:, 1, 0].tolist() == [1.0, 2.0, 3.0, 4.0]
    snapshot = runtime.source_context_snapshot(
        0,
        source_front_vp="front",
        source_high_level_step=2,
    )
    assert snapshot.context_contract == LOW_LEVEL_CONTEXT_CONTRACT
    assert snapshot.sampling_action == "MOVE_FORWARD"
    assert snapshot.teleport_anchor_policy == "clear_then_record_landing_rgb"
    assert snapshot.encode_batch_size == 64


def test_low_level_eval_diagnostics_summarize_counts_readiness_and_timing():
    summary = summarize_low_level_context_diagnostics(
        {
            "drain_count": 4,
            "environment_count": 10,
            "reset_events": 3,
            "frame_events": 20,
            "encoded_frames": 20,
            "encode_batches": 5,
            "encode_seconds": 2.0,
            "context_ready": 7,
            "context_ready_ratio": 2.5,
            "dropped_static_frames": 2,
        }
    )

    assert summary["reset_events"] == 3.0
    assert summary["frame_events"] == 20.0
    assert summary["context_ready_ratio"] == pytest.approx(0.7)
    assert summary["mean_drain_context_ready_ratio"] == pytest.approx(0.625)
    assert summary["frames_per_drain"] == 5.0
    assert summary["encode_seconds_per_frame"] == pytest.approx(0.1)


def test_eval_lookahead_payload_embeds_low_level_context_contract_and_metrics():
    trainer = object.__new__(SftTrainer)
    trainer.config = SimpleNamespace(
        MODEL=SimpleNamespace(
            RAENWM=SimpleNamespace(
                context_source=LOW_LEVEL_CONTEXT_SOURCE,
                context_size=4,
                low_level_encode_batch_size=64,
            )
        )
    )
    context_metrics = {"reset_events": 1.0, "frame_events": 6.0}

    payload = trainer._build_eval_lookahead_diagnostic_payload(
        checkpoint_path="ckpt.iter2.pth",
        checkpoint_index=2,
        split="val_unseen",
        episodes=1,
        elapsed_seconds=3.5,
        lookahead_diagnostics={"oracle_q1_requested": 0.0},
        low_level_context_diagnostics=context_metrics,
    )

    context = payload["low_level_context"]
    assert context["format_version"] == LOW_LEVEL_CONTEXT_DIAGNOSTIC_FORMAT
    assert context["metadata"]["context_contract"] == LOW_LEVEL_CONTEXT_CONTRACT
    assert context["metadata"]["encode_batch_size"] == 64
    assert context["metrics"] == context_metrics


def test_chunked_encoding_matches_whole_batch_and_distributes_by_environment():
    payloads = [
        _payload(
            [
                {"type": "reset"},
                _frame_event(1, [0.0, 0.0, 0.0]),
                _frame_event(2, [0.25, 0.0, 0.0]),
            ]
        ),
        _payload(
            [
                {"type": "reset"},
                _frame_event(3, [1.0, 0.0, 0.0]),
                _frame_event(4, [1.25, 0.0, 0.0]),
            ]
        ),
    ]
    chunked, chunked_encoder, _ = _drain(payloads, batch_size=2)
    whole, whole_encoder, _ = _drain(payloads, batch_size=64)

    assert chunked_encoder.batch_sizes == [2, 2]
    assert whole_encoder.batch_sizes == [4]
    for chunked_buffer, whole_buffer in zip(
        chunked.adapter.buffers, whole.adapter.buffers
    ):
        chunked_tokens = torch.stack([frame.latent for frame in chunked_buffer.frames])
        whole_tokens = torch.stack([frame.latent for frame in whole_buffer.frames])
        torch.testing.assert_close(chunked_tokens, whole_tokens)
    assert [frame.latent[0, 0].item() for frame in chunked.adapter.buffers[0].frames] == [
        1.0,
        2.0,
    ]
    assert [frame.latent[0, 0].item() for frame in chunked.adapter.buffers[1].frames] == [
        3.0,
        4.0,
    ]


def test_reset_and_frame_order_clears_pre_teleport_context_in_same_drain():
    payload = _payload(
        [
            {"type": "reset"},
            _frame_event(1, [0.0, 0.0, 0.0]),
            _frame_event(2, [0.25, 0.0, 0.0]),
            {"type": "reset"},
            _frame_event(9, [9.0, 0.0, 0.0]),
        ]
    )

    runtime, _encoder, diagnostics = _drain([payload], batch_size=64)

    frames = runtime.adapter.buffers[0].frames
    assert len(frames) == 1
    assert frames[0].latent[0, 0].item() == 9.0
    np.testing.assert_allclose(frames[0].position, [9.0, 0.0, 0.0])
    assert diagnostics["reset_events"] == 2.0
    assert diagnostics["context_ready_ratio"] == 0.0


def test_low_level_payload_contract_is_strict():
    payload = _payload([{"type": "reset"}])
    payload["contract"] = "wrong"

    with pytest.raises(ValueError, match="wrong low-level context contract"):
        _drain([payload], batch_size=64)


class _RecordingPredictionRuntime:
    def __init__(self):
        self.update_calls = 0

    def update_contexts(self, *args, **kwargs):
        self.update_calls += 1

    @staticmethod
    def predict(queries):
        return NwmPrediction(
            pred_latent=None,
            meta={"empty": True, "records": [], "skipped": {}},
        )


def _trainer_config(context_source):
    return SimpleNamespace(
        MODEL=SimpleNamespace(
            RAENWM=SimpleNamespace(
                enabled=True,
                context_source=context_source,
            )
        )
    )


def test_low_level_checkpoint_metadata_rejects_old_context_weights():
    trainer = object.__new__(SftTrainer)
    trainer.config = SimpleNamespace(
        MODEL=SimpleNamespace(
            RAENWM=SimpleNamespace(
                enabled=True,
                context_source=LOW_LEVEL_CONTEXT_SOURCE,
                context_size=4,
                low_level_encode_batch_size=64,
            )
        )
    )
    expected = trainer._raenwm_context_metadata()
    assert trainer._validate_raenwm_context_checkpoint_metadata(
        {"raenwm_context_metadata": expected},
        allow_missing=False,
    ) == expected
    with pytest.raises(ValueError, match="refuses context-dependent weights"):
        trainer._validate_raenwm_context_checkpoint_metadata(
            {"raenwm_rgb_fusion_adapter_state_dict": {"weight": torch.ones(1)}},
            allow_missing=True,
        )


def test_sft_low_level_mode_never_writes_high_level_panorama(monkeypatch):
    trainer = object.__new__(SftTrainer)
    trainer.config = _trainer_config(LOW_LEVEL_CONTEXT_SOURCE)
    trainer.raenwm_runtime = _RecordingPredictionRuntime()
    trainer.last_raenwm_prediction = None
    trainer._raenwm_context_source_logged = False
    monkeypatch.setattr(sft_module, "heading_from_quaternion", float)

    trainer._run_raenwm_prediction(
        torch.zeros(1, 768, 16, 16),
        [np.zeros(3, dtype=np.float32)],
        [0.0],
        [["g0"]],
        [[np.asarray([0.0, 0.0, -1.0], dtype=np.float32)]],
        front_cls=torch.zeros(1, 768),
    )

    assert trainer.raenwm_runtime.update_calls == 0


def test_sft_high_level_compatibility_still_updates_panorama(monkeypatch):
    trainer = object.__new__(SftTrainer)
    trainer.config = _trainer_config("high_level_nav_latent")
    trainer.raenwm_runtime = _RecordingPredictionRuntime()
    trainer.last_raenwm_prediction = None
    trainer._raenwm_context_source_logged = False
    monkeypatch.setattr(sft_module, "heading_from_quaternion", float)

    trainer._run_raenwm_prediction(
        torch.zeros(1, 768, 16, 16),
        [np.zeros(3, dtype=np.float32)],
        [0.0],
        [["g0"]],
        [[np.asarray([0.0, 0.0, -1.0], dtype=np.float32)]],
        front_cls=torch.zeros(1, 768),
    )

    assert trainer.raenwm_runtime.update_calls == 1


def test_frozen_grpo_low_level_mode_never_writes_high_level_panorama(monkeypatch):
    controller = object.__new__(FrozenLookaheadController)
    controller.config = _trainer_config(LOW_LEVEL_CONTEXT_SOURCE)
    controller.raenwm_runtime = _RecordingPredictionRuntime()
    controller.last_rgb_diagnostics = None
    controller.last_candidate_q0_prediction_diagnostics = None
    controller.rgb_fusion_adapter = object()
    controller._build_preview_queries = lambda *args: []
    monkeypatch.setattr(
        frozen_grpo_module,
        "heading_from_quaternion",
        float,
    )
    monkeypatch.setattr(
        frozen_grpo_module,
        "apply_rgb_fusion_to_current_candidates",
        lambda *args: [],
    )

    controller.inject_rgb(
        torch.zeros(1, 768, 16, 16),
        torch.zeros(1, 768),
        [np.zeros(3, dtype=np.float32)],
        [0.0],
        [[]],
        {},
    )

    assert controller.raenwm_runtime.update_calls == 0


def test_single_sensor_render_does_not_draw_depth_or_panorama_sensors():
    class Sensor:
        def __init__(self, value):
            self.value = value
            self.draws = 0

        def draw_observation(self):
            self.draws += 1

        def get_observation(self):
            return self.value

    class SensorAdapter:
        @staticmethod
        def get_observation(observations):
            return observations["rgb"]

    rgb = Sensor(np.ones((2, 2, 4), dtype=np.uint8))
    depth = Sensor(np.zeros((2, 2, 1), dtype=np.float32))
    fake = SimpleNamespace(
        config=SimpleNamespace(enable_batch_renderer=False),
        _sensors={"rgb": rgb, "depth": depth},
        _sensor_suite=SimpleNamespace(get=lambda name: SensorAdapter()),
        _prev_sim_obs=None,
        get_agent_state=lambda: SimpleNamespace(position=[0, 0, 0], rotation=[0, 0, 0, 1]),
        set_agent_state=lambda *args, **kwargs: True,
    )

    observation = Simulator.get_sensor_observation_at(
        fake,
        sensor_uuid="rgb",
    )

    assert observation.shape == (2, 2, 4)
    assert rgb.draws == 1
    assert depth.draws == 0
