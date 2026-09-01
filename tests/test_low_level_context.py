from types import SimpleNamespace

import numpy as np
import pytest
import torch

from habitat_extensions.habitat_simulator import Simulator
from vlnce_baselines.common import environments as environments_module
from vlnce_baselines.common.environments import VLNCEDaggerEnv
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
    frame_count = sum(event["type"] == "frame" for event in events)
    return {
        "format": LOW_LEVEL_EVENT_PAYLOAD_FORMAT,
        "contract": LOW_LEVEL_CONTEXT_CONTRACT,
        "events": events,
        "stats": {
            "reset_events": sum(event["type"] == "reset" for event in events),
            "frame_events": frame_count,
            "dropped_static_frames": dropped,
            "valid_poses": frame_count,
            "trimmed_poses": 0,
            "single_sensor_renders": 0,
            "reused_rgb_frames": frame_count,
            "replay_render_seconds": 0.0,
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
    assert buffer.materialize_pending(render_rgb=None) == 2
    # Teleport starts a new trace and records its already-rendered landing anchor.
    buffer.reset_trace()
    assert buffer.append_observation(rgb(3), [3.0, 0.0, 1.0], 0.0)
    assert buffer.materialize_pending(render_rgb=None) == 1

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
    assert payload["stats"]["single_sensor_renders"] == 0
    assert payload["stats"]["reused_rgb_frames"] == 3
    np.testing.assert_allclose(
        payload["events"][-1]["position"], [3.0, 0.0, 1.0]
    )


def _yaw_rotation(yaw):
    return np.asarray(
        [0.0, np.sin(0.5 * yaw), 0.0, np.cos(0.5 * yaw)],
        dtype=np.float64,
    )


def test_delayed_buffer_keeps_last_four_poses_and_reuses_final_observation():
    buffer = LowLevelContextEventBuffer(context_size=4)
    buffer.reset_trace()
    positions = []
    previous = np.zeros(3, dtype=np.float32)
    for index in range(6):
        position = np.asarray([(index + 1) * 0.25, 0.0, 0.0], dtype=np.float32)
        positions.append(position)
        assert buffer.append_pose(
            position,
            _yaw_rotation(0.0),
            0.0,
            movement_frame=True,
            previous_position=previous,
        )
        previous = position

    rendered_positions = []

    def render_rgb(position, _rotation):
        rendered_positions.append(position.copy())
        return np.full((4, 4, 3), round(float(position[0]) * 100), np.uint8)

    final_rgb = np.full((4, 4, 3), 99, dtype=np.uint8)
    assert buffer.materialize_pending(
        render_rgb=render_rgb,
        final_observations={"rgb": final_rgb},
        final_position=positions[-1],
        final_rotation=_yaw_rotation(0.0),
    ) == 4
    payload = buffer.pop_payload()
    frames = [event for event in payload["events"] if event["type"] == "frame"]

    assert len(frames) == 4
    np.testing.assert_allclose(
        np.stack([event["position"] for event in frames]),
        np.stack(positions[-4:]),
    )
    assert len(rendered_positions) == 3
    np.testing.assert_array_equal(frames[-1]["rgb"], final_rgb)
    assert payload["stats"]["valid_poses"] == 6
    assert payload["stats"]["trimmed_poses"] == 2
    assert payload["stats"]["single_sensor_renders"] == 3
    assert payload["stats"]["reused_rgb_frames"] == 1
    assert (
        payload["stats"]["single_sensor_renders"]
        + payload["stats"]["reused_rgb_frames"]
        == payload["stats"]["frame_events"]
    )


def test_collision_is_dropped_before_rendering():
    buffer = LowLevelContextEventBuffer(context_size=4)
    buffer.reset_trace()
    position = np.asarray([1.0, 0.0, 2.0], dtype=np.float32)
    assert not buffer.append_pose(
        position,
        _yaw_rotation(1.0),
        1.0,
        movement_frame=True,
        previous_position=position,
    )

    render_calls = []
    assert buffer.materialize_pending(
        render_rgb=lambda *_args: render_calls.append(True)
    ) == 0
    payload = buffer.pop_payload()

    assert render_calls == []
    assert payload["stats"]["dropped_static_frames"] == 1
    assert payload["stats"]["frame_events"] == 0
    assert payload["stats"]["single_sensor_renders"] == 0


def test_video_rgb_frames_are_trimmed_and_reused_without_replay_rendering():
    buffer = LowLevelContextEventBuffer(context_size=4)
    buffer.reset_trace()
    for index in range(7):
        position = [index * 0.25, 0.0, 0.0]
        buffer.append_pose(
            position,
            _yaw_rotation(0.0),
            0.0,
            observations={
                "rgb": np.full((4, 4, 3), index, dtype=np.uint8)
            },
        )

    assert buffer.materialize_pending(render_rgb=None) == 4
    payload = buffer.pop_payload()
    frames = [event for event in payload["events"] if event["type"] == "frame"]

    assert [int(event["rgb"][0, 0, 0]) for event in frames] == [3, 4, 5, 6]
    assert payload["stats"]["trimmed_poses"] == 3
    assert payload["stats"]["single_sensor_renders"] == 0
    assert payload["stats"]["reused_rgb_frames"] == 4


def test_delayed_render_matches_immediate_reference_for_final_four_frames():
    def deterministic_rgb(position, rotation):
        value = int(round(float(position[0]) * 40 + float(rotation[1]) * 10))
        return np.full((4, 4, 3), value, dtype=np.uint8)

    poses = [
        (
            np.asarray([index * 0.25, 0.0, -index * 0.1], dtype=np.float32),
            _yaw_rotation(index * 0.1),
            index * 0.1,
        )
        for index in range(7)
    ]
    immediate = [
        deterministic_rgb(position, rotation)
        for position, rotation, _yaw in poses
    ][-4:]
    buffer = LowLevelContextEventBuffer(context_size=4)
    for position, rotation, yaw in poses:
        buffer.append_pose(position, rotation, yaw)
    buffer.materialize_pending(render_rgb=deterministic_rgb)
    frames = [
        event["rgb"]
        for event in buffer.pop_payload()["events"]
        if event["type"] == "frame"
    ]

    assert len(frames) == len(immediate) == 4
    for actual, expected in zip(frames, immediate):
        np.testing.assert_array_equal(actual, expected)


def test_reset_discards_unmaterialized_pre_teleport_poses():
    buffer = LowLevelContextEventBuffer(context_size=4)
    buffer.append_pose([0.25, 0.0, 0.0], _yaw_rotation(0.0), 0.0)
    buffer.append_pose([0.5, 0.0, 0.0], _yaw_rotation(0.0), 0.0)

    buffer.reset_trace()
    buffer.append_pose([5.0, 0.0, 1.0], _yaw_rotation(0.0), 0.0)
    landing_rgb = np.full((4, 4, 3), 17, dtype=np.uint8)
    buffer.materialize_pending(
        render_rgb=lambda *_args: pytest.fail("final RGB should be reused"),
        final_observations={"rgb": landing_rgb},
        final_position=[5.0, 0.0, 1.0],
        final_rotation=_yaw_rotation(0.0),
    )
    payload = buffer.pop_payload()
    frames = [event for event in payload["events"] if event["type"] == "frame"]

    assert [event["type"] for event in payload["events"]] == ["reset", "frame"]
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0]["rgb"], landing_rgb)
    np.testing.assert_allclose(frames[0]["position"], [5.0, 0.0, 1.0])


def test_wrap_act_records_pose_without_render_and_drops_collision(monkeypatch):
    class Rotation:
        imag = np.zeros(3, dtype=np.float64)
        real = 1.0

    class FakeSimulator:
        def __init__(self):
            self.position = np.zeros(3, dtype=np.float32)
            self.rotation = Rotation()
            self.collide_next = False
            self.previous_step_collided = False
            self.render_calls = 0

        def get_agent_state(self):
            return SimpleNamespace(
                position=self.position.copy(),
                rotation=self.rotation,
            )

        def step_without_obs(self, action):
            assert action == "forward"
            self.previous_step_collided = self.collide_next
            if not self.collide_next:
                self.position[0] += 0.25
            self.collide_next = False

        def get_sensor_observation_at(self, *_args, **_kwargs):
            self.render_calls += 1
            return np.full((4, 4, 3), 1, dtype=np.uint8)

    monkeypatch.setattr(
        environments_module,
        "habitat_sim_action",
        lambda name: name.lower().replace("move_", ""),
    )
    simulator = FakeSimulator()
    env = object.__new__(VLNCEDaggerEnv)
    env.video_option = []
    env.raenwm_context_events = LowLevelContextEventBuffer(context_size=4)
    env._env = SimpleNamespace(
        sim=simulator,
        current_episode=object(),
        _task=SimpleNamespace(
            measurements=SimpleNamespace(update_measures=lambda **_kwargs: None)
        ),
    )

    assert env.wrap_act("forward", None) is None
    simulator.collide_next = True
    assert env.wrap_act("forward", None) is None

    assert simulator.render_calls == 0
    assert env.raenwm_context_events.pending_pose_count == 1
    final_rgb = np.full((4, 4, 3), 9, dtype=np.uint8)
    assert env._materialize_raenwm_context_events({"rgb": final_rgb}) == 1
    payload = env.pop_raenwm_context_events()
    assert simulator.render_calls == 0
    assert payload["stats"]["dropped_static_frames"] == 1
    assert payload["stats"]["single_sensor_renders"] == 0
    assert payload["stats"]["reused_rgb_frames"] == 1


def _bare_delayed_env(position=None):
    class Rotation:
        imag = np.zeros(3, dtype=np.float64)
        real = 1.0

    class FakeSimulator:
        def __init__(self):
            self.position = np.asarray(
                position if position is not None else [0.0, 0.0, 0.0],
                dtype=np.float32,
            )
            self.rotation = Rotation()

        def get_agent_state(self):
            return SimpleNamespace(
                position=self.position.copy(),
                rotation=self.rotation,
            )

        def set_agent_state(self, new_position, new_rotation):
            self.position = np.asarray(new_position, dtype=np.float32).copy()
            self.rotation = new_rotation
            return True

    env = object.__new__(VLNCEDaggerEnv)
    env.video_option = []
    env.video_frames = []
    env.plan_frames = []
    env.raenwm_context_events = LowLevelContextEventBuffer(context_size=4)
    env._env = SimpleNamespace(
        sim=FakeSimulator(),
        current_episode=SimpleNamespace(episode_id="0", scene_id="scene.glb"),
    )
    env.get_reward = lambda _observations: 0.0
    env.get_done = lambda _observations: False
    env.get_info = lambda _observations: {}
    return env


def test_back_path_materializes_only_latest_four_and_reuses_endpoint_rgb():
    env = _bare_delayed_env()
    replay_positions = []
    env._render_raenwm_context_rgb = lambda position, _rotation: (
        replay_positions.append(position.copy())
        or np.full((4, 4, 3), round(float(position[0]) * 10), np.uint8)
    )

    def move(count):
        for _ in range(count):
            previous = env._env.sim.position.copy()
            env._env.sim.position[0] += 0.25
            env._record_raenwm_context_observation(
                None,
                movement_frame=True,
                previous_position=previous,
            )

    env.multi_step_control = lambda _path, _tryout, _vis_info: move(2)
    env.single_step_control = lambda _pos, _tryout, _vis_info: move(3)
    endpoint_rgb = np.full((4, 4, 3), 77, dtype=np.uint8)
    env.get_observation_at = lambda _position, _rotation: {"rgb": endpoint_rgb}

    observations, _reward, _done, _info = env.step(
        {
            "act": 4,
            "back_path": [("vp", [0.0, 0.0, 0.0])],
            "tryout": False,
            "front_pos": [0.0, 0.0, 0.0],
            "ghost_pos": [1.25, 0.0, 0.0],
        }
    )
    payload = env.pop_raenwm_context_events()
    frames = [event for event in payload["events"] if event["type"] == "frame"]

    assert observations["rgb"] is endpoint_rgb
    assert len(frames) == 4
    assert len(replay_positions) == 3
    np.testing.assert_allclose(
        [event["position"][0] for event in frames],
        [0.5, 0.75, 1.0, 1.25],
    )
    np.testing.assert_array_equal(frames[-1]["rgb"], endpoint_rgb)
    assert payload["stats"]["trimmed_poses"] == 1
    assert payload["stats"]["single_sensor_renders"] == 3
    assert payload["stats"]["reused_rgb_frames"] == 1


def test_stop_after_teleport_reuses_stop_observation_without_replay(monkeypatch):
    env = _bare_delayed_env()
    monkeypatch.setattr(
        environments_module,
        "quat_from_heading",
        lambda _heading: env._env.sim.rotation,
    )
    replay_calls = []
    env._render_raenwm_context_rgb = lambda *_args: replay_calls.append(True)
    stop_rgb = np.full((4, 4, 3), 31, dtype=np.uint8)
    env._env.step = lambda action: (
        {"rgb": stop_rgb} if action == 0 else pytest.fail("unexpected action")
    )

    observations, _reward, _done, _info = env.step(
        {
            "act": 0,
            "back_path": None,
            "tryout": False,
            "stop_pos": [3.0, 0.0, 2.0],
        }
    )
    payload = env.pop_raenwm_context_events()
    frames = [event for event in payload["events"] if event["type"] == "frame"]

    assert observations["rgb"] is stop_rgb
    assert replay_calls == []
    assert [event["type"] for event in payload["events"]] == ["reset", "frame"]
    assert len(frames) == 1
    np.testing.assert_allclose(frames[0]["position"], [3.0, 0.0, 2.0])
    np.testing.assert_array_equal(frames[0]["rgb"], stop_rgb)
    assert payload["stats"]["single_sensor_renders"] == 0
    assert payload["stats"]["reused_rgb_frames"] == 1


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


def test_delayed_and_immediate_reference_produce_identical_context_latents():
    def deterministic_rgb(position, _rotation):
        value = round(float(position[0]) * 20)
        return np.full((4, 4, 3), value, dtype=np.uint8)

    positions = [
        np.asarray([(index + 1) * 0.25, 0.0, -0.1 * index], dtype=np.float32)
        for index in range(7)
    ]
    rotation = _yaw_rotation(0.0)
    buffer = LowLevelContextEventBuffer(context_size=4)
    buffer.reset_trace()
    for position in positions:
        buffer.append_pose(position, rotation, 0.0)
    buffer.materialize_pending(render_rgb=deterministic_rgb)
    delayed_payload = buffer.pop_payload()

    immediate_payload = _payload(
        [{"type": "reset"}]
        + [
            {
                "type": "frame",
                "rgb": deterministic_rgb(position, rotation),
                "position": position,
                "yaw": 0.0,
            }
            for position in positions[-4:]
        ]
    )
    delayed_runtime, _encoder, delayed_diagnostics = _drain(
        [delayed_payload], batch_size=2
    )
    immediate_runtime, _encoder, immediate_diagnostics = _drain(
        [immediate_payload], batch_size=2
    )

    delayed_frames = delayed_runtime.adapter.buffers[0].frames
    immediate_frames = immediate_runtime.adapter.buffers[0].frames
    assert len(delayed_frames) == len(immediate_frames) == 4
    for delayed, immediate in zip(delayed_frames, immediate_frames):
        np.testing.assert_array_equal(delayed.position, immediate.position)
        assert delayed.yaw == immediate.yaw
        torch.testing.assert_close(delayed.latent, immediate.latent)
    assert delayed_diagnostics["encoded_frames"] == 4.0
    assert immediate_diagnostics["encoded_frames"] == 4.0


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
            "valid_poses": 28,
            "trimmed_poses": 8,
            "single_sensor_renders": 12,
            "reused_rgb_frames": 8,
            "replay_render_seconds": 0.6,
        }
    )

    assert summary["reset_events"] == 3.0
    assert summary["frame_events"] == 20.0
    assert summary["context_ready_ratio"] == pytest.approx(0.7)
    assert summary["mean_drain_context_ready_ratio"] == pytest.approx(0.625)
    assert summary["frames_per_drain"] == 5.0
    assert summary["encode_seconds_per_frame"] == pytest.approx(0.1)
    assert summary["replay_render_seconds_per_frame"] == pytest.approx(0.05)


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


def test_synchronizer_rejects_more_than_context_size_frames_per_environment():
    payload = _payload(
        [_frame_event(index, [index * 0.25, 0.0, 0.0]) for index in range(5)]
    )

    with pytest.raises(ValueError, match="exceeding context_size=4"):
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


def test_single_sensor_replay_restores_pose_previous_observation_and_collision():
    class Sensor:
        def __init__(self, owner):
            self.owner = owner

        def draw_observation(self):
            return None

        def get_observation(self):
            return np.full(
                (2, 2, 4),
                round(float(self.owner.state.position[0]) * 10),
                dtype=np.uint8,
            )

    class SensorAdapter:
        @staticmethod
        def get_observation(observations):
            return observations["rgb"]

    class FakeSimulator:
        def __init__(self):
            self.config = SimpleNamespace(enable_batch_renderer=False)
            self.state = SimpleNamespace(
                position=np.asarray([4.0, 0.0, 2.0], dtype=np.float32),
                rotation=_yaw_rotation(0.5),
            )
            self._prev_sim_obs = {"collided": True, "marker": "final"}
            self._sensor_suite = SimpleNamespace(
                get=lambda _name: SensorAdapter()
            )
            self._sensors = {"rgb": Sensor(self)}

        def get_agent_state(self):
            return SimpleNamespace(
                position=self.state.position.copy(),
                rotation=self.state.rotation.copy(),
            )

        def set_agent_state(self, position, rotation, reset_sensors=False):
            assert reset_sensors is False
            self.state = SimpleNamespace(
                position=np.asarray(position, dtype=np.float32).copy(),
                rotation=np.asarray(rotation, dtype=np.float64).copy(),
            )
            return True

    fake = FakeSimulator()
    previous_sim_obs = fake._prev_sim_obs
    final_position = fake.state.position.copy()
    final_rotation = fake.state.rotation.copy()

    observation = Simulator.get_sensor_observation_at(
        fake,
        position=[1.0, 0.0, 0.0],
        rotation=_yaw_rotation(1.0),
        sensor_uuid="rgb",
    )

    assert int(observation[0, 0, 0]) == 10
    np.testing.assert_allclose(fake.state.position, final_position)
    np.testing.assert_allclose(fake.state.rotation, final_rotation)
    assert fake._prev_sim_obs is previous_sim_obs
    assert fake._prev_sim_obs["collided"] is True
