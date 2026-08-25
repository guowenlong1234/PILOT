from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlnce_baselines.common import environments as environments_module
from vlnce_baselines.common.environments import VLNCEDaggerEnv
from vlnce_baselines.models.R1Policy import pack_panoramic_observations
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.ss_trainer_ETP_R1 import (
    RLTrainer,
    _load_adamw_optimizer_state,
)
from vlnce_baselines import ss_trainer_ETP_R1 as sft_trainer_module
from vlnce_baselines.nwm.types import NwmPrediction


def _sensor(value, batch_size=2):
    return torch.full(
        (batch_size, 1, 1, 1),
        value,
        dtype=torch.float32,
    )


def test_pack_panoramic_observations_matches_legacy_view_order():
    observations = {
        "rgb": _sensor(100),
        "depth": _sensor(0),
    }
    for view_index, heading in enumerate(range(30, 360, 30), start=1):
        observations[f"rgb_{heading}"] = _sensor(100 + view_index)
        observations[f"depth_{heading}"] = _sensor(view_index)

    depth, rgb = pack_panoramic_observations(observations)

    expected_depth = torch.tensor(
        [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        dtype=torch.float32,
    )
    torch.testing.assert_close(depth[:12, 0, 0, 0], expected_depth)
    torch.testing.assert_close(
        depth[12:, 0, 0, 0],
        expected_depth,
    )
    torch.testing.assert_close(
        rgb[:12, 0, 0, 0],
        expected_depth + 100,
    )


def test_pack_panoramic_observations_sorts_shuffled_sensor_keys_numerically():
    observations = {}
    shuffled_headings = [
        120, 0, 330, 60, 300, 90, 270, 150, 240, 180, 210, 30,
    ]
    for heading in shuffled_headings:
        depth_key = "depth" if heading == 0 else f"depth_{heading}"
        rgb_key = "rgb" if heading == 0 else f"rgb_{heading}"
        observations[depth_key] = _sensor(heading, batch_size=1)
        observations[rgb_key] = _sensor(heading + 1000, batch_size=1)

    depth, rgb = pack_panoramic_observations(observations)

    expected_headings = torch.tensor(
        [0, 330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30],
        dtype=torch.float32,
    )
    torch.testing.assert_close(depth[:, 0, 0, 0], expected_headings)
    torch.testing.assert_close(rgb[:, 0, 0, 0], expected_headings + 1000)


def test_pack_panoramic_observations_rejects_incomplete_panorama():
    observations = {
        "rgb": _sensor(100),
        "depth": _sensor(0),
    }
    for heading in range(30, 330, 30):
        observations[f"rgb_{heading}"] = _sensor(100 + heading)
        observations[f"depth_{heading}"] = _sensor(heading)

    with pytest.raises(ValueError, match="Expected 12 panoramic depth sensors"):
        pack_panoramic_observations(observations)


class _RecordingNwmRuntime:
    def __init__(self):
        self.context_args = None
        self.queries = None

    def update_contexts(self, front_latents, positions, yaws):
        self.context_args = (front_latents, positions, yaws)

    def predict(self, queries):
        self.queries = list(queries)
        return NwmPrediction(
            pred_latent=None,
            meta={"empty": True, "records": [], "skipped": {}},
        )


def test_sft_nwm_shadow_bridge_builds_raw_candidate_queries_without_mutation(
    monkeypatch,
):
    trainer = object.__new__(RLTrainer)
    trainer.raenwm_runtime = _RecordingNwmRuntime()
    trainer.last_raenwm_prediction = None
    trainer._raenwm_context_source_logged = False
    monkeypatch.setattr(
        sft_trainer_module,
        "heading_from_quaternion",
        lambda orientation: float(orientation),
    )
    front_latents = torch.zeros(2, 768, 16, 16)
    cur_pos = [
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    ]
    cur_ori = [0.25, 0.5]
    cand_vp = [["0_0", "0_1"], ["1_0"]]
    cand_pos = [
        [
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        ],
        [np.asarray([1.0, 0.0, 0.0], dtype=np.float32)],
    ]
    cand_vp_before = [list(values) for values in cand_vp]
    cand_pos_before = [[value.copy() for value in values] for values in cand_pos]

    prediction = trainer._run_raenwm_prediction(
        front_latents,
        cur_pos,
        cur_ori,
        cand_vp,
        cand_pos,
    )

    assert prediction is trainer.last_raenwm_prediction
    assert trainer.raenwm_runtime.context_args[0] is front_latents
    assert trainer.raenwm_runtime.context_args[2] == [0.25, 0.5]
    assert [query.query_id for query in trainer.raenwm_runtime.queries] == [
        "0_0",
        "0_1",
        "1_0",
    ]
    assert [query.env_index for query in trainer.raenwm_runtime.queries] == [0, 0, 1]
    assert cand_vp == cand_vp_before
    for actual_env, expected_env in zip(cand_pos, cand_pos_before):
        for actual, expected in zip(actual_env, expected_env):
            np.testing.assert_array_equal(actual, expected)


def _fusion_config(*, trainable=False, gpu_numbers=1):
    return SimpleNamespace(
        GPU_NUMBERS=gpu_numbers,
        MODEL=SimpleNamespace(
            RGB_ENCODER=SimpleNamespace(type="rae_dinov2", output_size=768),
            RAENWM=SimpleNamespace(
                enabled=True,
                rgb_fusion_enabled=True,
                rgb_fusion_type="residual_gate",
                rgb_fusion_alpha=1.0,
                rgb_fusion_zero_init=True,
                rgb_fusion_trainable=trainable,
            ),
        ),
    )


def test_sft_rgb_fusion_groups_preview_queries_once_per_ghost(monkeypatch):
    trainer = object.__new__(RLTrainer)
    trainer.config = _fusion_config()
    monkeypatch.setattr(
        sft_trainer_module,
        "heading_from_quaternion",
        lambda value: float(value),
    )
    preview = SimpleNamespace
    queries = trainer._build_raenwm_preview_queries(
        [np.zeros(3, dtype=np.float32)],
        [0.25],
        [[
            preview(
                target_kind="new_ghost",
                target_vp="g0",
                position=np.asarray([1.0, 0.0, 0.0]),
            ),
            preview(
                target_kind="new_ghost",
                target_vp="g0",
                position=np.asarray([3.0, 0.0, 0.0]),
            ),
            preview(
                target_kind="node", target_vp="0", position=np.zeros(3)
            ),
        ]],
    )
    assert len(queries) == 1
    assert queries[0].query_id == "g0"
    np.testing.assert_array_equal(
        queries[0].target_position,
        np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
    )


def test_sft_rgb_fusion_checkpoint_loading_is_strict():
    trainer = object.__new__(RLTrainer)
    trainer.config = _fusion_config()
    trainer.device = torch.device("cpu")
    trainer.raenwm_rgb_fusion_adapter = None
    adapter = trainer._initialize_raenwm_rgb_fusion_adapter()
    state = adapter.state_dict()
    trainer._load_raenwm_rgb_fusion_from_checkpoint(
        {"raenwm_rgb_fusion_adapter_state_dict": state},
        allow_missing=False,
    )
    assert len(state) == 10
    with pytest.raises(ValueError, match="missing required"):
        trainer._load_raenwm_rgb_fusion_from_checkpoint({}, allow_missing=False)
    assert trainer._load_raenwm_rgb_fusion_from_checkpoint(
        {}, allow_missing=True
    ) is None


def test_sft_trainable_rgb_fusion_rejects_multiple_gpus():
    trainer = object.__new__(RLTrainer)
    trainer.config = _fusion_config(trainable=True, gpu_numbers=2)
    trainer.device = torch.device("cpu")
    trainer.raenwm_rgb_fusion_adapter = None
    with pytest.raises(RuntimeError, match="only GPU_NUMBERS=1"):
        trainer._initialize_raenwm_rgb_fusion_adapter()


def test_graph_map_keeps_goal_distances_aligned_with_real_positions():
    graph = GraphMap(
        has_real_pos=True,
        loc_noise=0.5,
        merge_ghost=True,
        ghost_aug=0.0,
    )
    current_position = np.zeros(3, dtype=np.float32)
    candidate_position = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    real_position = np.asarray([0.75, 0.0, 0.0], dtype=np.float32)

    graph.update_graph(
        prev_vp=None,
        step_id=1,
        cur_vp="0",
        cur_pos=current_position,
        cur_embeds=torch.ones(4),
        cand_vp=["0_0"],
        cand_pos=[candidate_position],
        cand_embeds=[torch.ones(4)],
        cand_real_pos=[real_position],
        cand_goal_dists=[3.5],
    )

    assert graph.ghost_goal_dists["g0"] == [3.5]
    np.testing.assert_array_equal(
        graph.ghost_real_pos["g0"][0],
        real_position,
    )

    graph.delete_ghost("g0")
    assert "g0" not in graph.ghost_goal_dists
    assert "g0" not in graph.ghost_real_pos


class _NoRpcEnv:
    def call_at(self, *_args, **_kwargs):
        raise AssertionError("cached R2R teacher action must not use RPC")


def test_r2r_teacher_action_uses_cached_goal_distance():
    trainer = object.__new__(RLTrainer)
    trainer.config = SimpleNamespace(
        IL=SimpleNamespace(expert_policy="spl"),
    )
    trainer.device = torch.device("cpu")
    trainer.envs = _NoRpcEnv()
    trainer.gmaps = [
        SimpleNamespace(
            ghost_real_pos={
                "g0": [np.asarray([1.0, 0.0, 0.0])],
                "g1": [np.asarray([2.0, 0.0, 0.0])],
            },
            ghost_goal_dists={
                "g0": [4.0],
                "g1": [2.0],
            },
        )
    ]

    action = trainer._teacher_action_new(
        [[None, "g0", "g1"]],
        [False],
        is_train=True,
        current_goal_distances=[8.0],
    )

    assert action.tolist() == [2]


class _EvalOnlyModule:
    def eval(self):
        return self


class _RecordingDdpNet:
    def __init__(self):
        self.module = SimpleNamespace(
            rgb_encoder=_EvalOnlyModule(),
            depth_encoder=_EvalOnlyModule(),
        )
        self.no_sync_active = False
        self.no_sync_calls = 0

    @contextmanager
    def no_sync(self):
        assert not self.no_sync_active
        self.no_sync_calls += 1
        self.no_sync_active = True
        try:
            yield
        finally:
            self.no_sync_active = False


class _FakePolicy:
    def __init__(self, net):
        self.net = net

    def train(self):
        return self


class _FakeOptimizer:
    def zero_grad(self, set_to_none=False):
        assert set_to_none is True


class _FakeScaler:
    def __init__(self):
        self._scale = 1.0

    def get_scale(self):
        return self._scale

    @staticmethod
    def scale(loss):
        return loss

    @staticmethod
    def step(_optimizer):
        return None

    @staticmethod
    def update():
        return None


def test_sft_gradient_accumulation_only_syncs_final_microbatch(monkeypatch):
    trainer = object.__new__(RLTrainer)
    net = _RecordingDdpNet()
    trainer.policy = _FakePolicy(net)
    trainer.waypoint_predictor = _EvalOnlyModule()
    trainer.world_size = 2
    trainer.local_rank = 1
    trainer.device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        IL=SimpleNamespace(
            gradient_accumulation_steps=2,
            log_cuda_memory=False,
        )
    )
    trainer.optimizer = _FakeOptimizer()
    trainer.scheduler = None
    trainer.scaler = _FakeScaler()

    forward_no_sync_states = []
    backward_no_sync_states = []

    def fake_rollout(_mode, _ml_weight, _sample_ratio):
        forward_no_sync_states.append(net.no_sync_active)
        loss = torch.ones((), requires_grad=True)
        loss.register_hook(
            lambda grad: (
                backward_no_sync_states.append(net.no_sync_active)
                or grad
            )
        )
        trainer.loss += loss
        trainer.logs["IL_loss"].append(float(loss))

    trainer.rollout = fake_rollout
    monkeypatch.setattr(
        sft_trainer_module,
        "autocast",
        nullcontext,
    )

    trainer._train_interval(
        interval=1,
        ml_weight=1.0,
        sample_ratio=0.75,
    )

    assert net.no_sync_calls == 1
    assert forward_no_sync_states == [True, False]
    assert backward_no_sync_states == [True, False]


@torch.no_grad()
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused AdamW requires CUDA",
)
def test_legacy_adamw_state_can_resume_with_fused_implementation():
    legacy_parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    legacy_optimizer = torch.optim.AdamW([legacy_parameter], lr=1e-5)
    legacy_parameter.grad = torch.ones_like(legacy_parameter)
    legacy_optimizer.step()
    legacy_state = legacy_optimizer.state_dict()

    resumed_parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    fused_optimizer = torch.optim.AdamW(
        [resumed_parameter],
        lr=1e-5,
        fused=True,
    )
    _load_adamw_optimizer_state(
        fused_optimizer,
        legacy_state,
        use_fused_adamw=True,
    )

    assert all(
        group["fused"] is True
        for group in fused_optimizer.param_groups
    )
    assert all(
        value.device == resumed_parameter.device
        for state in fused_optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )

    resumed_parameter.grad = torch.ones_like(resumed_parameter)
    fused_optimizer.step()


class _Rotation:
    def __init__(self):
        self.imag = np.zeros(3, dtype=np.float64)
        self.real = 1.0


class _State:
    def __init__(self, position, rotation):
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation = rotation


class _FakeSimulator:
    def __init__(self):
        self.state = _State(np.zeros(3), _Rotation())
        self.action = SimpleNamespace(actuation=SimpleNamespace(amount=0.25))

    def get_agent_state(self):
        return self.state

    def get_agent(self, _index):
        return SimpleNamespace(
            agent_config=SimpleNamespace(
                action_space={"forward": self.action}
            )
        )

    def set_agent_state(self, position, rotation):
        self.state = _State(np.array(position, copy=True), rotation)

    def step_without_obs(self, _action):
        self.state.position[0] += 0.25

    @staticmethod
    def geodesic_distance(source, target):
        return float(
            np.linalg.norm(np.asarray(source) - np.asarray(target))
        )


def test_navigation_state_batches_candidates_and_goal_distances(monkeypatch):
    monkeypatch.setattr(
        environments_module,
        "habitat_sim_action",
        lambda _name: "forward",
    )
    simulator = _FakeSimulator()
    env = object.__new__(VLNCEDaggerEnv)
    env._env = SimpleNamespace(
        sim=simulator,
        current_episode=SimpleNamespace(
            goals=[SimpleNamespace(position=np.asarray([2.0, 0.0, 0.0]))]
        ),
    )

    state = env.get_navigation_state(
        angles=[0.0, 0.0],
        forwards=[0.25, 0.5],
        include_current_goal_distance=True,
        include_candidate_goal_distances=True,
    )

    np.testing.assert_allclose(state["position"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        state["candidate_positions"],
        [[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]],
    )
    assert state["current_goal_distance"] == 2.0
    assert state["candidate_goal_distances"] == [1.75, 1.5]
    np.testing.assert_allclose(
        simulator.get_agent_state().position,
        [0.0, 0.0, 0.0],
    )
