from types import SimpleNamespace

import numpy as np
import torch

from vlnce_baselines.common import environments as environments_module
from vlnce_baselines.common.environments import VLNCEDaggerEnv
from vlnce_baselines.models.R1Policy import pack_panoramic_observations
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer


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
