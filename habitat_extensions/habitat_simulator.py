#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Union,
    cast,
)

import numpy as np
from gym import spaces
from gym.spaces.box import Box
from numpy import ndarray

if TYPE_CHECKING:
    from torch import Tensor

import habitat_sim

from habitat_sim.simulator import MutableMapping, MutableMapping_T
from habitat.config import DictConfig as Config
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    DepthSensor,
    Observations,
    RGBSensor,
    SemanticSensor,
    Sensor,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
    VisualObservation,
)
from habitat.core.spaces import Space

# inherit habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py
@registry.register_simulator(name="Sim-v1")
class Simulator(HabitatSim):
    r"""Simulator wrapper over habitat-sim

    habitat-sim repo: https://github.com/facebookresearch/habitat-sim

    Args:
        config: configuration for initializing the simulator.
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get_sensor_observation_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        sensor_uuid: str = "rgb",
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[VisualObservation]:
        """Render one named sensor without evaluating the full sensor suite."""

        if self.config.enable_batch_renderer:
            raise RuntimeError(
                "single-sensor low-level context rendering is not supported "
                "with Habitat's batch renderer"
            )

        if sensor_uuid not in self._sensors:
            raise KeyError(f"Simulator has no sensor with UUID {sensor_uuid!r}")
        current_state = self.get_agent_state()
        previous_sim_obs = self._prev_sim_obs
        try:
            if position is None or rotation is None:
                success = True
            else:
                success = self.set_agent_state(
                    position,
                    rotation,
                    reset_sensors=False,
                )
            if not success:
                return None

            sensor = self._sensors[sensor_uuid]
            sensor.draw_observation()
            sim_obs = {sensor_uuid: sensor.get_observation()}
            self._prev_sim_obs = sim_obs
            return self._sensor_suite.get(sensor_uuid).get_observation(sim_obs)
        finally:
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
                # previous_step_collided is derived from _prev_sim_obs.  Replay
                # rendering must not change collision handling or task state.
                self._prev_sim_obs = previous_sim_obs

    def step_without_obs(self,
        action: Union[str, int, MutableMapping_T[int, Union[str, int]]],
        dt: float = 1.0 / 60.0,):
        self._num_total_frames += 1
        if isinstance(action, MutableMapping):
            return_single = False
        else:
            action = cast(Dict[int, Union[str, int]], {self._default_agent_id: action})
            return_single = True
        collided_dict: Dict[int, bool] = {}
        for agent_id, agent_act in action.items():
            agent = self.get_agent(agent_id)
            collided_dict[agent_id] = agent.act(agent_act)
            self.__last_state[agent_id] = agent.get_state()

        # # step physics by dt
        # step_start_Time = time.time()
        # super().step_world(dt)
        # self._previous_step_time = time.time() - step_start_Time

        multi_observations = {}
        for agent_id in action.keys():
            agent_observation = {}
            agent_observation["collided"] = collided_dict[agent_id]
            multi_observations[agent_id] = agent_observation

        if return_single:
            sim_obs = multi_observations[self._default_agent_id]
        else:
            sim_obs = multi_observations

        self._prev_sim_obs = sim_obs
