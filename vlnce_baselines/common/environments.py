from typing import Any, Dict, Optional, Tuple, List, Union
import math
import random
import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_extensions import habitat_sim_action
from habitat_extensions.utils import generate_video, heading_from_quaternion, navigator_video_frame, planner_video_frame
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
import cv2
import os

from vlnce_baselines.common.episode_iterator_state import (
    capture_episode_iterator_state,
    restore_episode_iterator_state,
)
from vlnce_baselines.nwm.low_level_context import (
    LOW_LEVEL_CONTEXT_SOURCE,
    LOW_LEVEL_RGB_SENSOR_UUID,
    LowLevelContextEventBuffer,
    normalize_context_source,
)


class _LegacyRootDictConfig(DictConfig):
    def __deepcopy__(self, memo):
        # OmegaConf 2.3 hardcodes DictConfig as the copied root type.
        return type(self)(super().__deepcopy__(memo))

    def defrost(self):
        OmegaConf.set_readonly(self, False)

    def freeze(self):
        OmegaConf.set_readonly(self, True)

    def is_frozen(self):
        return bool(OmegaConf.is_readonly(self))


def _to_omegaconf_compatible(value):
    if hasattr(value, "items"):
        return {
            key: _to_omegaconf_compatible(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_omegaconf_compatible(child) for child in value]
    return value


def _config_type(value, default):
    if isinstance(value, str):
        return value
    if hasattr(value, "keys"):
        keys = list(value.keys())
        if len(keys) == 1:
            return keys[0]
    return default


def _value(source, key, default=None):
    return source[key] if key in source else default


def _lowercase_fields(source, mapping):
    converted = dict(source)
    for legacy_name, modern_name in mapping:
        if legacy_name in source:
            converted[modern_name] = source[legacy_name]
    return converted


def _sensor_vector3(value, sensor_name, field_name):
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,):
        raise ValueError(
            f"{sensor_name} {field_name} must contain exactly 3 values, "
            f"got shape {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError(
            f"{sensor_name} {field_name} must contain only finite values"
        )
    return vector


def _task_config_for_habitat(config):
    legacy = _to_omegaconf_compatible(config.TASK_CONFIG)
    environment_legacy = legacy.get("ENVIRONMENT", {})
    iterator_legacy = environment_legacy.get("ITERATOR_OPTIONS", {})
    environment = {
        "max_episode_steps": _value(
            environment_legacy, "MAX_EPISODE_STEPS", 1000
        ),
        "max_episode_seconds": _value(
            environment_legacy, "MAX_EPISODE_SECONDS", 10000000
        ),
        "iterator_options": {
            "cycle": _value(iterator_legacy, "CYCLE", True),
            "shuffle": _value(iterator_legacy, "SHUFFLE", True),
            "group_by_scene": _value(iterator_legacy, "GROUP_BY_SCENE", True),
            "num_episode_sample": _value(
                iterator_legacy, "NUM_EPISODE_SAMPLE", -1
            ),
            "max_scene_repeat_episodes": _value(
                iterator_legacy, "MAX_SCENE_REPEAT_EPISODES", -1
            ),
            "max_scene_repeat_steps": _value(
                iterator_legacy, "MAX_SCENE_REPEAT_STEPS", 10000
            ),
            "step_repetition_range": _value(
                iterator_legacy, "STEP_REPETITION_RANGE", 0.2
            ),
        },
    }

    simulator_legacy = legacy.get("SIMULATOR", {})
    simulator = _lowercase_fields(
        simulator_legacy,
        (
            ("FORWARD_STEP_SIZE", "forward_step_size"),
            ("TURN_ANGLE", "turn_angle"),
            ("SCENE", "scene"),
            ("ACTION_SPACE_CONFIG", "action_space_config"),
        ),
    )
    simulator.setdefault("forward_step_size", 0.25)
    simulator.update(
        {
            "type": _config_type(simulator_legacy.get("TYPE"), "Sim-v1"),
            "scene": simulator.get("scene", ""),
            "scene_dataset": simulator.get("scene_dataset", "default"),
            "additional_object_paths": simulator.get(
                "additional_object_paths", []
            ),
            "default_agent_id": simulator.get("default_agent_id", 0),
            "default_agent_navmesh": simulator.get(
                "default_agent_navmesh", True
            ),
            "navmesh_include_static_objects": simulator.get(
                "navmesh_include_static_objects", False
            ),
            "debug_render": simulator.get("debug_render", False),
            "debug_render_articulated_agent": simulator.get(
                "debug_render_articulated_agent", False
            ),
            "kinematic_mode": simulator.get("kinematic_mode", False),
            "should_setup_semantic_ids": simulator.get(
                "should_setup_semantic_ids", True
            ),
            "debug_render_goal": simulator.get("debug_render_goal", True),
            "robot_joint_start_noise": simulator.get(
                "robot_joint_start_noise", 0.0
            ),
            "ctrl_freq": simulator.get("ctrl_freq", 120.0),
            "ac_freq_ratio": simulator.get("ac_freq_ratio", 4),
            "load_objs": simulator.get("load_objs", True),
            "hold_thresh": simulator.get("hold_thresh", 0.15),
            "grasp_impulse": simulator.get("grasp_impulse", 10000.0),
            "renderer": simulator.get(
                "renderer", {"enable_batch_renderer": False}
            ),
        }
    )

    habitat_sim_legacy = simulator_legacy.get("HABITAT_SIM_V0", {})
    habitat_sim_v0 = _lowercase_fields(
        habitat_sim_legacy,
        (
            ("GPU_DEVICE_ID", "gpu_device_id"),
            ("GPU_GPU", "gpu_gpu"),
            ("ALLOW_SLIDING", "allow_sliding"),
            ("FRUSTUM_CULLING", "frustum_culling"),
            ("ENABLE_PHYSICS", "enable_physics"),
        ),
    )
    habitat_sim_v0.setdefault("gpu_device_id", 0)
    habitat_sim_v0.setdefault("gpu_gpu", False)
    habitat_sim_v0.setdefault("allow_sliding", True)
    simulator["habitat_sim_v0"] = habitat_sim_v0

    agent_legacy = simulator_legacy.get("AGENT_0", {})
    sim_sensors = {}
    sensor_defaults = {
        "RGB_SENSOR": "HabitatSimRGBSensor",
        "DEPTH_SENSOR": "HabitatSimDepthSensor",
        "SEMANTIC_SENSOR": "HabitatSimSemanticSensor",
    }
    sensor_field_mapping = (
        ("TYPE", "type"),
        ("UUID", "uuid"),
        ("HEIGHT", "height"),
        ("WIDTH", "width"),
        ("HFOV", "hfov"),
        ("MIN_DEPTH", "min_depth"),
        ("MAX_DEPTH", "max_depth"),
        ("NORMALIZE_DEPTH", "normalize_depth"),
        ("POSITION", "position"),
        ("ORIENTATION", "orientation"),
    )
    for sensor_name in agent_legacy.get("SENSORS", []):
        sensor_legacy = simulator_legacy.get(sensor_name, {})
        sensor = _lowercase_fields(sensor_legacy, sensor_field_mapping)
        default_sensor_type = sensor_defaults.get(sensor_name)
        if default_sensor_type is None:
            for prefix, sensor_type in sensor_defaults.items():
                if sensor_name.startswith(prefix.removesuffix("_SENSOR")):
                    default_sensor_type = sensor_type
                    break
        sensor.setdefault("type", default_sensor_type or sensor_name)
        sensor_uuid = sensor_name.lower()
        if sensor_uuid.endswith("_sensor"):
            sensor_uuid = sensor_uuid[: -len("_sensor")]
        sensor.setdefault("uuid", sensor_uuid)
        sensor.setdefault("position", [0.0, 1.25, 0.0])
        sensor.setdefault("orientation", [0.0, 0.0, 0.0])
        sensor.pop("POSITION", None)
        sensor.pop("ORIENTATION", None)
        sensor["position"] = _sensor_vector3(
            sensor["position"], sensor_name, "position"
        )
        sensor["orientation"] = _sensor_vector3(
            sensor["orientation"], sensor_name, "orientation"
        )
        if "DEPTH" in sensor_name:
            sensor.setdefault("min_depth", 0.0)
            sensor.setdefault("max_depth", 10.0)
            sensor.setdefault("normalize_depth", True)
        sim_sensors[sensor_name.lower()] = sensor

    agent = _lowercase_fields(
        agent_legacy,
        (
            ("HEIGHT", "height"),
            ("RADIUS", "radius"),
            ("MAX_CLIMB", "max_climb"),
            ("MAX_SLOPE", "max_slope"),
            ("START_POSITION", "start_position"),
            ("START_ROTATION", "start_rotation"),
            ("IS_SET_START_STATE", "is_set_start_state"),
        ),
    )
    agent.update(
        {
            "height": agent.get("height", 1.5),
            "radius": agent.get("radius", 0.1),
            "max_climb": agent.get("max_climb", 0.2),
            "max_slope": agent.get("max_slope", 45.0),
            "grasp_managers": agent.get("grasp_managers", 1),
            "sim_sensors": sim_sensors,
            "is_set_start_state": agent.get("is_set_start_state", False),
            "start_position": agent.get("start_position", [0.0, 0.0, 0.0]),
            "start_rotation": agent.get(
                "start_rotation", [0.0, 0.0, 0.0, 1.0]
            ),
        }
    )
    simulator["agents"] = {"agent_0": agent}
    simulator["agents_order"] = ["agent_0"]

    task_legacy = legacy.get("TASK", {})
    task = dict(task_legacy)
    task["type"] = _config_type(task_legacy.get("TYPE"), "VLN-v0")
    task["physics_target_sps"] = task.get("physics_target_sps", 60.0)

    legacy_actions = task_legacy.get("ACTIONS", {})
    default_action_types = {
        "STOP": "StopAction",
        "MOVE_FORWARD": "MoveForwardAction",
        "TURN_LEFT": "TurnLeftAction",
        "TURN_RIGHT": "TurnRightAction",
        "HIGHTOLOW": "MoveHighToLowAction",
        "HIGHTOLOWEVAL": "MoveHighToLowActionEval",
        "HIGHTOLOWINFERENCE": "MoveHighToLowActionInference",
    }
    task["actions"] = {}
    for action_name in task_legacy.get("POSSIBLE_ACTIONS", []):
        action_legacy = legacy_actions.get(action_name, {})
        action = dict(action_legacy)
        action["type"] = _config_type(
            action_legacy.get("TYPE"), default_action_types[action_name]
        )
        task["actions"][action_name.lower()] = action

    default_sensor_types = {
        "INSTRUCTION_SENSOR": "InstructionSensor",
        "RXR_INSTRUCTION_SENSOR": "RxRInstructionSensor",
        "SHORTEST_PATH_SENSOR": "ShortestPathSensor",
        "VLN_ORACLE_PROGRESS_SENSOR": "VLNOracleProgressSensor",
    }
    task["lab_sensors"] = {}
    for sensor_name in task_legacy.get("SENSORS", []):
        sensor_legacy = task_legacy.get(sensor_name, {})
        sensor = _lowercase_fields(
            sensor_legacy,
            (
                ("TYPE", "type"),
                ("GOAL_RADIUS", "goal_radius"),
                ("USE_ORIGINAL_FOLLOWER", "use_original_follower"),
            ),
        )
        sensor["type"] = _config_type(
            sensor_legacy.get("TYPE"), default_sensor_types[sensor_name]
        )
        task["lab_sensors"][sensor_name.lower()] = sensor

    default_measurement_types = {
        "DISTANCE_TO_GOAL": "DistanceToGoal",
        "SUCCESS": "Success",
        "SPL": "SPL",
        "NDTW": "NDTW",
        "SDTW": "SDTW",
        "PATH_LENGTH": "PathLength",
        "ORACLE_SUCCESS": "OracleSuccess",
        "STEPS_TAKEN": "StepsTaken",
        "COLLISIONS": "Collisions",
        "POSITION": "Position",
        "POSITION_TRAIN": "PositionTrain",
        "POSITION_INFER": "PositionInfer",
        "TOP_DOWN_MAP_VLNCE": "TopDownMapVLNCE",
    }
    task["measurements"] = {}
    for measurement_name in task_legacy.get("MEASUREMENTS", []):
        measurement_legacy = task_legacy.get(measurement_name, {})
        measurement = _lowercase_fields(
            measurement_legacy,
            (
                ("TYPE", "type"),
                ("SUCCESS_DISTANCE", "success_distance"),
                ("GT_PATH", "gt_path"),
            ),
        )
        measurement["type"] = _config_type(
            measurement_legacy.get("TYPE"),
            default_measurement_types[measurement_name],
        )
        task["measurements"][measurement_name.lower()] = measurement

    dataset_legacy = legacy.get("DATASET", {})
    dataset = _lowercase_fields(
        dataset_legacy,
        (
            ("TYPE", "type"),
            ("SPLIT", "split"),
            ("SCENES_DIR", "scenes_dir"),
            ("CONTENT_SCENES", "content_scenes"),
            ("DATA_PATH", "data_path"),
            ("ROLES", "roles"),
            ("LANGUAGES", "languages"),
            ("EPISODES_ALLOWED", "episodes_allowed"),
            ("SUFFIX", "suffix"),
        ),
    )
    dataset.setdefault("type", "")
    dataset.setdefault("split", "train")
    dataset.setdefault("scenes_dir", "data/scene_datasets/")
    dataset.setdefault("content_scenes", ["*"])

    modern = dict(legacy)
    modern.update(
        {
            "seed": legacy.get("SEED", 100),
            "environment": environment,
            "simulator": simulator,
            "task": task,
            "dataset": dataset,
        }
    )
    task_config = _LegacyRootDictConfig(
        modern,
        flags={"allow_objects": True},
    )
    OmegaConf.set_readonly(task_config, True)
    return task_config


def quat_from_heading(heading, elevation=0):
    array_h = np.array([0, heading, 0])
    array_e = np.array([0, elevation, 0])
    rotvec_h = R.from_rotvec(array_h)
    rotvec_e = R.from_rotvec(array_e)
    quat = (rotvec_h * rotvec_e).as_quat()
    return quat

def calculate_vp_rel_pos(p1, p2, base_heading=0, base_elevation=0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    # xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    heading = np.arcsin(-dx / xz_dist)  # (-pi/2, pi/2)
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    # to (0, 2pi)
    while heading < 0:
        heading += 2*np.pi
    heading = heading % (2*np.pi)

    return heading, xz_dist

@baseline_registry.register_env(name="R1Env")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(_task_config_for_habitat(config), dataset)
        self.prev_episode_id = "something different"

        self.video_option = config.VIDEO_OPTION
        self.video_dir = config.VIDEO_DIR
        self.video_frames = []
        self.plan_frames = []
        raenwm_config = getattr(getattr(config, "MODEL", None), "RAENWM", None)
        self.raenwm_context_source = normalize_context_source(
            getattr(raenwm_config, "context_source", None)
        )
        self.raenwm_context_events = None
        if self.raenwm_context_source == LOW_LEVEL_CONTEXT_SOURCE:
            if not bool(getattr(raenwm_config, "enabled", False)):
                raise ValueError(
                    "low-level NWM context requires MODEL.RAENWM.enabled=True"
                )
            configured_sensors = {
                str(sensor).upper()
                for sensor in config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS
            }
            if "RGB_SENSOR" not in configured_sensors:
                raise ValueError(
                    "low-level NWM context requires the front RGB_SENSOR"
                )
            self.raenwm_context_events = LowLevelContextEventBuffer(
                pos_eps=float(getattr(raenwm_config, "pos_eps", 1.0e-3)),
                yaw_eps=float(getattr(raenwm_config, "yaw_eps", 1.0e-3)),
                rgb_key=LOW_LEVEL_RGB_SENSOR_UUID,
                context_size=int(getattr(raenwm_config, "context_size", 4)),
            )

    @property
    def original_action_space(self):
        return self.action_space

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self, 
        node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
    
    def get_pos_ori(self):
        agent_state = self._env.sim.get_agent_state()
        pos = agent_state.position
        ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])
        return (pos, ori)

    def _low_level_context_enabled(self):
        return self.raenwm_context_events is not None

    def _record_raenwm_context_observation(
        self,
        observations,
        *,
        movement_frame=False,
        previous_position=None,
    ):
        if not self._low_level_context_enabled():
            return False
        agent_state = self._env.sim.get_agent_state()
        return self.raenwm_context_events.append_pose(
            agent_state.position,
            agent_state.rotation,
            heading_from_quaternion(agent_state.rotation),
            observations=observations,
            movement_frame=bool(movement_frame),
            previous_position=previous_position,
        )

    @staticmethod
    def _rotation_from_raenwm_values(rotation):
        values = np.asarray(rotation, dtype=np.float64).reshape(4)
        return np.quaternion(values[3], values[0], values[1], values[2])

    def _render_raenwm_context_rgb(self, position, rotation):
        if not self._low_level_context_enabled():
            return None
        rgb = self._env.sim.get_sensor_observation_at(
            position,
            self._rotation_from_raenwm_values(rotation),
            sensor_uuid=LOW_LEVEL_RGB_SENSOR_UUID,
        )
        if rgb is None:
            raise RuntimeError("front RGB sensor returned no low-level observation")
        return rgb

    def _materialize_raenwm_context_events(self, observations):
        if not self._low_level_context_enabled():
            return 0
        agent_state = self._env.sim.get_agent_state()
        return self.raenwm_context_events.materialize_pending(
            render_rgb=self._render_raenwm_context_rgb,
            final_observations=observations,
            final_position=agent_state.position,
            final_rotation=agent_state.rotation,
        )

    def pop_raenwm_context_events(self):
        if not self._low_level_context_enabled():
            raise RuntimeError(
                "low-level context events requested while high-level mode is active"
            )
        return self.raenwm_context_events.pop_payload()

    def _normalize_candidate_q0_trajectory(self, trajectory):
        """Normalize each temporary action endpoint on the start navmesh island."""

        sim = self._env.sim
        pathfinder = sim.pathfinder
        raw_points = [
            np.asarray(point, dtype=np.float32).reshape(3) for point in trajectory
        ]
        try:
            raw_start = raw_points[0]
            normalized = (
                raw_start.copy()
                if sim.is_navigable(raw_start)
                else np.asarray(pathfinder.snap_point(raw_start), dtype=np.float32)
            )
            if (
                normalized.shape != (3,)
                or not np.isfinite(normalized).all()
                or not sim.is_navigable(normalized)
            ):
                raise RuntimeError("live agent start is not navigable")
            island = int(pathfinder.get_island(normalized))
            normalized_trace = [normalized.copy()]
            for step_index, raw_endpoint in enumerate(raw_points[1:], start=1):
                endpoint = np.asarray(
                    pathfinder.try_step(normalized, raw_endpoint), dtype=np.float32
                )
                if (
                    endpoint.shape == (3,)
                    and np.isfinite(endpoint).all()
                    and not sim.is_navigable(endpoint)
                ):
                    endpoint = np.asarray(
                        pathfinder.snap_point(endpoint, island), dtype=np.float32
                    )
                if (
                    endpoint.shape != (3,)
                    or not np.isfinite(endpoint).all()
                    or not sim.is_navigable(endpoint)
                ):
                    raise RuntimeError(
                        f"trajectory step {step_index} is not navigable"
                    )
                if int(pathfinder.get_island(endpoint)) != island:
                    raise RuntimeError(
                        f"trajectory step {step_index} changed navmesh island"
                    )
                normalized = endpoint
                normalized_trace.append(normalized.copy())
            raw_final = raw_points[-1]
            correction = normalized - raw_final
            return {
                "valid": True,
                "position": normalized.copy(),
                "navmesh_island": island,
                "normalized_trajectory": normalized_trace,
                "q0_was_normalized": not np.allclose(
                    normalized, raw_final, atol=1.0e-7, rtol=0.0
                ),
                "q0_normalization_distance_m": float(np.linalg.norm(correction)),
                "q0_normalization_horizontal_m": float(
                    np.hypot(correction[0], correction[2])
                ),
                "q0_normalization_vertical_m": float(abs(correction[1])),
                "error_type": None,
                "error": None,
            }
        except Exception as exc:
            return {
                "valid": False,
                "position": raw_points[-1].copy(),
                "navmesh_island": None,
                "normalized_trajectory": [],
                "q0_was_normalized": False,
                "q0_normalization_distance_m": 0.0,
                "q0_normalization_horizontal_m": 0.0,
                "q0_normalization_vertical_m": 0.0,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    def get_navigation_state(
        self,
        angles,
        forwards,
        include_current_goal_distance=True,
        include_candidate_goal_distances=True,
    ):
        """Return pose and all candidate states in one worker request."""
        if len(angles) != len(forwards):
            raise ValueError(
                "candidate angles and forward distances must have equal length"
            )

        sim = self._env.sim
        init_state = sim.get_agent_state()
        init_position = np.array(init_state.position, copy=True)
        init_rotation = init_state.rotation
        orientation = np.array(
            [*(init_rotation.imag), init_rotation.real],
            copy=True,
        )
        goal_position = None
        if (
            include_current_goal_distance
            or include_candidate_goal_distances
        ):
            goal_position = self._env.current_episode.goals[-1].position
        current_goal_distance = None
        if include_current_goal_distance:
            current_goal_distance = sim.geodesic_distance(
                init_position,
                goal_position,
            )

        forward_action = habitat_sim_action("MOVE_FORWARD")
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action
        ].actuation.amount
        candidate_positions = []
        candidate_goal_distances = []
        candidate_q0_records = []
        try:
            for angle, forward in zip(angles, forwards):
                trajectory = [init_position.copy()]
                post_position = init_position.copy()
                estimated_position = init_position.copy()
                estimated_heading = (
                    2.0
                    * np.arctan2(init_rotation.imag[1], init_rotation.real)
                    + float(angle)
                )
                estimated_position[0] -= float(forward) * np.sin(estimated_heading)
                estimated_position[2] -= float(forward) * np.cos(estimated_heading)
                try:
                    theta = (
                        np.arctan2(init_rotation.imag[1], init_rotation.real)
                        + angle / 2
                    )
                    rotation = np.quaternion(
                        np.cos(theta), 0, np.sin(theta), 0
                    )
                    sim.set_agent_state(init_position, rotation)
                    low_level_steps = int(forward // init_forward)
                    for _ in range(low_level_steps):
                        sim.step_without_obs(forward_action)
                        trajectory.append(
                            np.asarray(
                                sim.get_agent_state().position,
                                dtype=np.float32,
                            ).copy()
                        )
                    post_position = trajectory[-1].copy()
                    record = self._normalize_candidate_q0_trajectory(trajectory)
                    record.update(
                        {
                            "raw_position": post_position.copy(),
                            "estimated_position": estimated_position.copy(),
                            "candidate_angle_rad": float(angle),
                            "candidate_forward_m": float(forward),
                            "low_level_step_m": float(init_forward),
                            "trajectory_steps": low_level_steps,
                        }
                    )
                except Exception as exc:
                    record = {
                        "valid": False,
                        "position": post_position.copy(),
                        "raw_position": post_position.copy(),
                        "estimated_position": estimated_position.copy(),
                        "navmesh_island": None,
                        "normalized_trajectory": [],
                        "candidate_angle_rad": float(angle),
                        "candidate_forward_m": float(forward),
                        "low_level_step_m": float(init_forward),
                        "trajectory_steps": max(0, len(trajectory) - 1),
                        "q0_was_normalized": False,
                        "q0_normalization_distance_m": 0.0,
                        "q0_normalization_horizontal_m": 0.0,
                        "q0_normalization_vertical_m": 0.0,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                finally:
                    sim.set_agent_state(init_position, init_rotation)
                candidate_positions.append(post_position)
                candidate_q0_records.append(record)
                if include_candidate_goal_distances:
                    candidate_goal_distances.append(
                        sim.geodesic_distance(post_position, goal_position)
                    )
        finally:
            sim.set_agent_state(init_position, init_rotation)

        return {
            "position": init_position,
            "orientation": orientation,
            "current_goal_distance": current_goal_distance,
            "candidate_positions": candidate_positions,
            "candidate_goal_distances": candidate_goal_distances,
            "candidate_q0_records": candidate_q0_records,
        }

    def get_observation_at(self,
        source_position: List[float],
        source_rotation: List[Union[int, np.float64]],
        keep_agent_at_new_pose: bool = False):
        
        obs = self._env.sim.get_observations_at(source_position, source_rotation, keep_agent_at_new_pose)
        obs.update(self._env.task.sensor_suite.get_observations(
            observations=obs, episode=self._env.current_episode, task=self._env.task
        ))
        return obs

    def get_nwm_quality_views(self, requests):
        """Diagnostic RGB at explicit poses; does not step or update context.

        Caller separates observed-history requests from oracle target labels.
        Production context construction uses recorded images, never this API.
        """
        sim = self._env.sim
        before = sim.get_agent_state()
        result = []
        for request in requests:
            rgb = sim.get_sensor_observation_at(
                request['position'], self._rotation_from_raenwm_values(request['rotation']),
                sensor_uuid='rgb')
            result.append({'rgb': np.asarray(rgb).copy(),
                           'navigable': bool(sim.is_navigable(request['position']))})
        after = sim.get_agent_state()
        if not np.allclose(before.position, after.position, atol=1e-7, rtol=0):
            raise RuntimeError('diagnostic rendering changed agent position')
        if abs(float(np.dot(np.r_[before.rotation.imag,before.rotation.real],
                            np.r_[after.rotation.imag,after.rotation.real]))) < 1-1e-7:
            raise RuntimeError('diagnostic rendering changed agent rotation')
        return result

    def current_dist_to_goal(self, is_train):
        init_state = self._env.sim.get_agent_state() 
        if is_train:
            init_distance = self._env.sim.geodesic_distance(
                init_state.position, self._env.current_episode.goals[-1].position,
            )
        else:
            init_distance = self._env.sim.geodesic_distance(
                init_state.position, self._env.current_episode.goals[-1].position,
            )
        return init_distance
    
    def point_dist_to_goal(self, pos, is_train):
        if is_train:
            dist = self._env.sim.geodesic_distance(
                pos, self._env.current_episode.goals[-1].position,
            )
        else:
            dist = self._env.sim.geodesic_distance(
                pos, self._env.current_episode.goals[-1].position,
            )
        return dist
    
    
    def get_cand_real_pos(self, forward, angle):
        '''get cand real_pos by executing action'''
        state = self.get_navigation_state(
            angles=[angle],
            forwards=[forward],
            include_current_goal_distance=False,
            include_candidate_goal_distances=False,
        )
        return state["candidate_positions"][0]

    def current_dist_to_refpath(self, path):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        current_pos = init_state.position
        circle_dists = []
        for pos in path:
            circle_dists.append(
                self._env.sim.geodesic_distance(current_pos, pos)
            )
        # circle_dists = np.linalg.norm(np.array(path)-current_pos, axis=1).tolist()
        return circle_dists

    def ghost_dist_to_ref(self, ghost_vp_pos, ref_path):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress

        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0] 
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] 
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes 
                if np.all(compare): 
                    sub_goal_idx = cand_idxes[-1]
                else: 
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        # ghost dis to subgoal
        ghost_dists_to_subgoal = []
        for ghost_vp, ghost_pos in ghost_vp_pos:
            dist = self._env.sim.geodesic_distance(ghost_pos, sub_goal_pos)
            ghost_dists_to_subgoal.append(dist)

        oracle_ghost_vp = ghost_vp_pos[np.argmin(ghost_dists_to_subgoal)][0]
        self.prev_episode_id = episode_id
            
        return oracle_ghost_vp

    def get_cand_idx(self, ref_path, angles, distances, candidate_length):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress

        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        cand_dists_to_goal = []
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        for k in range(len(angles)):
            angle_k = angles[k]
            forward_k = distances[k]
            dist_k = self.cand_dist_to_subgoal(angle_k, forward_k, sub_goal_pos)
            # distance to subgoal
            cand_dists_to_goal.append(dist_k)

        # distance to final goal
        curr_dist_to_goal = self.current_dist_to_goal()
        # if within target range (which def as 3.0)
        if curr_dist_to_goal < 1.5:
            oracle_cand_idx = candidate_length - 1
        else:
            oracle_cand_idx = np.argmin(cand_dists_to_goal)

        self.prev_episode_id = episode_id
        # if curr_dist_to_goal == np.inf:
            
        return oracle_cand_idx #, sub_goal_pos

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = habitat_sim_action("MOVE_FORWARD")
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def cand_dist_to_subgoal(self, 
        angle: float, forward: float,
        sub_goal: Any):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = habitat_sim_action("MOVE_FORWARD")
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        prev_pos = init_state.position
        dis = 0.
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
            pos = sim.get_agent_state().position
            dis += np.linalg.norm(prev_pos - pos)
            prev_pos = pos
        post_state = sim.get_agent_state()

        post_distance = self._env.sim.geodesic_distance(
            post_state.position, sub_goal,
        ) + dis

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def reset(self):
        observations = self._env.reset()
        if self._low_level_context_enabled():
            self.raenwm_context_events.reset_trace()
            self._record_raenwm_context_observation(observations)
            self._materialize_raenwm_context_events(observations)
        if self.video_option:
            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations, 
                    info,
                )
            ]
        return observations

    def get_episode_iterator_state(self):
        return capture_episode_iterator_state(self._env)

    def set_episode_iterator_state(self, state):
        restore_episode_iterator_state(self._env, state)
        self.prev_episode_id = "something different"

    def reset_current_episode(self):
        self._env._reset_stats()
        self.prev_episode_id = "something different"
        self._env.reconfigure(self._env._config)
        observations = self._env.task.reset(episode=self._env.current_episode)
        self._env._task.measurements.reset_measures(
            episode=self._env.current_episode, task=self._env.task
        )
        if self._low_level_context_enabled():
            self.raenwm_context_events.reset_trace()
            self._record_raenwm_context_observation(observations)
            self._materialize_raenwm_context_events(observations)
        return observations

    # def wrap_act(self, act, ang, dis, cand_wp, action_wp, oracle_wp, start_p, start_h):
    def wrap_act(self, act, vis_info):
        ''' wrap action, get obs if video_option '''
        observations = None
        is_context_move = (
            self._low_level_context_enabled()
            and act == habitat_sim_action("MOVE_FORWARD")
        )
        previous_position = None
        if is_context_move:
            previous_position = np.asarray(
                self._env.sim.get_agent_state().position,
                dtype=np.float32,
            ).copy()
        if self.video_option:
            observations = self._env.step(act)
            info = self.get_info(observations)
            self.video_frames.append(
                navigator_video_frame(
                    observations,
                    info,
                    vis_info,
                )
            )
        else:
            self._env.sim.step_without_obs(act)
            self._env._task.measurements.update_measures(
                episode=self._env.current_episode, action=act, task=self._env.task 
            )
        if is_context_move:
            self._record_raenwm_context_observation(
                observations,
                movement_frame=True,
                previous_position=previous_position,
            )
        return observations

    def turn(self, ang, vis_info):    
        ''' angle: 0 ~ 360 degree '''
        act_l = habitat_sim_action("TURN_LEFT")
        act_r = habitat_sim_action("TURN_RIGHT")
        uni_l = self._env.sim.get_agent(0).agent_config.action_space[act_l].actuation.amount
        ang_degree = math.degrees(ang)
        ang_degree = round(ang_degree / uni_l) * uni_l
        observations = None

        if 180 < ang_degree <= 360:
            ang_degree -= 360
        if ang_degree >=0:
            turns = [act_l] * ( ang_degree // uni_l)
        else:
            turns = [act_r] * (-ang_degree // uni_l)

        for turn in turns:
            observations = self.wrap_act(turn, vis_info)
        return observations

    def teleport(self, pos):
        self._env.sim.set_agent_state(pos, quat_from_heading(0))
        if self._low_level_context_enabled():
            self.raenwm_context_events.reset_trace()
            self._record_raenwm_context_observation(None)

    def single_step_control(self, pos, tryout, vis_info):
        act_f = habitat_sim_action("MOVE_FORWARD")
        uni_f = self._env.sim.get_agent(0).agent_config.action_space[act_f].actuation.amount
        agent_state = self._env.sim.get_agent_state()
        ang, dis = calculate_vp_rel_pos(agent_state.position, pos, heading_from_quaternion(agent_state.rotation))
        self.turn(ang, vis_info)

        ksteps = int(dis // uni_f)
        if not tryout:
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
        else:
            cnt = 0 
            for _ in range(ksteps):
                self.wrap_act(act_f, vis_info)
                if self._env.sim.previous_step_collided:
                    break
                else:
                    cnt += 1
            # left forward step
            ksteps = ksteps - cnt
            if ksteps > 0:
                try_ang = random.choice([math.radians(90), math.radians(270)]) # left or right randomly
                self.turn(try_ang, vis_info)
                if try_ang == math.radians(90):     # from left to right
                    turn_seqs = [
                        (0, 270),   # 90, turn_left=30, turn_right=330
                        (330, 300), # 60
                        (330, 330), # 30
                        (300, 30),  # -30
                        (330, 60),  # -60
                        (330, 90),  # -90
                    ]
                elif try_ang == math.radians(270):  # from right to left
                    turn_seqs = [
                        (0, 90),   # -90
                        (30, 60),  # -60
                        (30, 30),  # -30
                        (60, 330), # 30
                        (30, 300), # 60
                        (30, 270), # 90
                    ]
                # try each direction, if pos change, do tail_turns, then do left forward actions
                for turn_seq in turn_seqs:
                    # do head_turns
                    self.turn(math.radians(turn_seq[0]), vis_info)
                    prev_position = self._env.sim.get_agent_state().position
                    self.wrap_act(act_f, vis_info)
                    post_posiiton = self._env.sim.get_agent_state().position
                    # pos change
                    if list(prev_position) != list(post_posiiton):
                        # do tail_turns
                        self.turn(math.radians(turn_seq[1]), vis_info)
                        # do left forward actions
                        for _ in range(ksteps):
                            self.wrap_act(act_f, vis_info)
                            if self._env.sim.previous_step_collided:
                                break
                        break
    
    def multi_step_control(self, path, tryout, vis_info):
        for vp, vp_pos in path: #path[::-1]:
            self.single_step_control(vp_pos, tryout, vis_info)

    def get_plan_frame(self, vis_info):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        info = self.get_info(observations)

        frame = planner_video_frame(observations, info, vis_info)
        frame = cv2.copyMakeBorder(frame, 6,6,5,5, cv2.BORDER_CONSTANT, value=(255,255,255))
        self.plan_frames.append(frame)

    @staticmethod
    def _normalize_step_payload(action, vis_info):
        if (
            vis_info is None
            and isinstance(action, dict)
            and "action" in action
        ):
            return action["action"], action.get("vis_info")
        return action, vis_info

    def step(self, action, vis_info=None, *args, **kwargs):
        action, vis_info = self._normalize_step_payload(action, vis_info)
        act = action['act']

        if act == 4: # high to low
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to front node
            teleported = action['back_path'] is None
            if action['back_path'] is None:
                self.teleport(action['front_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)
            agent_state = self._env.sim.get_agent_state()
            if self.video_option:
                observations = self.get_observation_at(
                    agent_state.position,
                    agent_state.rotation,
                )
            else:
                observations = None
            if teleported and self._low_level_context_enabled() and self.video_option:
                attached = self.raenwm_context_events.attach_latest_observation(
                    observations,
                    agent_state.position,
                    agent_state.rotation,
                )
                if not attached:
                    raise RuntimeError(
                        "failed to reuse the video RGB for the teleport anchor"
                    )

            # 2. forward to ghost node
            self.single_step_control(action['ghost_pos'], action['tryout'], vis_info)
            agent_state = self._env.sim.get_agent_state()
            observations = self.get_observation_at(agent_state.position, agent_state.rotation)
            self._materialize_raenwm_context_events(observations)

        elif act == 0:   # stop
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to stop node
            teleported = action['back_path'] is None
            if action['back_path'] is None:
                self.teleport(action['stop_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)

            # 2. stop
            observations = self._env.step(act) 
            self._materialize_raenwm_context_events(observations)
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        vis_info,
                        
                    )
                )
                self.get_plan_frame(vis_info)

        else:
            raise NotImplementedError                

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        if self.video_option and done:
            # if 0 < info["spl"] <= 0.6:  #TODO backtrack
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2],
                checkpoint_idx=0,
                metrics={"SPL": round(info["spl"], 3)},
                tb_writer=None,
                fps=8,
            )
            # for pano visualization
            metrics={
                        # "sr": round(info["success"], 3), 
                        "spl": round(info["spl"], 3),
                        # "ndtw": round(info["ndtw"], 3),
                        # "sdtw": round(info["sdtw"], 3),
                    }
            metric_strs = []
            for k, v in metrics.items():
                metric_strs.append(f"{k}{v:.2f}")
            episode_id=self._env.current_episode.episode_id
            scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2]
            tmp_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
            tmp_name = tmp_name.replace(" ", "_").replace("\n", "_") + ".png"
            tmp_fn = os.path.join(self.video_dir, tmp_name)
            tmp = np.concatenate(self.plan_frames, axis=0)
            cv2.imwrite(tmp_fn, tmp)
            self.plan_frames = []

        return observations, reward, done, info

@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
