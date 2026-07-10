from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_removed_numpy_bool_alias_is_not_used():
    source_paths = (
        ROOT / "vlnce_baselines" / "ss_trainer_ETP_R1.py",
        ROOT / "vlnce_baselines" / "GRPO_trainer_ETP_R1.py",
        ROOT / "pretrain_src" / "pretrain_src" / "data" / "common.py",
    )

    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        assert "dtype=np.bool)" not in source
        assert "dtype=np.bool_" in source


def test_removed_numpy_float_alias_is_not_used():
    source_paths = (
        ROOT / "vlnce_baselines" / "common" / "base_il_trainer.py",
        ROOT / "vlnce_baselines" / "ss_trainer_ETP_R1.py",
        ROOT / "vlnce_baselines" / "GRPO_trainer_ETP_R1.py",
        ROOT / "habitat_extensions" / "sensors.py",
    )

    for source_path in source_paths:
        source = source_path.read_text(encoding="utf-8")
        assert re.search(r"\bnp\.float(?![\w])", source) is None


def test_zero_length_sequence_masks_use_numpy_bool_dtype():
    from pretrain_src.pretrain_src.data.common import gen_seq_masks

    masks = gen_seq_masks([0, 0])

    assert masks.shape == (2, 0)
    assert masks.dtype == np.dtype(np.bool_)


def test_main_entrypoints_import_and_register_trainers_in_modern_runtime():
    import habitat_extensions.task  # noqa: F401
    import run  # noqa: F401
    import vlnce_baselines.GRPO_trainer_ETP_R1  # noqa: F401
    import vlnce_baselines.ss_trainer_ETP_R1  # noqa: F401
    from habitat_baselines.common.baseline_registry import baseline_registry

    assert np.__version__ == "1.26.4"
    assert baseline_registry.get_trainer("SS-ETP-R1") is not None
    assert baseline_registry.get_trainer("GRPO-R1") is not None


def test_legacy_config_can_merge_yaml_defrost_and_freeze():
    import vlnce_baselines  # noqa: F401
    from vlnce_baselines.config.default import get_config

    config = get_config("run_r2r/iter_train.yaml")

    assert config.TRAINER_NAME == "SS-ETP-R1"
    assert config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH == 224
    assert config.is_frozen()
    config.defrost()
    config.LOG_FILE = "compat.log"
    config.freeze()
    assert config.LOG_FILE == "compat.log"
    assert config.is_frozen()


@pytest.mark.parametrize(
    ("config_path", "sensor_name", "dataset_type", "agent_height"),
    (
        ("run_r2r/iter_train.yaml", "instruction_sensor", "VLN-CE-v2", 1.5),
        ("run_rxr/iter_train.yaml", "rxr_instruction_sensor", "RxR-VLN-CE-v2", 0.88),
    ),
)
def test_real_task_config_converts_to_modern_habitat_shape(
    config_path,
    sensor_name,
    dataset_type,
    agent_height,
):
    from habitat import read_write
    from omegaconf import OmegaConf
    from vlnce_baselines.common.environments import _task_config_for_habitat
    from vlnce_baselines.config.default import get_config

    legacy_config = get_config(config_path)
    task_config = _task_config_for_habitat(legacy_config)

    assert OmegaConf.is_config(task_config)
    assert task_config.seed == legacy_config.TASK_CONFIG.SEED
    assert task_config.environment.max_episode_steps == 5000
    assert task_config.simulator.agents_order == ["agent_0"]
    assert task_config.simulator.agents.agent_0.height == agent_height
    assert task_config.simulator.agents.agent_0.sim_sensors.rgb_sensor.width == 224
    assert task_config.simulator.agents.agent_0.sim_sensors.depth_sensor.height == 256
    assert sensor_name in task_config.task.lab_sensors
    assert task_config.dataset.type == dataset_type
    assert task_config.DATASET.TYPE == dataset_type
    assert task_config.is_frozen()
    assert OmegaConf.is_readonly(task_config)

    original_seed = task_config.seed
    with read_write(task_config):
        task_config.seed += 1
        assert not OmegaConf.is_readonly(task_config)
    assert task_config.seed == original_seed + 1
    assert OmegaConf.is_readonly(task_config)

    task_config.defrost()
    task_config.seed += 1
    assert not task_config.is_frozen()
    task_config.freeze()
    assert task_config.is_frozen()


def test_r1_env_passes_modern_config_to_habitat_rl_env(monkeypatch):
    import habitat
    from omegaconf import OmegaConf
    from vlnce_baselines.common.environments import VLNCEDaggerEnv
    from vlnce_baselines.config.default import get_config

    calls = {}
    dataset = object()

    def fake_rl_env_init(self, config, dataset):
        calls["config"] = config
        calls["dataset"] = dataset

    monkeypatch.setattr(habitat.RLEnv, "__init__", fake_rl_env_init)

    VLNCEDaggerEnv(get_config("run_r2r/iter_train.yaml"), dataset)

    assert OmegaConf.is_config(calls["config"])
    assert OmegaConf.is_readonly(calls["config"])
    assert calls["config"].simulator.agents.agent_0.sim_sensors.rgb_sensor.width == 224
    assert calls["dataset"] is dataset


def test_make_env_fn_converts_dataset_config_and_seeds_env(monkeypatch):
    import habitat
    from omegaconf import OmegaConf
    from vlnce_baselines.common.runtime_compat import make_env_fn
    from vlnce_baselines.config.default import get_config

    config = get_config("run_r2r/iter_train.yaml")
    calls = {}
    dataset = object()

    def fake_make_dataset(dataset_type, config):
        calls["dataset_type"] = dataset_type
        calls["dataset_config"] = config
        return dataset

    class FakeEnv:
        def __init__(self, config, dataset):
            calls["env_config"] = config
            calls["env_dataset"] = dataset

        def seed(self, seed):
            calls["seed"] = seed

    monkeypatch.setattr(habitat, "make_dataset", fake_make_dataset)

    env = make_env_fn(config, FakeEnv)

    assert isinstance(env, FakeEnv)
    assert calls["dataset_type"] == "VLN-CE-v2"
    assert OmegaConf.is_config(calls["dataset_config"])
    assert calls["dataset_config"].type == "VLN-CE-v2"
    assert calls["dataset_config"].TYPE == "VLN-CE-v2"
    assert calls["env_config"] is config
    assert calls["env_dataset"] is dataset
    assert calls["seed"] == config.TASK_CONFIG.SEED


def test_make_env_fn_does_not_swallow_dataset_errors(monkeypatch):
    import habitat
    from vlnce_baselines.common.runtime_compat import make_env_fn
    from vlnce_baselines.config.default import get_config

    class DatasetFailure(RuntimeError):
        pass

    def fail_make_dataset(*args, **kwargs):
        raise DatasetFailure("dataset failed")

    monkeypatch.setattr(habitat, "make_dataset", fail_make_dataset)

    with pytest.raises(DatasetFailure, match="dataset failed"):
        make_env_fn(get_config("run_r2r/iter_train.yaml"), object)


def test_legacy_compat_install_is_idempotent_and_preserves_native_config():
    import habitat
    import habitat.config as habitat_config
    import habitat.config.default as habitat_config_default
    import habitat_baselines.config.default as habitat_baselines_default
    from vlnce_baselines.common import runtime_compat

    assert (
        habitat_config_default.get_config
        is runtime_compat._NATIVE_HABITAT_GET_CONFIG
    )
    assert habitat_config.get_config is runtime_compat._NATIVE_HABITAT_GET_CONFIG
    assert habitat.get_config is runtime_compat._NATIVE_HABITAT_GET_CONFIG
    baseline_config = habitat_baselines_default._C
    baseline_config.defrost()
    baseline_config.TASK2_SENTINEL = "preserve"
    baseline_config.freeze()

    runtime_compat.install_legacy_config_compat()
    runtime_compat.install_legacy_config_compat()

    assert habitat_config_default.get_config is runtime_compat._NATIVE_HABITAT_GET_CONFIG
    assert habitat_config.get_config is runtime_compat._NATIVE_HABITAT_GET_CONFIG
    assert habitat.get_config is runtime_compat._NATIVE_HABITAT_GET_CONFIG
    assert habitat_baselines_default._C is baseline_config
    assert habitat_baselines_default._C.TASK2_SENTINEL == "preserve"


def test_action_compat_does_not_extend_modern_action_numbering():
    from habitat_extensions import habitat_sim_action
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    action_count = len(HabitatSimActions)
    known_actions = dict(HabitatSimActions._known_actions)

    assert len(HabitatSimActions) == action_count == 6
    assert habitat_sim_action("MOVE_FORWARD") == HabitatSimActions["move_forward"]
    assert habitat_sim_action("STOP") == HabitatSimActions["stop"]
    with pytest.raises(KeyError, match="Unknown Habitat-Sim action"):
        habitat_sim_action("NOT_AN_ACTION")
    assert dict(HabitatSimActions._known_actions) == known_actions


def test_waypoint_predictor_builds_with_transformers_bert_config():
    from transformers import BertConfig
    from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM

    predictor = BinaryDistPredictor_TRM(device="cpu")

    assert isinstance(predictor.waypoint_TRM.config, BertConfig)
    assert predictor.waypoint_TRM.config.num_hidden_layers == predictor.TRM_LAYER


def test_run_help_exits_successfully():
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "usage:" in result.stdout.lower()
