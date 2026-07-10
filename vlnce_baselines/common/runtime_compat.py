from pathlib import Path
from typing import List, Optional, Union

from yacs.config import CfgNode


class LegacyConfig(CfgNode):
    """YACS config node matching the interface used by legacy ETP-R1."""

    def __init__(self, init_dict=None, *args, new_allowed=True, **kwargs):
        super().__init__(
            init_dict=init_dict,
            *args,
            new_allowed=new_allowed,
            **kwargs,
        )


def _node(**values) -> LegacyConfig:
    return LegacyConfig(values)


def _legacy_habitat_defaults() -> LegacyConfig:
    config = _node(
        SEED=100,
        ENVIRONMENT=_node(
            MAX_EPISODE_STEPS=1000,
            ITERATOR_OPTIONS=_node(
                SHUFFLE=True,
                MAX_SCENE_REPEAT_STEPS=10000,
            ),
        ),
        TASK=_node(ACTIONS=_node()),
        SIMULATOR=_node(
            AGENT_0=_node(SENSORS=[]),
            HABITAT_SIM_V0=_node(GPU_DEVICE_ID=0),
        ),
        DATASET=_node(CONTENT_SCENES=["*"]),
    )
    return config


def get_legacy_habitat_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> LegacyConfig:
    config = _legacy_habitat_defaults()
    if config_paths:
        paths = (
            config_paths.split(",")
            if isinstance(config_paths, str)
            else config_paths
        )
        for config_path in paths:
            config.merge_from_file(str(Path(config_path)))
    if opts:
        config.merge_from_list(opts)
    config.freeze()
    return config


def _legacy_baselines_defaults() -> LegacyConfig:
    return _node(
        BASE_TASK_CONFIG_PATH="habitat_extensions/config/r2r_vlnce.yaml",
        TASK_CONFIG=_node(),
        CMD_TRAILING_OPTS=[],
        TRAINER_NAME="dagger",
        ENV_NAME="VLNCEDaggerEnv",
        SIMULATOR_GPU_ID=0,
        TORCH_GPU_ID=0,
        VIDEO_OPTION=[],
        TENSORBOARD_DIR="data/tensorboard_dirs/debug",
        VIDEO_DIR="videos/debug",
        TEST_EPISODE_COUNT=-1,
        EVAL_CKPT_PATH_DIR="data/checkpoints",
        NUM_ENVIRONMENTS=1,
        NUM_PROCESSES=-1,
        SENSORS=["RGB_SENSOR", "DEPTH_SENSOR"],
        CHECKPOINT_FOLDER="data/checkpoints",
        LOG_FILE="train.log",
        EVAL=_node(SPLIT="val", USE_CKPT_CONFIG=True),
        RL=_node(POLICY=_node(OBS_TRANSFORMS=_node())),
    )


def install_legacy_config_compat() -> None:
    import habitat
    import habitat.config as habitat_config
    import habitat.config.default as habitat_config_default
    import habitat_baselines.config.default as habitat_baselines_default

    for module in (habitat, habitat_config, habitat_config_default):
        module.Config = LegacyConfig

    habitat.get_config = get_legacy_habitat_config
    habitat_config.get_config = get_legacy_habitat_config
    habitat_config_default.get_config = get_legacy_habitat_config
    habitat_baselines_default._C = _legacy_baselines_defaults()


def get_env_class(env_name):
    from habitat_baselines.common.baseline_registry import baseline_registry

    return baseline_registry.get_env(env_name)


def make_env_fn(config, env_class):
    from habitat import make_dataset

    dataset = make_dataset(
        config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(config.TASK_CONFIG.SEED)
    return env
