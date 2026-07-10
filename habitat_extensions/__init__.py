def _ensure_config_aliases() -> None:
    import habitat
    import habitat.config as habitat_config
    import habitat.config.default as habitat_config_default
    import habitat.core.utils as habitat_core_utils

    try:
        from habitat.config import DictConfig as habitat_dict_config
    except ImportError:
        from habitat.config import Config as habitat_dict_config

    if not hasattr(habitat, "Config"):
        habitat.Config = habitat_dict_config
    if not hasattr(habitat_config, "Config"):
        habitat_config.Config = habitat_dict_config
    if not hasattr(habitat_config, "DictConfig"):
        habitat_config.DictConfig = habitat_dict_config
    if not hasattr(habitat_config_default, "Config"):
        habitat_config_default.Config = habitat_dict_config
    if not hasattr(habitat_config_default, "DictConfig"):
        habitat_config_default.DictConfig = habitat_dict_config
    if not hasattr(habitat_core_utils, "try_cv2_import"):
        def try_cv2_import():
            import cv2

            return cv2

        habitat_core_utils.try_cv2_import = try_cv2_import


def _ensure_action_aliases() -> None:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    for legacy_name, current_name in (
        ("STOP", "stop"),
        ("MOVE_FORWARD", "move_forward"),
        ("TURN_LEFT", "turn_left"),
        ("TURN_RIGHT", "turn_right"),
    ):
        if (
            HabitatSimActions.has_action(current_name)
            and not HabitatSimActions.has_action(legacy_name)
        ):
            HabitatSimActions._known_actions[legacy_name] = HabitatSimActions[
                current_name
            ]


_ensure_config_aliases()
_ensure_action_aliases()

from habitat_extensions import measures, obs_transformers, sensors, nav
from habitat_extensions.config.default import get_extended_config
from habitat_extensions.task import VLNCEDatasetV1, VLNCEDatasetV2
from habitat_extensions.habitat_simulator import Simulator
