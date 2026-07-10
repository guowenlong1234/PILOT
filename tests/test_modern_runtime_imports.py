from pathlib import Path
import subprocess
import sys

import numpy as np


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
