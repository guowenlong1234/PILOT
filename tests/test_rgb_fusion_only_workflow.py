from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "run_r2r/iter_train_rae_dino_native_cls_rgb_fusion.yaml"
JOB_PATH = ROOT / "scripts/run_rae_r2r_native_cls_rgb_fusion_server_job.sh"
MANAGER_PATH = ROOT / "scripts/manage_rae_r2r_native_cls_rgb_fusion_server.sh"
EVAL_MANAGER_PATH = (
    ROOT / "scripts/manage_rae_r2r_native_cls_rgb_fusion_eval_watch_host.sh"
)


def test_rgb_fusion_only_config_matches_the_native_cls_long_run_contract():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    il = config["IL"]
    model = config["MODEL"]
    nwm = model["RAENWM"]

    assert config["GPU_NUMBERS"] == 2
    assert config["NUM_ENVIRONMENTS"] == 4
    assert il["batch_size"] == 4
    assert il["gradient_accumulation_steps"] == 1
    assert il["iters"] == 10000
    assert il["log_every"] == 200
    assert il["load_from_ckpt"] is True
    assert il["is_requeue"] is False
    assert il["ckpt_to_load"] == (
        "pretrained/active_lookahead/base_iter14200.pth"
    )
    assert il["lr"] == 1.0e-5
    assert il["sample_ratio"] == 0.75
    assert il["decay_interval"] == 3000
    assert il["sample_ratio_iteration_offset"] == 14200
    assert il["sample_ratio_zero_threshold"] == 0.0
    assert il["warmup_iters"] == 0
    assert il["min_lr_ratio"] == 1.0
    assert il["waypoint_aug"] is True

    assert model["ACTIVE_LOOKAHEAD"] == {"enabled": False}
    assert nwm["enabled"] is True
    assert nwm["predict_cls_token"] is True
    assert nwm["token_count"] == 257
    assert nwm["num_steps"] == 10
    assert nwm["final_only_euler"] is False
    assert nwm["rgb_fusion_enabled"] is True
    assert nwm["rgb_fusion_trainable"] is True
    assert nwm["rgb_fusion_gate_bias_init"] == 0.0
    assert config["CHECKPOINT_FOLDER"].startswith(
        "data/logs/raenwm_rgb_fusion/native_cls_sft/"
    )


def test_rgb_fusion_only_launcher_validates_assets_without_lookahead_assets():
    launcher = JOB_PATH.read_text()
    for contract in (
        "1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61",
        "38b24af13b76ba8faef367559244c3a0401e0557e7c299870c273cbee8a07064",
        "84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59",
        'verify_sha256 "$START_CKPT" "$START_CKPT_SHA"',
        'verify_sha256 "$NWM_CHECKPOINT" "$NWM_CHECKPOINT_SHA"',
        'verify_sha256 "$NWM_STAT" "$NWM_STAT_SHA"',
        "IL.load_from_ckpt True IL.is_requeue False",
        "MODEL.ACTIVE_LOOKAHEAD.enabled False",
        "MODEL.RAENWM.rgb_fusion_enabled True",
        "MODEL.RAENWM.rgb_fusion_trainable True",
        "MODEL.RAENWM.rgb_fusion_gate_bias_init 0.0",
        "ETPR1_RGB_FUSION_ITERS:-10000",
        "ETPR1_RGB_FUSION_NPROC_PER_NODE:-2",
        "ETPR1_RGB_FUSION_NUM_ENVIRONMENTS:-4",
        "ETPR1_RGB_FUSION_BATCH_SIZE:-4",
    ):
        assert contract in launcher
    for forbidden in (
        "e24_avg3.pth",
        "dino_cwp_best.pt",
        "WARM_START",
    ):
        assert forbidden not in launcher


def test_rgb_fusion_only_manager_has_isolated_formal_and_smoke_outputs():
    manager = MANAGER_PATH.read_text()
    for contract in (
        "etpr1_native_cls_rgb_fusion_sft",
        "data/logs/raenwm_rgb_fusion/native_cls_sft",
        "run_rae_r2r_native_cls_rgb_fusion_server_job.sh",
        "smoke-single)",
        "ETPR1_RGB_FUSION_ITERS=2",
        "ETPR1_RGB_FUSION_SYNC_ENABLED=False",
        "ETPR1_RGB_FUSION_NPROC_PER_NODE=1",
        "ETPR1_RGB_FUSION_NUM_ENVIRONMENTS=1",
        "ETPR1_RGB_FUSION_BATCH_SIZE=1",
    ):
        assert contract in manager


def test_rgb_fusion_only_shell_entrypoints_parse():
    for path in (JOB_PATH, MANAGER_PATH, EVAL_MANAGER_PATH):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_rgb_fusion_only_eval_watcher_isolated_and_matches_training_config():
    manager = EVAL_MANAGER_PATH.read_text()
    for contract in (
        "iter_train_rae_dino_native_cls_rgb_fusion.yaml",
        "native_cls_sft/checkpoints/etpr1_native_cls_rgb_fusion_sft",
        "data/logs/raenwm_rgb_fusion/native_cls_eval",
        "etpr1_native_cls_rgb_fusion_eval_watch",
        "model_best_step_465000.pt",
        "ETPR1_RGB_FUSION_EVAL_NUM_ENVIRONMENTS:-8",
        "ETPR1_RGB_FUSION_EVAL_EPISODE_COUNT:--1",
        "ETPR1_RGB_FUSION_EVAL_CHECKPOINT_ORDER:-ascending",
        "ETPR1_R2R_EVAL_READY_SHA_REQUIRED=False",
        'manage_rae_r2r_eval_watch_host.sh" "${1:-status}"',
    ):
        assert contract in manager
