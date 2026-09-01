import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml


ROOT = Path(__file__).parents[1]


def test_native_cls_config_uses_reusable_q0_cache_contract():
    path = ROOT / "run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml"
    config = yaml.safe_load(path.read_text())
    nwm = config["MODEL"]["RAENWM"]
    active = config["MODEL"]["ACTIVE_LOOKAHEAD"]

    assert nwm["predict_cls_token"] is True
    assert nwm["token_count"] == 257
    assert nwm["checkpoint_sha256"] == (
        "38b24af13b76ba8faef367559244c3a0401e0557e7c299870c273cbee8a07064"
    )
    assert "head_checkpoint_path" not in nwm
    assert "head_checkpoint_sha256" not in nwm
    assert nwm["rgb_fusion_gate_bias_init"] == 0.0
    assert nwm["context_source"] == "low_level_move_rgb_anchor"
    assert nwm["low_level_encode_batch_size"] == 64
    assert config["IL"]["checkpoint_sync_destination"].endswith(
        "native_cls_e24_joint_sft/checkpoints/etpr1_native_cls_e24_joint_sft"
    )
    assert active["checkpoint_format_version"] == (
        "etpr1-native-cls-e24-joint-q0-cache-v4"
    )
    assert active["warm_start_checkpoint_path"] == ""
    assert active["warm_start_checkpoint_sha256"] == ""
    assert active["warm_start_source_context_contract"] == ""
    assert active["warm_start_expected_q0_contract"] == (
        "temporary_action_same_island_navmesh"
    )
    assert active["top5_cls_lr"] == 1.0e-5
    assert active["e24_head_lr"] == 5.0e-6

    inference = yaml.safe_load(
        (ROOT / "configs/nwm/raenwm_mp3d_fresh_cls.yaml").read_text()
    )
    assert inference["predict_cls_token"] is True
    assert inference["token_count"] == 257
    assert inference["transport"]["num_steps"] == 10
    assert inference["transport"]["sampling_method"] == "euler"
    assert inference["transport"]["final_only_euler"] is False


def test_native_cls_smoke_and_single_episode_wrappers_are_isolated():
    launcher = (ROOT / "scripts/run_rae_r2r_e24_joint_server_job.sh").read_text()
    assert "*native_cls*)" in launcher
    assert "pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar" in launcher
    assert "38b24af13b76ba8faef367559244c3a0401e0557e7c299870c273cbee8a07064" in launcher
    assert "ETPR1_E24_WARM_START_CHECKPOINT" in launcher
    assert "ETPR1_E24_WARM_START_SHA256" in launcher
    assert "ETPR1_E24_WARM_START_SOURCE_CONTEXT_CONTRACT" in launcher
    assert "ETPR1_E24_JOINT_GRADIENT_ACCUMULATION_STEPS:-1" in launcher
    assert 'IL.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"' in launcher
    assert 'global_batch_size=$((NPROC_PER_NODE * NUM_ENVIRONMENTS * GRADIENT_ACCUMULATION_STEPS))' in launcher
    assert "Weights-only migration requires checkpoint, SHA256, and source context contract" in launcher

    smoke = (
        ROOT / "scripts/manage_rae_r2r_native_cls_e24_joint_server.sh"
    ).read_text()
    for contract in (
        "iter_train_rae_dino_native_cls_e24_joint.yaml",
        "native_cls_e24_joint_smoke",
        "ETPR1_NATIVE_CLS_SYNC_DESTINATION",
        "native_cls_e24_joint_sft/checkpoints/${ETPR1_E24_JOINT_EXP_NAME}",
        "ETPR1_E24_JOINT_ITERS=2",
        "ETPR1_E24_JOINT_SYNC_ENABLED=False",
        "ETPR1_E24_JOINT_SMOKE_FREEZE_CHECK=True",
        "smoke-single)",
        "ETPR1_E24_JOINT_NPROC_PER_NODE=1",
        "ETPR1_E24_JOINT_NUM_ENVIRONMENTS=1",
        "ETPR1_E24_JOINT_BATCH_SIZE=1",
    ):
        assert contract in smoke

    single = (
        ROOT / "scripts/manage_native_cls_e24_single_episode_eval_host.sh"
    ).read_text()
    for contract in (
        "ETPR1_E24_EVAL_EXPECTED_CHECKPOINTS=1",
        "ETPR1_E24_EVAL_EPISODE_COUNT=1",
        "ETPR1_E24_EVAL_RESULT_EPISODE_COUNT=1",
        "ETPR1_E24_EVAL_SELECTION_ENABLED=False",
        "ETPR1_E24_EVAL_NUM_ENVIRONMENTS=1",
    ):
        assert contract in single


def test_active_lookahead_eval_queries_real_candidate_positions():
    source = (ROOT / "vlnce_baselines/ss_trainer_ETP_R1.py").read_text()
    condition = """if (
                mode == 'train'
                or self.config.VIDEO_OPTION
                or self._active_lookahead_enabled()
            ):"""
    assert condition in source


def test_native_cls_formal_eval_wrapper_uses_isolated_atomic_sync_directory():
    wrapper = (
        ROOT / "scripts/manage_rae_r2r_native_cls_e24_eval_watch_host.sh"
    ).read_text()
    for contract in (
        "run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml",
        "native_cls_e24_joint_sft/checkpoints/etpr1_native_cls_e24_joint_sft",
        "native_cls_e24_joint_eval",
        "etpr1_native_cls_e24_joint_eval_watch",
        "ETPR1_E24_EVAL_NUM_ENVIRONMENTS",
        "ETPR1_E24_EVAL_EXPECTED_CHECKPOINTS",
        "ETPR1_E24_EVAL_SELECTION_ENABLED=True",
        "ETPR1_E24_EVAL_CHECKPOINT_ORDER",
        "ETPR1_E24_EVAL_READY_SHA_REQUIRED=False",
        'manage_e24_joint_eval_watch_host.sh" "${1:-status}"',
    ):
        assert contract in wrapper


def test_native_cls_parity_tool_pins_complete_sequence_comparison():
    tool = (ROOT / "scripts/validate_native_cls_nwm_parity.py").read_text()
    for contract in (
        'CDiT_models["CDiT-B/2"]',
        "num_steps=10",
        'sampling_method="euler"',
        '"tokens"',
        '"cls_normalized"',
        '"patch_tokens"',
        '"max_abs"',
        '"cosine"',
        '"passed"',
        'default=0.1',
        'default=0.9999',
    ):
        assert contract in tool


def test_joint_config_and_launchers_fix_the_formal_contract():
    config = (ROOT / "run_r2r/iter_train_rae_dino_e24_joint.yaml").read_text()
    assert "iters: 10000" in config
    assert "log_every: 200" in config
    assert "sample_ratio_iteration_offset: 14200" in config
    assert "sample_ratio_zero_threshold: 0.0" in config
    assert "offline_topk: 5" in config
    assert "e24_action_warmup_iters: 400" in config
    assert "e24_replay_micro_batch_size: 2" in config
    assert "e24_replay_storage: cpu_fp16" in config
    assert "e24_head_gradient_clip_norm: 10.0" in config
    assert "dino_cwp_context_strategy: fixed_initial" in config
    assert "dino_cwp_heading_policy: face_motion" in config

    launcher = (ROOT / "scripts/run_rae_r2r_e24_joint_server_job.sh").read_text()
    for digest in (
        "1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61",
        "bae7a9664000235dfc6fb66b43a8a0a38e7645b876e6a6369f732eb7bf404ed8",
        "6a45291219907dd027203d224f3f8400631651a83bd01c45b1dea55d93ec0979",
    ):
        assert digest in launcher
    manager = (ROOT / "scripts/manage_rae_r2r_e24_joint_server.sh").read_text()
    assert "smoke)" in manager and "pilot)" in manager and "resume)" in manager
    status_body = manager.split("show_status() {", 1)[1].split("\n}", 1)[0]
    assert "return 0" in status_body
    assert 'habitat_version=getattr(habitat, "__version__", "unknown")' in launcher


def test_eval_watcher_requires_metrics_diagnostics_space_and_protected_gpu():
    watcher = (ROOT / "scripts/manage_e24_joint_eval_watch_host.sh").read_text()
    for contract in (
        "protected_task_running",
        "gpu_is_idle",
        "lookahead_ckpt_",
        "EXPECTED_CHECKPOINTS:-50",
        "MIN_RESERVE_GIB:-40",
        "ETPR1_E24_EVAL_EPISODE_COUNT:--1",
        'EVAL.EPISODE_COUNT "$EPISODE_COUNT"',
        "ETPR1_E24_EVAL_SELECTION_ENABLED:-True",
        "RESULT_EPISODE_COUNT=1839",
        '--episodes "$RESULT_EPISODE_COUNT"',
        "select_best_e24_joint_checkpoint.py",
    ):
        assert contract in watcher


def test_pilot_eval_wrapper_pins_one_checkpoint_and_sixteen_episodes():
    wrapper = (ROOT / "scripts/manage_e24_joint_pilot_eval_host.sh").read_text()
    for contract in (
        "20260825T165434",
        "etpr1_e24_joint_pilot",
        "etpr1_e24_joint_pilot_eval",
        "ETPR1_E24_EVAL_EXPECTED_CHECKPOINTS=1",
        "ETPR1_E24_EVAL_EPISODE_COUNT=16",
        "ETPR1_E24_EVAL_SELECTION_ENABLED=False",
        'manage_e24_joint_eval_watch_host.sh" "$ACTION"',
    ):
        assert contract in wrapper


def test_selection_tool_uses_sr_spl_tie_break_and_atomic_outputs(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    results = tmp_path / "results"
    output = tmp_path / "selection"
    checkpoints.mkdir()
    results.mkdir()
    rows = {
        200: (0.60, 0.55, 1.0),
        400: (0.59, 0.56, 2.0),
    }
    for iteration, (sr, spl, elapsed) in rows.items():
        checkpoint = checkpoints / f"ckpt.iter{iteration}.pth"
        checkpoint.write_bytes(f"checkpoint-{iteration}".encode())
        (results / f"stats_ckpt_{iteration}_val_unseen.json").write_text(
            json.dumps({"success": sr, "spl": spl})
        )
        (results / f"lookahead_ckpt_{iteration}_val_unseen.json").write_text(
            json.dumps(
                {
                    "episodes": 1839,
                    "oracle_q1_calls": 0,
                    "elapsed_seconds": elapsed,
                    "metrics": {"valid_rate": 0.5},
                }
            )
        )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/select_best_e24_joint_checkpoint.py"),
            "--checkpoint-dir", str(checkpoints),
            "--result-dir", str(results),
            "--output-dir", str(output),
            "--expected", "2",
        ],
        check=True,
    )
    selection = json.loads((output / "best_selection.json").read_text())
    assert selection["best"]["iteration"] == 400
    expected_sha = hashlib.sha256(b"checkpoint-400").hexdigest()
    assert selection["best"]["checkpoint_sha256"] == expected_sha
    assert not list(output.glob("*.tmp"))
