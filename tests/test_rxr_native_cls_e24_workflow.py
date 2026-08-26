import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).parents[1]


def test_rxr_baseline_and_joint_configs_preserve_task_contract():
    baseline = yaml.safe_load(
        (ROOT / "run_rxr/iter_train_rae_dino_sft.yaml").read_text()
    )
    joint = yaml.safe_load(
        (ROOT / "run_rxr/iter_train_rae_dino_native_cls_e24_joint.yaml").read_text()
    )
    for config in (baseline, joint):
        assert config["BASE_TASK_CONFIG_PATH"] == "run_rxr/rxr_vlnce.yaml"
        assert config["MODEL"]["task_type"] == "rxr"
        assert config["IL"]["expert_policy"] == "ndtw"
        assert config["IL"]["max_text_len"] == 250
        assert config["IL"]["max_traj_len"] == 25
        assert config["TASK_CONFIG"]["DATASET"]["ROLES"] == ["guide"]
        assert config["TASK_CONFIG"]["DATASET"]["LANGUAGES"] == [
            "en-US", "en-IN", "hi-IN", "te-IN"
        ]
        assert config["EVAL"]["LANGUAGES"] == [
            "en-US", "en-IN", "hi-IN", "te-IN"
        ]
        assert config["INFERENCE"]["LANGUAGES"] == [
            "en-US", "en-IN", "hi-IN", "te-IN"
        ]
        assert config["TASK_CONFIG"]["DATASET"]["SUFFIX"] == "_90"
        assert config["IL"]["resumable_checkpoints"] is True
        assert config["IL"]["strict_rng_resume"] is True

    assert baseline["GPU_NUMBERS"] == 2
    assert baseline["NUM_ENVIRONMENTS"] == 6
    assert baseline["IL"]["gradient_accumulation_steps"] == 2
    assert baseline["IL"]["iters"] == 30000
    assert baseline["IL"]["lr"] == 1.5e-5
    assert baseline["IL"]["warmup_iters"] == 1000
    assert baseline["IL"]["min_lr_ratio"] == 0.6
    assert baseline["IL"]["decay_interval"] == 5000
    assert baseline["IL"]["checkpoint_sync_enabled"] is False

    assert joint["NUM_ENVIRONMENTS"] == 2
    assert joint["IL"]["gradient_accumulation_steps"] == 2
    assert joint["IL"]["iters"] == 10000
    assert joint["IL"]["sample_ratio_zero_threshold"] == 0.15
    assert joint["MODEL"]["RAENWM"]["predict_cls_token"] is True
    assert joint["MODEL"]["RAENWM"]["token_count"] == 257
    active = joint["MODEL"]["ACTIVE_LOOKAHEAD"]
    assert active["checkpoint_format_version"] == (
        "etpr1-rxr-native-cls-e24-joint-v1"
    )
    assert active["e24_action_warmup_iters"] == 400
    assert active["e24_replay_micro_batch_size"] == 2
    assert active["e24_replay_storage"] == "cpu_fp16"


def test_rxr_sensor_contract_is_hfov63():
    task = yaml.safe_load((ROOT / "run_rxr/rxr_vlnce.yaml").read_text())
    assert task["SIMULATOR"]["RGB_SENSOR"]["HFOV"] == 63
    assert task["SIMULATOR"]["DEPTH_SENSOR"]["HFOV"] == 63
    assert task["DATASET"]["ROLES"] == ["guide"]


def test_rxr_management_scripts_are_isolated_and_selection_is_explicit():
    baseline = (ROOT / "scripts/manage_rae_rxr_sft_server.sh").read_text()
    joint = (
        ROOT / "scripts/manage_rae_rxr_native_cls_e24_joint_server.sh"
    ).read_text()
    launcher = (ROOT / "scripts/run_rae_r2r_e24_joint_server_job.sh").read_text()
    for token in (
        "ETPR1_RXR_SFT_NPROC_PER_NODE:-2",
        "ETPR1_RXR_SFT_NUM_ENVIRONMENTS:-6",
        "ETPR1_RXR_SFT_BATCH_SIZE:-6",
        "ETPR1_RXR_SFT_GRADIENT_ACCUMULATION_STEPS:-2",
        "ETPR1_R2R_SFT_LR=1.5e-5",
        "smoke-resume)",
    ):
        assert token in baseline
    for token in (
        "ETPR1_RXR_ALLOW_SMOKE_BASE=True",
        "ETPR1_RXR_JOINT_NPROC_PER_NODE:-2",
        "ETPR1_RXR_JOINT_NUM_ENVIRONMENTS:-2",
        "smoke-resume)",
    ):
        assert token in joint
    for token in (
        "ETPR1_RXR_BASE_SELECTION",
        "resolve_rxr_base_selection.py",
        'MODEL.ACTIVE_LOOKAHEAD.base_checkpoint_sha256 "$START_CKPT_SHA"',
        'MODEL.ACTIVE_LOOKAHEAD.base_iteration "$BASE_ITERATION"',
        "base_selection_manifest_sha256",
        'IL.sample_ratio_iteration_offset "$BASE_ITERATION"',
        "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True",
        "TASK_CONFIG.DATASET.SUFFIX _90",
        "TASK_CONFIG.DATASET.ROLES",
        "TASK_CONFIG.DATASET.LANGUAGES",
    ):
        assert token in launcher


def _write_eval_result(result_dir, iteration, episode_ids, metrics, diagnostic=True):
    (result_dir / f"stats_ckpt_{iteration}_val_unseen.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    (result_dir / f"stats_ep_ckpt_{iteration}_val_unseen_r0_w1.json").write_text(
        json.dumps({str(value): {"success": 1} for value in episode_ids}),
        encoding="utf-8",
    )
    if diagnostic:
        (result_dir / f"lookahead_ckpt_{iteration}_val_unseen.json").write_text(
            json.dumps({
                "episodes": len(episode_ids),
                "oracle_q1_calls": 0,
                "metrics": {
                    "q0_batch_failures": 0,
                    "q0_row_failures": 0,
                    "cwp_batch_failures": 0,
                    "cwp_row_failures": 0,
                    "q1_batch_failures": 0,
                    "q1_row_failures": 0,
                    "valid_rate": 0.5,
                    "action_flip_rate": 0.1,
                },
            }), encoding="utf-8"
        )


def test_rxr_selector_uses_sdtw_order_and_writes_atomic_manifest(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    results = tmp_path / "results"
    output = tmp_path / "selection"
    checkpoints.mkdir()
    results.mkdir()
    episode_ids = ["1", "2"]
    rows = {
        200: {"sdtw": 0.7, "ndtw": 0.9, "spl": 0.8, "success": 0.8},
        400: {"sdtw": 0.7, "ndtw": 0.91, "spl": 0.1, "success": 0.1},
    }
    for iteration, metrics in rows.items():
        (checkpoints / f"ckpt.iter{iteration}.pth").write_bytes(
            f"checkpoint-{iteration}".encode()
        )
        _write_eval_result(results, iteration, episode_ids, metrics)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/select_best_rxr_sft_checkpoint.py"),
        "--mode", "joint", "--checkpoint-dir", str(checkpoints),
        "--result-dir", str(results), "--output-dir", str(output),
        "--expected-episodes", "2", "--total-iterations", "400",
        "--checkpoint-interval", "200", "--source-commit", "deadbeef",
    ], check=True)
    selection = json.loads((output / "best_selection.json").read_text())
    assert selection["best"]["iteration"] == 400
    assert selection["selection_rule"] == [
        "sdtw", "ndtw", "spl", "sr", "iteration"
    ]
    assert selection["format_version"] == (
        "etpr1-rxr-native-cls-e24-joint-selection-v1"
    )
    assert not list(output.glob("*.tmp*"))


def test_rxr_selector_rejects_partial_and_failed_joint_results(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    results = tmp_path / "results"
    checkpoints.mkdir()
    results.mkdir()
    (checkpoints / "ckpt.iter200.pth").write_bytes(b"200")
    _write_eval_result(
        results, 200, ["1", "2"],
        {"sdtw": 0.5, "ndtw": 0.5, "spl": 0.5, "success": 0.5},
    )
    command = [
        sys.executable, str(ROOT / "scripts/select_best_rxr_sft_checkpoint.py"),
        "--mode", "joint", "--checkpoint-dir", str(checkpoints),
        "--result-dir", str(results), "--output-dir", str(tmp_path / "out"),
        "--expected-episodes", "2", "--total-iterations", "400",
        "--checkpoint-interval", "200", "--source-commit", "deadbeef",
    ]
    partial = subprocess.run(command, capture_output=True, text=True)
    assert partial.returncode != 0
    assert "checkpoint iterations differ" in partial.stderr

    (checkpoints / "ckpt.iter400.pth").write_bytes(b"400")
    _write_eval_result(
        results, 400, ["1", "2"],
        {"sdtw": 0.6, "ndtw": 0.6, "spl": 0.6, "success": 0.6},
    )
    diagnostic = results / "lookahead_ckpt_400_val_unseen.json"
    payload = json.loads(diagnostic.read_text())
    payload["metrics"]["cwp_row_failures"] = 1
    diagnostic.write_text(json.dumps(payload))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0
    assert "contains NWM/CWP failures" in failed.stderr


def test_rxr_base_resolver_rejects_smoke_and_sha_tampering(tmp_path):
    checkpoint = tmp_path / "ckpt.iter2.pth"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = tmp_path / "selection.json"
    manifest.write_text(json.dumps({
        "format_version": "etpr1-rxr-sft-base-selection-v1",
        "task_type": "rxr",
        "dataset_role": "guide",
        "dataset_languages": ["en-US", "en-IN", "hi-IN", "te-IN"],
        "rgb_hfov": 63,
        "smoke_only": True,
        "best": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": digest,
            "iteration": 2,
        },
    }))
    base = [sys.executable, str(ROOT / "scripts/resolve_rxr_base_selection.py"), str(manifest)]
    rejected = subprocess.run(base, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "smoke-only" in rejected.stderr
    accepted = subprocess.run(base + ["--allow-smoke"], check=True, capture_output=True, text=True)
    assert accepted.stdout.splitlines()[0] == str(checkpoint.resolve())
    checkpoint.write_bytes(b"tampered")
    tampered = subprocess.run(base + ["--allow-smoke"], capture_output=True, text=True)
    assert tampered.returncode != 0
    assert "SHA256 mismatch" in tampered.stderr


def test_rxr_eval_and_sync_wrappers_enforce_safe_defaults():
    baseline = (ROOT / "scripts/manage_rae_rxr_sft_eval_watch_host.sh").read_text()
    joint = (ROOT / "scripts/manage_rae_rxr_native_cls_e24_eval_watch_host.sh").read_text()
    sync = (ROOT / "scripts/manage_r2r_sft_checkpoint_sync_server.sh").read_text()
    single = (ROOT / "scripts/run_rxr_single_episode_eval_server.sh").read_text()
    for source in (baseline, joint):
        assert "NUM_ENVIRONMENTS:-3" in source
        assert "EPISODE_COUNT:--1" in source
        assert "READY_SHA_REQUIRED=True" in source
    assert "ETPR1_E24_EVAL_SELECTION_ENABLED=False" in joint
    assert "sha256sum '$incoming'" in sync
    assert "ready_marker" in sync
    assert "EVAL.EPISODE_COUNT 1" in single
    assert 'TASK_CONFIG.DATASET.SUFFIX ""' in single
    assert "TASK_CONFIG.DATASET.LANGUAGES" in single
    assert "oracle_q1_calls" in single


@pytest.mark.parametrize("script", [
    "manage_rae_rxr_sft_server.sh",
    "manage_rae_rxr_native_cls_e24_joint_server.sh",
    "manage_rae_rxr_sft_eval_watch_host.sh",
    "manage_rae_rxr_native_cls_e24_eval_watch_host.sh",
    "run_rxr_single_episode_eval_server.sh",
    "finalize_rxr_sft_selection_host.sh",
])
def test_rxr_shell_scripts_parse(script):
    subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)
