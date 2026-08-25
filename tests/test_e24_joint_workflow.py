import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]


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


def test_eval_watcher_requires_metrics_diagnostics_space_and_protected_gpu():
    watcher = (ROOT / "scripts/manage_e24_joint_eval_watch_host.sh").read_text()
    for contract in (
        "protected_task_running",
        "gpu_is_idle",
        "lookahead_ckpt_",
        "EXPECTED_CHECKPOINTS:-50",
        "MIN_RESERVE_GIB:-40",
        "select_best_e24_joint_checkpoint.py",
    ):
        assert contract in watcher


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
