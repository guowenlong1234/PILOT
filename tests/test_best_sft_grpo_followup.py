import json
import subprocess
import sys
from pathlib import Path


SELECTOR = Path("scripts/select_best_r2r_sft_checkpoint.py")
MONITOR = Path("scripts/manage_best_sft_grpo_followup_server.sh")


def write_run(root, name, values):
    result_dir = root / name / "results"
    checkpoint_dir = root / name / "checkpoints"
    result_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    for iteration, success, spl in values:
        (result_dir / f"stats_ckpt_{iteration}_val_unseen.json").write_text(
            json.dumps({"success": success, "spl": spl}), encoding="utf-8"
        )
        (checkpoint_dir / f"ckpt.iter{iteration}.pth").write_bytes(b"model")
    return f"{name}={result_dir}={checkpoint_dir}"


def test_selector_picks_one_global_best_by_success_plus_spl(tmp_path):
    first = write_run(
        tmp_path, "round1", [(200, 0.60, 0.40), (400, 0.55, 0.50)]
    )
    second = write_run(
        tmp_path, "round2", [(200, 0.65, 0.39), (400, 0.52, 0.52)]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--total-iterations",
            "400",
            "--checkpoint-interval",
            "200",
            "--candidate",
            first,
            "--candidate",
            second,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    selected = json.loads(result.stdout)

    assert selected["selection_metric"] == "success+spl"
    assert selected["validated_runs"] == 2
    assert selected["validated_results"] == 4
    assert selected["run"] == "round1"
    assert selected["iteration"] == 400
    assert selected["score"] == 1.05


def test_selector_rejects_an_incomplete_evaluation_set(tmp_path):
    candidate = write_run(tmp_path, "round1", [(200, 0.6, 0.5)])
    result = subprocess.run(
        [
            sys.executable,
            str(SELECTOR),
            "--total-iterations",
            "400",
            "--checkpoint-interval",
            "200",
            "--candidate",
            candidate,
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "missing=[400]" in result.stderr


def test_followup_runs_one_grpo_and_keeps_original_checkpoint_interval():
    source = MONITOR.read_text(encoding="utf-8")
    required_tokens = (
        "selection_metric=success+spl",
        'GRPO_LOG_EVERY=${ETPR1_BEST_SFT_GRPO_LOG_EVERY:-10}',
        'GRPO_KEEP_LAST_STATES=${ETPR1_BEST_SFT_GRPO_KEEP_LAST_STATES:-3}',
        'GRPO_KEEP_STATE_EVERY=${ETPR1_BEST_SFT_GRPO_KEEP_STATE_EVERY:-250}',
        'first="round1=${FIRST_REMOTE_RESULT_DIR}=${FIRST_REMOTE_CKPT_DIR}"',
        'second="round2=${SECOND_REMOTE_RESULT_DIR}=${SECOND_REMOTE_CKPT_DIR}"',
        '"$GRPO_MANAGER" start',
        "start_remote_eval",
        "remote_grpo_results_complete",
        "grpo_not_successfully_complete",
        "eval_disk_space",
        "TRAIN_MIN_FREE_GIB",
        "GRPO_CHECKPOINT_SYNC_DESTINATION",
    )
    for token in required_tokens:
        assert token in source

    assert source.count('"$GRPO_MANAGER" start') == 1
