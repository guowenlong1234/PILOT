from pathlib import Path


def test_server_sync_publishes_only_complete_checkpoints_atomically():
    source = Path(
        "scripts/manage_r2r_sft_checkpoint_sync_server.sh"
    ).read_text(encoding="utf-8")

    assert "train_state.iter${iteration}.pth" in source
    assert "[ -s \"$training_state\" ]" in source
    assert ".incoming/${base}.part.$$" in source
    assert "remote_size" in source
    assert "mv '$incoming' '$remote_checkpoint'" in source
    assert "a6000@10.10.10.2" in source


def test_eval_watch_uses_isolated_container_and_skips_valid_results():
    source = Path(
        "scripts/manage_rae_r2r_eval_watch_host.sh"
    ).read_text(encoding="utf-8")

    required_tokens = (
        "gwl-etpr1-rae",
        "gwl-etpnav",
        "conda activate etpr1_rae",
        "scripts/etpr1_rae_runtime_exec.sh python run.py",
        "EVAL.EPISODE_COUNT -1",
        "EVAL.SAVE_RESULTS True",
        "NUM_ENVIRONMENTS=${ETPR1_R2R_EVAL_NUM_ENVIRONMENTS:-8}",
        'valid_result "$result" && return 0',
        "protected_etpnav_task",
        "gpu_busy",
    )
    for token in required_tokens:
        assert token in source
