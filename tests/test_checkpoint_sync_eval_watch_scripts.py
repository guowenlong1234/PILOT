from pathlib import Path
import subprocess


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


def test_server_sync_does_not_let_remote_commands_consume_checkpoint_list():
    source = Path(
        "scripts/manage_r2r_sft_checkpoint_sync_server.sh"
    ).read_text(encoding="utf-8")

    assert source.count("ssh -n ") == 4
    assert '"${DEST_HOST}:${incoming}" </dev/null' in source
    assert "mapfile -t checkpoints" in source
    assert 'for checkpoint in "${checkpoints[@]}"' in source
    assert "done < <(find" not in source


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
        "rae_dinov2_etpnav_cls_768_raw_cls_20260810/best/model_best_step_220000.pt",
        "NUM_ENVIRONMENTS=${ETPR1_R2R_EVAL_NUM_ENVIRONMENTS:-8}",
        'valid_result "$result" && return 0',
        "protected_etpnav_task",
        "blocking_project_task",
        "ETPR1_R2R_EVAL_BLOCKING_PROCESS_PATTERN",
        "gpu_busy",
    )
    for token in required_tokens:
        assert token in source


def test_legacy452500_eval_watch_pins_current_sft_and_pretrain_blocker():
    source = Path(
        "scripts/manage_legacy452500_r2r_sft_eval_watch_host.sh"
    ).read_text(encoding="utf-8")

    required_tokens = (
        "r2r_sft_legacy452500_nonvisual_20260815/checkpoints/",
        "rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft",
        "r2r_sft_legacy452500_nonvisual_20260815/eval_watch_val_unseen",
        "model_best_step_220000.pt",
        "ETPR1_R2R_EVAL_CHECKPOINT_ORDER:-ascending",
        "pretrain_src/pretrain_src/train_r2r.py",
        'manage_rae_r2r_eval_watch_host.sh" "${1:-status}"',
    )
    for token in required_tokens:
        assert token in source


def test_checkpoint_order_and_watch_directory_come_from_one_config(tmp_path):
    checkpoint_dir = tmp_path / "received"
    checkpoint_dir.mkdir()
    for iteration in (1000, 200, 600):
        (checkpoint_dir / f"ckpt.iter{iteration}.pth").write_bytes(b"x")
    config = tmp_path / "config.yaml"
    config.write_text(
        "EVAL:\n"
        "  checkpoint_order: ascending\n"
        "IL:\n"
        "  checkpoint_sync_enabled: true\n"
        f"  checkpoint_sync_destination: 'a6000@10.10.10.2:{checkpoint_dir}'\n",
        encoding="utf-8",
    )
    command = f"""
        source scripts/checkpoint_order.sh
        order=$(checkpoint_order_from_config {config})
        directory=$(checkpoint_watch_dir_from_config {config})
        echo "$order|$directory"
        list_ordered_checkpoints "$directory" "$order"
    """

    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert result[0] == f"ascending|{checkpoint_dir}"
    assert [Path(path).name for path in result[1:]] == [
        "ckpt.iter200.pth",
        "ckpt.iter600.pth",
        "ckpt.iter1000.pth",
    ]


def test_checkpoint_order_can_prioritize_newest(tmp_path):
    for iteration in (1000, 200, 600):
        (tmp_path / f"ckpt.iter{iteration}.pth").write_bytes(b"x")
    command = (
        "source scripts/checkpoint_order.sh; "
        f"list_ordered_checkpoints {tmp_path} descending"
    )

    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert [Path(path).name for path in result] == [
        "ckpt.iter1000.pth",
        "ckpt.iter600.pth",
        "ckpt.iter200.pth",
    ]
