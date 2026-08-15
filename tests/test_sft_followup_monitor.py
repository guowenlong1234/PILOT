from pathlib import Path


def test_followup_monitor_waits_for_both_jobs_and_verifies_final_best():
    source = Path(
        "scripts/manage_eval_best_pretrain_sft_followup_server.sh"
    ).read_text(encoding="utf-8")

    required_tokens = (
        "ckpt.iter${CURRENT_ITERS}.pth",
        "train_state.iter${CURRENT_ITERS}.pth",
        "grep -q '^exit_code=0$'",
        "remote_pretrain_running",
        "model_step_500000.pt",
        "train_state_500000.pt",
        "remote_pretrain_not_successfully_complete",
        "final_best_metrics_unavailable",
        "final_best_copy_failed",
        "followup_launch_failed",
        "best_metrics.json",
        'model != f"model_best_step_{step}.pt"',
        "sha256sum '$remote_checkpoint'",
        'scp -p "${REMOTE_HOST}:${remote_checkpoint}" "$part"',
        'mv -- "$part" "$local_checkpoint"',
        "ETPR1_R2R_SFT_ITERS=15000",
        "ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS=1",
        'ETPR1_R2R_SFT_PRETRAINED_PATH="$local_checkpoint"',
        "ETPR1_R2R_SFT_CHECKPOINT_SYNC_ENABLED=True",
        'manage_rae_r2r_sft_server.sh" start',
        "current_sft_not_successfully_complete",
        "remote_pretrain_running",
    )
    for token in required_tokens:
        assert token in source


def test_followup_monitor_uses_new_output_and_does_not_overwrite_it():
    source = Path(
        "scripts/manage_eval_best_pretrain_sft_followup_server.sh"
    ).read_text(encoding="utf-8")

    assert "r2r_sft_eval_best${best_step}_${launch_date}" in source
    assert 'if [ -e "${REPO_ROOT}/${followup_output}" ]' in source
    assert "Refusing to overwrite existing follow-up output" in source
    assert "launched.env" in source
