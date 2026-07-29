from pathlib import Path

import pytest

from vlnce_baselines.common.online_checkpoint import (
    checkpoint_iteration,
    latest_checkpoint_path,
    prune_checkpoints,
)


def _checkpoint(directory, iteration):
    path = directory / f"ckpt.iter{iteration}.pth"
    path.write_bytes(str(iteration).encode())
    return path


def test_latest_checkpoint_uses_numeric_iteration(tmp_path):
    _checkpoint(tmp_path, 200)
    expected = _checkpoint(tmp_path, 1000)
    _checkpoint(tmp_path, 800)

    assert latest_checkpoint_path(tmp_path) == str(expected)


def test_prune_keeps_recent_and_milestones(tmp_path):
    for iteration in range(200, 6200, 200):
        _checkpoint(tmp_path, iteration)

    removed = prune_checkpoints(
        tmp_path,
        keep_last=3,
        keep_every_n_iters=5000,
    )

    retained = {
        checkpoint_iteration(path)
        for path in Path(tmp_path).glob("ckpt.iter*.pth")
    }
    assert retained == {5000, 5600, 5800, 6000}
    assert len(removed) == 26


@pytest.mark.parametrize("keep_last", (0, -1))
def test_prune_rejects_invalid_keep_last(tmp_path, keep_last):
    with pytest.raises(ValueError, match="keep_last"):
        prune_checkpoints(tmp_path, keep_last, 0)


def test_formal_r2r_sft_job_preserves_original_global_batch_and_schedule():
    source = (
        Path("scripts/run_rae_r2r_sft_job.sh").read_text(encoding="utf-8")
    )

    required_tokens = (
        "NUM_ENVIRONMENTS 8",
        "IL.batch_size 8",
        "IL.gradient_accumulation_steps 4",
        "IL.iters 30000",
        "IL.lr 1e-5",
        "IL.sample_ratio 0.75",
        "IL.decay_interval 2000",
        "IL.warmup_iters 500",
        "IL.min_lr_ratio 1.0",
        "IL.waypoint_aug True",
        "TASK_CONFIG.DATASET.SUFFIX _90",
        "model_best_step_452500.pt",
    )
    for token in required_tokens:
        assert token in source
