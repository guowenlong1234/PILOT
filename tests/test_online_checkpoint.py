from pathlib import Path

import pytest

from vlnce_baselines.common.online_checkpoint import (
    checkpoint_iteration,
    latest_complete_checkpoint_pair,
    latest_checkpoint_path,
    prune_training_states,
    training_state_iteration,
)


def _checkpoint(directory, iteration):
    path = directory / f"ckpt.iter{iteration}.pth"
    path.write_bytes(str(iteration).encode())
    return path


def _training_state(directory, iteration):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"train_state.iter{iteration}.pth"
    path.write_bytes(str(iteration).encode())
    return path


def test_latest_checkpoint_uses_numeric_iteration(tmp_path):
    _checkpoint(tmp_path, 200)
    expected = _checkpoint(tmp_path, 1000)
    _checkpoint(tmp_path, 800)

    assert latest_checkpoint_path(tmp_path) == str(expected)


def test_latest_complete_pair_ignores_unpaired_files(tmp_path):
    state_dir = tmp_path / "train_states"
    expected_model = _checkpoint(tmp_path, 800)
    expected_state = _training_state(state_dir, 800)
    _checkpoint(tmp_path, 1000)
    _training_state(state_dir, 600)

    assert latest_complete_checkpoint_pair(tmp_path) == (
        str(expected_model),
        str(expected_state),
    )


def test_prune_keeps_all_models_but_only_recent_and_milestone_states(tmp_path):
    state_dir = tmp_path / "train_states"
    for iteration in range(200, 6200, 200):
        _checkpoint(tmp_path, iteration)
        _training_state(state_dir, iteration)

    removed = prune_training_states(
        state_dir,
        keep_last=3,
        keep_every_n_iters=5000,
    )

    retained_models = {
        checkpoint_iteration(path) for path in tmp_path.glob("ckpt.iter*.pth")
    }
    retained_states = {
        training_state_iteration(path)
        for path in state_dir.glob("train_state.iter*.pth")
    }
    assert retained_models == set(range(200, 6200, 200))
    assert retained_states == {5000, 5600, 5800, 6000}
    assert len(removed) == 26


@pytest.mark.parametrize("keep_last", (0, -1))
def test_prune_rejects_invalid_keep_last(tmp_path, keep_last):
    with pytest.raises(ValueError, match="keep_last"):
        prune_training_states(tmp_path, keep_last, 0)


def test_formal_r2r_sft_job_preserves_original_global_batch_and_schedule():
    source = (
        Path("scripts/run_rae_r2r_sft_job.sh").read_text(encoding="utf-8")
    )

    required_tokens = (
        "NUM_ENVIRONMENTS 8",
        "IL.batch_size 8",
        "IL.gradient_accumulation_steps 4",
        "IL.keep_last_train_states 3",
        "IL.keep_train_state_every_n_iters 5000",
        "IL.iters 30000",
        "IL.lr 1e-5",
        "IL.sample_ratio 0.75",
        "IL.decay_interval 2000",
        "IL.warmup_iters 500",
        "IL.min_lr_ratio 1.0",
        "IL.waypoint_aug True",
        "IL.use_fused_adamw True",
        "IL.cudnn_benchmark True",
        "IL.log_cuda_memory True",
        "MODEL.RGB_ENCODER.precision ambient",
        "PYTORCH_CUDA_ALLOC_CONF=",
        "TASK_CONFIG.DATASET.SUFFIX _90",
        "model_best_step_452500.pt",
    )
    for token in required_tokens:
        assert token in source


def test_server_r2r_sft_job_uses_requested_batch_and_schedule_defaults():
    source = Path(
        "scripts/run_rae_r2r_sft_server_job.sh"
    ).read_text(encoding="utf-8")

    required_tokens = (
        "SFT_ITERS=${ETPR1_R2R_SFT_ITERS:-2000}",
        "GRADIENT_ACCUMULATION_STEPS=${ETPR1_R2R_SFT_GRADIENT_ACCUMULATION_STEPS:-1}",
        "--nproc_per_node=2",
        "GPU_NUMBERS 2",
        "NUM_ENVIRONMENTS 8",
        "IL.batch_size 8",
        'IL.gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"',
        'IL.iters "$SFT_ITERS"',
        "IL.sample_ratio 0.75",
        "IL.decay_interval 3000",
    )
    for token in required_tokens:
        assert token in source
