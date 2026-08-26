import importlib.util
from pathlib import Path
import subprocess

import torch
import yaml


MODULE_PATH = (
    Path(__file__).parents[1]
    / "vlnce_baselines/nwm/active_lookahead/grpo_policy.py"
)
SPEC = importlib.util.spec_from_file_location("grpo_policy_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
frozen_lookahead_probs = MODULE.frozen_lookahead_probs
resolve_executed_actions = MODULE.resolve_executed_actions
ROOT = Path(__file__).parents[1]


def test_zero_delta_exactly_recovers_base_distribution():
    logits = torch.tensor([
        [0.4, 1.2, -0.7, -torch.inf],
        [2.0, -1.0, 0.5, 0.25],
    ])
    valid = torch.isfinite(logits)

    actual = frozen_lookahead_probs(logits, torch.zeros_like(logits), valid)

    expected = torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_top5_delta_preserves_stop_mass_and_invalid_actions():
    logits = torch.tensor([[0.5, 0.2, 0.1, -1.0, -torch.inf]])
    delta = torch.tensor([[99.0, 0.0, 3.0, -2.0, 100.0]])
    valid = torch.tensor([[True, True, True, True, False]])

    actual = frozen_lookahead_probs(logits, delta, valid)
    base = torch.softmax(logits.masked_fill(~valid, -torch.inf), dim=-1)

    torch.testing.assert_close(actual[:, 0], base[:, 0])
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1))
    assert actual[0, 4].item() == 0.0
    assert actual[0, 2] > actual[0, 1]


def test_distribution_keeps_gradients_for_current_policy_only():
    logits = torch.tensor([[0.2, 0.5, -0.1]], requires_grad=True)
    delta = torch.tensor([[0.0, 1.0, 0.0]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    probs = frozen_lookahead_probs(logits, delta, valid)
    (-probs[0, 1].log()).backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_resolve_executed_actions_marks_external_forced_stop_invalid():
    sampled = torch.tensor([0, 2, 1, 3])
    no_vp_left = [False, True, False, False]

    executed, policy_valid = resolve_executed_actions(
        sampled,
        no_vp_left=no_vp_left,
        final_step=True,
    )

    assert executed.tolist() == [0, 0, 0, 0]
    assert policy_valid.tolist() == [True, False, False, False]


def test_resolve_executed_actions_is_identity_without_forcing():
    sampled = torch.tensor([0, 2, 1])
    executed, policy_valid = resolve_executed_actions(
        sampled,
        no_vp_left=[False, False, False],
        final_step=False,
    )
    assert torch.equal(executed, sampled)
    assert policy_valid.all()


def test_r2r_and_rxr_frozen_grpo_overlays_are_isolated():
    for dataset, max_traj_len, max_text_len in (
        ("r2r", 15, 150),
        ("rxr", 25, 250),
    ):
        config = yaml.safe_load(
            (ROOT / f"run_{dataset}/grpo_native_cls_e24_frozen.yaml").read_text()
        )
        grpo = config["GRPO"]
        assert config["TRAINER_NAME"] == "GRPO-R1"
        assert grpo["lookahead_distribution_version"] == (
            "etpr1-frozen-lookahead-stop-mass-v1"
        )
        assert grpo["train_rgb_fusion"] is False
        assert grpo["train_top5_e24"] is False
        assert grpo["update_epochs"] == 1
        assert grpo["max_traj_len"] == max_traj_len
        assert grpo["max_text_len"] == max_text_len
        assert grpo["reference_ckpt_to_load"] == ""
        assert grpo["reference_checkpoint_sha256"] == ""


def test_active_grpo_launcher_pins_reference_and_joint_configs():
    launcher = (
        ROOT / "scripts/run_native_cls_e24_grpo_server_job.sh"
    ).read_text()
    manager = (
        ROOT / "scripts/manage_native_cls_e24_grpo_server.sh"
    ).read_text()
    for token in (
        "ETPR1_R2R_ACTIVE_GRPO_SOURCE_CHECKPOINT",
        "ETPR1_RXR_ACTIVE_GRPO_SOURCE_CHECKPOINT",
        "iter_train_rae_dino_native_cls_e24_joint.yaml",
        "grpo_native_cls_e24_frozen.yaml",
        'GRPO.reference_ckpt_to_load "$SOURCE_CHECKPOINT"',
        'GRPO.reference_checkpoint_sha256 "$SOURCE_SHA"',
        "MODEL.RAENWM.rgb_fusion_trainable False",
        'CHECKPOINT_FOLDER "$OUTPUT_ROOT/checkpoints/"',
    ):
        assert token in launcher
    assert "<r2r|rxr>" in manager
    for script in (
        "run_native_cls_e24_grpo_server_job.sh",
        "manage_native_cls_e24_grpo_server.sh",
    ):
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / script)], check=True
        )


def test_grpo_update_uses_adjusted_old_current_and_reference_distributions():
    source = (
        ROOT / "vlnce_baselines/GRPO_trainer_ETP_R1.py"
    ).read_text()
    assert '"frozen_lookahead_delta"' in source
    assert source.count("frozen_lookahead_probs(") >= 3
    assert "policy_action_valid" in source
    assert "_load_active_reference_checkpoint" in source
    assert "self.ref_policy.load_state_dict(\n                    self._reference_state_dict(reference_checkpoint)" in source
    controller = (ROOT / "vlnce_baselines/nwm/frozen_grpo.py").read_text()
    assert "def _e24_joint_train_module" in controller
    assert "return self.e24_joint_head" in controller
