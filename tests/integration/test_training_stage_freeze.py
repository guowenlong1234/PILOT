from copy import deepcopy
import ast
import gzip
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from pretrain_src.pretrain_src.model.vilmodel import (
    ImageEmbeddings as PretrainImageEmbeddings,
)
from vlnce_baselines.GRPO_trainer_ETP_R1 import RLTrainer as GrpoTrainer
from vlnce_baselines.common.amp_utils import step_amp_optimizer
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.models.encoders import rae_dinov2_encoder as encoder_module
from vlnce_baselines.models.encoders.rae_dinov2_encoder import (
    RaeDinov2RgbEncoder,
)
from vlnce_baselines.models.etp.ETP_R1_vilmodel_cmt import (
    ImageEmbeddings as OnlineImageEmbeddings,
)
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer as SftTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_MODEL_DIR = (
    PROJECT_ROOT / "pretrained" / "rae_dinov2_with_registers_base"
)


class _FakeScaler:
    def __init__(self, *, scale_before, scale_after, run_optimizer_step):
        self.scale = scale_before
        self.scale_after = scale_after
        self.run_optimizer_step = run_optimizer_step

    def get_scale(self):
        return self.scale

    def step(self, optimizer):
        if self.run_optimizer_step:
            optimizer.step()

    def update(self):
        self.scale = self.scale_after


class _CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


def test_amp_overflow_skips_optimizer_and_scheduler_together():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = _CountingScheduler()
    scaler = _FakeScaler(
        scale_before=65536.0,
        scale_after=32768.0,
        run_optimizer_step=False,
    )
    parameter.grad = torch.tensor(1.0)

    optimizer_stepped = step_amp_optimizer(scaler, optimizer, scheduler)

    assert not optimizer_stepped
    assert parameter.item() == 1.0
    assert scheduler.step_count == 0


def test_amp_success_steps_optimizer_and_scheduler_once():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = _CountingScheduler()
    scaler = _FakeScaler(
        scale_before=32768.0,
        scale_after=32768.0,
        run_optimizer_step=True,
    )
    parameter.grad = torch.tensor(1.0)

    optimizer_stepped = step_amp_optimizer(scaler, optimizer, scheduler)

    assert optimizer_stepped
    assert torch.isclose(parameter.detach(), torch.tensor(0.9))
    assert scheduler.step_count == 1


def test_sft_amp_init_scale_defaults_and_can_be_overridden():
    from vlnce_baselines.config.default import get_config

    base_config = get_config()
    default_config = get_config("run_r2r/iter_train_rae_dino.yaml")
    override_config = get_config(
        "run_r2r/iter_train_rae_dino.yaml",
        opts=["IL.amp_init_scale", "16384.0"],
    )

    assert base_config.IL.amp_init_scale == 65536.0
    assert default_config.IL.amp_init_scale == 65536.0
    assert override_config.IL.amp_init_scale == 16384.0


def test_sft_grad_scaler_receives_configured_init_scale(monkeypatch):
    calls = []

    def fake_grad_scaler(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        "vlnce_baselines.ss_trainer_ETP_R1.GradScaler",
        fake_grad_scaler,
    )
    trainer = object.__new__(SftTrainer)
    trainer.config = SimpleNamespace(
        IL=SimpleNamespace(amp_init_scale=16384.0)
    )

    scaler = trainer._create_grad_scaler()

    assert scaler is not None
    assert calls == [{"init_scale": 16384.0}]


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("inf"), float("nan")])
def test_sft_grad_scaler_rejects_invalid_init_scale(invalid):
    trainer = object.__new__(SftTrainer)
    trainer.config = SimpleNamespace(
        IL=SimpleNamespace(amp_init_scale=invalid)
    )

    with pytest.raises(ValueError, match="IL.amp_init_scale"):
        trainer._create_grad_scaler()


def _image_config():
    return PretrainedConfig(
        rgb_encoder_type="rae_dinov2",
        image_feat_size=768,
        hidden_size=768,
        depth_feat_size=128,
        angle_feat_size=4,
        hidden_dropout_prob=0.0,
        num_pano_layers=0,
        layer_norm_eps=1e-5,
        use_depth_embedding=True,
        obj_feat_size=0,
    )


class _FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.layernorm = torch.nn.LayerNorm(768)

    def forward(self, pixel_values, **_kwargs):
        cls = self.scale * torch.ones(
            pixel_values.shape[0],
            1,
            768,
            dtype=pixel_values.dtype,
            device=pixel_values.device,
        )
        return SimpleNamespace(last_hidden_state=cls)


def _fake_dino(monkeypatch, tmp_path):
    backbone = _FakeBackbone()
    monkeypatch.setattr(
        encoder_module.Dinov2WithRegistersModel,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: backbone),
    )
    monkeypatch.setattr(
        encoder_module.AutoImageProcessor,
        "from_pretrained",
        staticmethod(
            lambda *_args, **_kwargs: SimpleNamespace(
                image_mean=[0.5, 0.5, 0.5],
                image_std=[0.5, 0.5, 0.5],
            )
        ),
    )
    return RaeDinov2RgbEncoder(
        tmp_path / "fake-model",
        torch.device("cpu"),
        cls_residual_mlp_enabled=True,
        cls_residual_mlp_hidden_dim=768,
        cls_residual_mlp_zero_init=True,
    )


def _assert_finite_nonzero_linear_grad(linear):
    grad = linear.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


def _one_linear_update(linear, features):
    optimizer = torch.optim.SGD(linear.parameters(), lr=1e-3)
    before = linear.weight.detach().clone()
    loss = linear(features).square().mean()
    loss.backward()
    _assert_finite_nonzero_linear_grad(linear)
    optimizer.step()
    assert not torch.equal(linear.weight.detach(), before)


def test_pretrain_and_sft_update_etpnav_visual_layers_but_never_dino(
    monkeypatch, tmp_path
):
    torch.manual_seed(7)
    pretrain = PretrainImageEmbeddings(_image_config())
    _one_linear_update(pretrain.img_linear, torch.randn(2, 768))

    dino = _fake_dino(monkeypatch, tmp_path)
    online = OnlineImageEmbeddings(_image_config())
    dino.eval()
    optimizer = torch.optim.SGD(
        list(dino.cls_residual_mlp.parameters())
        + list(online.img_linear.parameters()),
        lr=1e-3,
    )
    img_linear_before = online.img_linear.weight.detach().clone()
    residual_before = (
        dino.cls_residual_mlp.layers[-1].weight.detach().clone()
    )
    raw_features = dino(
        {"rgb": torch.zeros(2, 224, 224, 3, dtype=torch.uint8)}
    )
    loss = online.img_linear(raw_features).square().mean()
    loss.backward()
    _assert_finite_nonzero_linear_grad(online.img_linear)
    residual_grad = dino.cls_residual_mlp.layers[-1].weight.grad
    assert residual_grad is not None
    assert torch.isfinite(residual_grad).all()
    assert torch.count_nonzero(residual_grad) > 0
    optimizer.step()

    assert not dino.training
    assert not dino.backbone.training
    assert all(
        not parameter.requires_grad for parameter in dino.backbone.parameters()
    )
    assert all(
        parameter.grad is None for parameter in dino.backbone.parameters()
    )
    assert not torch.equal(online.img_linear.weight, img_linear_before)
    assert not torch.equal(
        dino.cls_residual_mlp.layers[-1].weight,
        residual_before,
    )
    assert all(
        parameter.requires_grad
        for parameter in pretrain.img_linear.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in online.img_linear.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in dino.cls_residual_mlp.parameters()
    )


class _FakeVlnBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_embeddings = OnlineImageEmbeddings(_image_config())
        self.global_encoder = torch.nn.Linear(2, 2)
        self.local_encoder = torch.nn.Linear(2, 2)


class _FakeNet(torch.nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.rgb_encoder = dino
        self.vln_bert = _FakeVlnBert()


class _FakePolicy(torch.nn.Module):
    def __init__(self, dino):
        super().__init__()
        self.net = _FakeNet(dino)


def test_grpo_freezes_etpnav_visual_path_and_only_unfreezes_global_module(
    monkeypatch, tmp_path
):
    trainer = object.__new__(GrpoTrainer)
    trainer.policy = _FakePolicy(_fake_dino(monkeypatch, tmp_path))
    global_encoder = trainer.policy.net.vln_bert.global_encoder
    trainer.trainable_parts = [global_encoder]

    trainer.setup_training_parts()

    img_linear = trainer.policy.net.vln_bert.img_embeddings.img_linear
    dino = trainer.policy.net.rgb_encoder
    assert all(not parameter.requires_grad for parameter in img_linear.parameters())
    assert all(not parameter.requires_grad for parameter in dino.parameters())
    assert all(parameter.grad is None for parameter in dino.parameters())
    assert not dino.training
    assert not dino.backbone.training
    assert all(parameter.requires_grad for parameter in global_encoder.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in trainer.policy.net.vln_bert.local_encoder.parameters()
    )


def test_real_dino_asset_probe_is_frozen_and_eval():
    model_path = REAL_MODEL_DIR / "model.safetensors"
    assert model_path.is_file(), f"missing real DINO asset: {model_path}"

    encoder = RaeDinov2RgbEncoder(
        REAL_MODEL_DIR,
        torch.device("cpu"),
        cls_residual_mlp_enabled=True,
        cls_residual_mlp_hidden_dim=768,
        cls_residual_mlp_zero_init=True,
    )
    encoder.train(True)

    assert encoder.training
    assert not encoder.backbone.training
    assert all(
        not parameter.requires_grad
        for parameter in encoder.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in encoder.cls_residual_mlp.parameters()
    )
    assert encoder.backbone.layernorm.elementwise_affine is False
    assert encoder.backbone.layernorm.weight is None
    assert encoder.backbone.layernorm.bias is None
    assert not hasattr(encoder, "latent_mean")
    assert not hasattr(encoder, "latent_var")


def test_sft_trainer_initialization_uses_legacy_output_directories(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    results_dir = tmp_path / "results"
    config = SimpleNamespace(
        local_rank=0,
        TORCH_GPU_ID=0,
        CHECKPOINT_FOLDER=str(checkpoint_dir),
        RESULTS_DIR=str(results_dir),
        EVAL=SimpleNamespace(SAVE_RESULTS=True),
        IL=SimpleNamespace(max_traj_len=1),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    trainer = SftTrainer(config)

    assert trainer.config is config
    assert checkpoint_dir.is_dir()
    assert results_dir.is_dir()


def test_per_episode_eval_stats_are_guarded_by_save_results():
    methods = (
        SftTrainer._eval_checkpoint,
        BaseVLNCETrainer._eval_checkpoint,
    )
    for method in methods:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        guarded = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "SAVE_RESULTS" not in ast.unparse(node.test):
                continue
            constants = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
            }
            if any("stats_ep_ckpt_" in value for value in constants):
                guarded = True
                break
        assert guarded, f"{method.__qualname__} writes episode stats unguarded"


def test_dynamic_depth_camera_keeps_habitat_depth_sensor_type():
    from vlnce_baselines.common.environments import _task_config_for_habitat
    from vlnce_baselines.config.default import get_config

    config = get_config("run_r2r/iter_train_rae_dino.yaml")
    config.defrost()
    depth = deepcopy(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
    depth.UUID = "depth_30"
    config.TASK_CONFIG.SIMULATOR.DEPTH_30 = depth
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_30")
    config.freeze()

    modern = _task_config_for_habitat(config)

    assert (
        modern.simulator.agents.agent_0.sim_sensors.depth_30.type
        == "HabitatSimDepthSensor"
    )


def test_rxr_camera_vectors_are_habitat_sim_compatible_numpy_arrays():
    from vlnce_baselines.common.environments import _task_config_for_habitat
    from vlnce_baselines.config.default import get_config

    config = get_config("run_rxr/iter_train_rae_dino.yaml")

    modern = _task_config_for_habitat(config)

    for sensor_name in ("rgb_sensor", "depth_sensor"):
        sensor = modern.simulator.agents.agent_0.sim_sensors[sensor_name]
        assert isinstance(sensor.position, np.ndarray)
        assert sensor.position.dtype == np.float32
        np.testing.assert_array_equal(
            sensor.position,
            np.asarray([0.0, 0.88, 0.0], dtype=np.float32),
        )
        assert isinstance(sensor.orientation, np.ndarray)
        assert sensor.orientation.dtype == np.float32
        np.testing.assert_array_equal(
            sensor.orientation,
            np.zeros(3, dtype=np.float32),
        )


def test_dynamic_camera_vectors_are_habitat_sim_compatible_numpy_arrays():
    from vlnce_baselines.common.environments import _task_config_for_habitat
    from vlnce_baselines.config.default import get_config

    config = get_config("run_rxr/iter_train_rae_dino.yaml")
    config.defrost()
    depth = deepcopy(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
    depth.UUID = "depth_30"
    depth.POSITION = [0.0, 0.88, 0.0]
    depth.ORIENTATION = [0.0, np.pi / 6.0, 0.0]
    config.TASK_CONFIG.SIMULATOR.DEPTH_30 = depth
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append("DEPTH_30")
    config.freeze()

    modern = _task_config_for_habitat(config)
    sensor = modern.simulator.agents.agent_0.sim_sensors.depth_30

    assert isinstance(sensor.position, np.ndarray)
    assert sensor.position.dtype == np.float32
    assert isinstance(sensor.orientation, np.ndarray)
    assert sensor.orientation.dtype == np.float32
    np.testing.assert_allclose(
        sensor.orientation,
        np.asarray([0.0, np.pi / 6.0, 0.0], dtype=np.float32),
    )


def test_rxr_instruction_sensor_uses_a_valid_placeholder_space():
    from habitat_extensions.sensors import RxRInstructionSensor

    sensor = RxRInstructionSensor(sim=None, config=None)
    episode = SimpleNamespace(
        instruction=SimpleNamespace(
            instruction_text="turn left",
            instruction_tokens=[11, 12, 13],
        ),
        trajectory_id=42,
    )

    assert sensor.uuid == "instruction"
    assert sensor.observation_space.n == 1
    assert sensor.get_observation({}, episode=episode) == {
        "text": "turn left",
        "tokens": [11, 12, 13],
        "trajectory_id": 42,
    }


def test_rxr_eval_trajectory_collection_uses_task_ndtw_gt_path(tmp_path):
    split = "val_unseen"
    gt_template = str(tmp_path / "{split}_{role}_gt.json.gz")
    guide_path = Path(gt_template.format(split=split, role="guide"))
    with gzip.open(guide_path, "wt") as writer:
        json.dump({"episode-a": {"locations": []}}, writer)

    trainer = object.__new__(BaseVLNCETrainer)
    trainer.config = SimpleNamespace(
        BASE_TASK_CONFIG_PATH="run_rxr/rxr_vlnce.yaml",
        local_rank=0,
        GPU_NUMBERS=1,
        TASK_CONFIG=SimpleNamespace(
            DATASET=SimpleNamespace(SPLIT=split, ROLES=["guide"]),
            TASK=SimpleNamespace(
                NDTW=SimpleNamespace(GT_PATH=gt_template),
            ),
        ),
        IL=SimpleNamespace(
            RECOLLECT_TRAINER=SimpleNamespace(
                gt_file=str(tmp_path / "missing" / "{split}_{role}.json.gz"),
            ),
        ),
    )

    trajectories = trainer.collect_val_traj()

    assert trajectories == ["episode-a"]
    assert trainer.gt_data == {"episode-a": {"locations": []}}


def test_unified_rae_smoke_script_covers_all_training_stages():
    script = PROJECT_ROOT / "scripts" / "smoke_rae_dino.sh"
    text = script.read_text()

    assert "set -euo pipefail" in text
    assert "ETPR1_SOURCE_COMMIT" in text
    assert "timeout" in text
    assert "--query-compute-apps=pid" in text
    assert "REFUSE_EXISTING_GPU_COMPUTE_PIDS" in text
    assert "inspect_etpr1_runtime.py" in text
    assert "test_real_mlm_batch" in text
    assert "test_real_sap_batch" in text
    assert "torchrun --nproc_per_node=1" in text
    assert "pretrain_src/pretrain_src/train_r2r.py" in text
    assert "prepare_rae_smoke_pretrain.py" in text
    assert "--num_train_steps 1" in text
    assert "--train_batch_size 1" in text
    assert "--val_batch_size 1" in text
    assert "run_r2r/iter_train_rae_dino.yaml" in text
    assert "--run-type grpo" in text
    assert "run_rxr/iter_train_rae_dino.yaml" in text
    assert text.count("EVAL.EPISODE_COUNT 1") == 2
    assert text.count("IL.iters 1 IL.log_every 1") == 2
    assert text.count("IL.amp_init_scale 16384.0") == 2
    assert text.count("ckpt.iter1.pth") >= 3
    assert "IL.iters 2" not in text
    assert "IL.iters 3" not in text
    assert "audit_rae_smoke.py" in text


def test_r1_env_exposes_original_action_space_for_modern_vector_env():
    from vlnce_baselines.common.environments import VLNCEDaggerEnv

    env = object.__new__(VLNCEDaggerEnv)
    action_space = object()
    env.action_space = action_space

    assert env.original_action_space is action_space


def test_legacy_config_builds_registered_observation_transforms():
    from vlnce_baselines.common.runtime_compat import (
        get_active_obs_transforms_compat,
    )
    from vlnce_baselines.config.default import get_config

    config = get_config("run_r2r/iter_train_rae_dino.yaml")

    transforms = get_active_obs_transforms_compat(config)

    assert [type(transform).__name__ for transform in transforms] == [
        "CenterCropperPerSensor"
    ]


def test_real_etp_implements_modern_net_abstract_properties():
    from vlnce_baselines.models.R1Policy import ETP

    assert not inspect.isabstract(ETP)
    assert "recurrent_hidden_size" not in ETP.__abstractmethods__
    assert "perception_embedding_size" not in ETP.__abstractmethods__


def test_il_policy_is_a_torch_module_in_modern_habitat():
    from vlnce_baselines.models.policy import ILPolicy

    class ConcretePolicy(ILPolicy):
        @classmethod
        def from_config(cls, *args, **kwargs):
            raise NotImplementedError

    net = torch.nn.Linear(2, 2)
    policy = ConcretePolicy(net, dim_actions=3)

    assert isinstance(policy, torch.nn.Module)
    assert policy.to(torch.device("cpu")) is policy
    assert "net.weight" in policy.state_dict()


def test_batch_obs_compat_materializes_autograd_safe_tensors():
    from vlnce_baselines.common.runtime_compat import batch_obs_compat

    batch = batch_obs_compat(
        [{"instruction": np.asarray([1, 2, 3], dtype=np.int64)}],
        torch.device("cpu"),
    )

    assert torch.equal(batch["instruction"], torch.tensor([[1, 2, 3]]))
    assert not torch.is_inference(batch["instruction"])


def test_r1_env_unpacks_modern_vector_env_step_payload():
    from vlnce_baselines.common.environments import VLNCEDaggerEnv

    action = {"act": 4, "ghost_pos": [0.0, 0.0, 0.0]}
    vis_info = {"predict_ghost": [0.0, 0.0, 0.0]}

    unpacked_action, unpacked_vis = VLNCEDaggerEnv._normalize_step_payload(
        {"action": action, "vis_info": vis_info},
        None,
    )
    legacy_action, legacy_vis = VLNCEDaggerEnv._normalize_step_payload(
        action,
        vis_info,
    )

    assert unpacked_action is action
    assert unpacked_vis is vis_info
    assert legacy_action is action
    assert legacy_vis is vis_info
