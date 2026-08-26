from typing import List, Optional, Union

import habitat_baselines.config.default
from habitat.config.default import CONFIG_FILE_SEPARATOR
from vlnce_baselines.common.runtime_compat import LegacyConfig as CN

from habitat_extensions.config.default import (
    get_extended_config as get_task_config,
)

# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.BASE_TASK_CONFIG_PATH = "habitat_extensions/config/vlnce_task.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.TRAINER_NAME = "dagger"
_C.ENV_NAME = "VLNCEDaggerEnv"
_C.SIMULATOR_GPU_IDS = [0]
_C.VIDEO_OPTION = []  # options: "disk", "tensorboard"
_C.VIDEO_DIR = "videos/debug"
_C.TENSORBOARD_DIR = "data/tensorboard_dirs/debug"
_C.RESULTS_DIR = "data/checkpoints/pretrained/evals"

# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val_seen"
_C.EVAL.EPISODE_COUNT = -1
_C.EVAL.LANGUAGES = ["en-US", "en-IN"]
_C.EVAL.SAMPLE = False
_C.EVAL.SAVE_RESULTS = True
_C.EVAL.checkpoint_order = "descending"
_C.EVAL.PRETRAINED_ONLY = False
_C.EVAL.EVAL_NONLEARNING = False
_C.EVAL.NONLEARNING = CN()
_C.EVAL.NONLEARNING.AGENT = "RandomAgent"

# -----------------------------------------------------------------------------
# INFERENCE CONFIG
# -----------------------------------------------------------------------------
_C.INFERENCE = CN()
_C.INFERENCE.SPLIT = "test"
_C.INFERENCE.LANGUAGES = ["en-US", "en-IN"]
_C.INFERENCE.SAMPLE = False
_C.INFERENCE.USE_CKPT_CONFIG = True
_C.INFERENCE.CKPT_PATH = "data/checkpoints/CMA_PM_DA_Aug.pth"
_C.INFERENCE.PREDICTIONS_FILE = "predictions.json"
_C.INFERENCE.INFERENCE_NONLEARNING = False
_C.INFERENCE.NONLEARNING = CN()
_C.INFERENCE.NONLEARNING.AGENT = "RandomAgent"
_C.INFERENCE.FORMAT = "rxr"  # either 'rxr' or 'r2r'
# -----------------------------------------------------------------------------
# GRPO LEARNING CONFIG
# -----------------------------------------------------------------------------
_C.GRPO = CN()
_C.GRPO.lr = 1e-5
_C.GRPO.iters = 5000
_C.GRPO.batch_size = 1
_C.GRPO.load_from_ckpt = True
_C.GRPO.is_requeue = False
_C.GRPO.grpo_beta = 0.01
_C.GRPO.grpo_epsilon = 0.2
_C.GRPO.max_grad_norm = 1.0
_C.GRPO.train_all = False
_C.GRPO.resumable_checkpoints = False
_C.GRPO.keep_last_train_states = 3
_C.GRPO.keep_train_state_every_n_iters = 0
_C.GRPO.checkpoint_sync_enabled = False
_C.GRPO.checkpoint_sync_destination = ""
# -----------------------------------------------------------------------------
# IMITATION LEARNING CONFIG
# -----------------------------------------------------------------------------
_C.IL = CN()
_C.IL.lr = 2.5e-4
_C.IL.amp_init_scale = 65536.0
_C.IL.batch_size = 5
_C.IL.epochs = 4
_C.IL.use_iw = True
# inflection coefficient for RxR training set GT trajectories (guide): 1.9
# inflection coefficient for R2R training set GT trajectories: 3.2
_C.IL.inflection_weight_coef = 3.2
# load an already trained model for fine tuning
_C.IL.waypoint_aug = False
_C.IL.load_from_ckpt = False
_C.IL.ckpt_to_load = "data/checkpoints/ckpt.0.pth"
# if True, loads the optimizer state, epoch, and step_id from the ckpt dict.
_C.IL.is_requeue = False
_C.IL.gradient_accumulation_steps = 1
_C.IL.use_fused_adamw = False
_C.IL.cudnn_benchmark = False
_C.IL.log_cuda_memory = False
_C.IL.resumable_checkpoints = False
_C.IL.keep_last_train_states = 3
_C.IL.keep_train_state_every_n_iters = 0
_C.IL.checkpoint_sync_enabled = False
_C.IL.checkpoint_sync_destination = ""
_C.IL.sample_ratio_iteration_offset = 0
_C.IL.sample_ratio_zero_threshold = 0.15
# it True, start training from the saved epoch
# -----------------------------------------------------------------------------
# IL: RXR TRAINER CONFIG
# -----------------------------------------------------------------------------
_C.IL.RECOLLECT_TRAINER = CN()
_C.IL.RECOLLECT_TRAINER.preload_trajectories_file = True
_C.IL.RECOLLECT_TRAINER.trajectories_file = (
    "data/trajectories_dirs/debug/trajectories.json.gz"
)
# if set to a positive int, episodes with longer paths are ignored in training
_C.IL.RECOLLECT_TRAINER.max_traj_len = -1
# if set to a positive int, effective_batch_size must be some multiple of
# IL.batch_size. Gradient accumulation enables an arbitrarily high "effective"
# batch size.
_C.IL.RECOLLECT_TRAINER.effective_batch_size = -1
_C.IL.RECOLLECT_TRAINER.preload_size = 30
_C.IL.RECOLLECT_TRAINER.use_iw = True
_C.IL.RECOLLECT_TRAINER.gt_file = (
    "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
)
# -----------------------------------------------------------------------------
# IL: DAGGER CONFIG
# -----------------------------------------------------------------------------
_C.IL.DAGGER = CN()
_C.IL.DAGGER.iterations = 10
_C.IL.DAGGER.update_size = 5000
_C.IL.DAGGER.p = 0.75
_C.IL.DAGGER.expert_policy_sensor = "SHORTEST_PATH_SENSOR"
_C.IL.DAGGER.expert_policy_sensor_uuid = "shortest_path_sensor"
_C.IL.DAGGER.load_space = False
# if True, load saved observation space and action space
_C.IL.DAGGER.lmdb_map_size = 1.0e12
# if True, saves data to disk in fp16 and converts back to fp32 when loading.
_C.IL.DAGGER.lmdb_fp16 = False
# How often to commit the writes to the DB, less commits is
# better, but everything must be in memory until a commit happens/
_C.IL.DAGGER.lmdb_commit_frequency = 500
# If True, load precomputed features directly from lmdb_features_dir.
_C.IL.DAGGER.preload_lmdb_features = False
_C.IL.DAGGER.lmdb_features_dir = (
    "data/trajectories_dirs/debug/trajectories.lmdb"
)
# -----------------------------------------------------------------------------
# RL CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.POLICY = CN()
_C.RL.POLICY.OBS_TRANSFORMS = CN()
_C.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS = [
    "CenterCropperPerSensor",
]
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = [
    ("rgb", (224, 224)),
    ("depth", (256, 256)),
]
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR = CN()
_C.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = [
    ("rgb", (224, 298)),
    ("depth", (256, 341)),
]
# -----------------------------------------------------------------------------
# MODELING CONFIG
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.policy_name = "CMAPolicy"  # or "Seq2SeqPolicy"
_C.MODEL.ablate_depth = False
_C.MODEL.ablate_rgb = False
_C.MODEL.ablate_instruction = False

_C.MODEL.INSTRUCTION_ENCODER = CN()
_C.MODEL.INSTRUCTION_ENCODER.sensor_uuid = "instruction"
_C.MODEL.INSTRUCTION_ENCODER.vocab_size = 2504
_C.MODEL.INSTRUCTION_ENCODER.use_pretrained_embeddings = True
_C.MODEL.INSTRUCTION_ENCODER.embedding_file = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/embeddings.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.dataset_vocab = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train.json.gz"
)
_C.MODEL.INSTRUCTION_ENCODER.fine_tune_embeddings = False
_C.MODEL.INSTRUCTION_ENCODER.embedding_size = 50
_C.MODEL.INSTRUCTION_ENCODER.hidden_size = 128
_C.MODEL.INSTRUCTION_ENCODER.rnn_type = "LSTM"
_C.MODEL.INSTRUCTION_ENCODER.final_state_only = True
_C.MODEL.INSTRUCTION_ENCODER.bidirectional = False

_C.MODEL.spatial_output = True
_C.MODEL.RGB_ENCODER = CN()
_C.MODEL.RGB_ENCODER.cnn_type = "TorchVisionResNet50"
_C.MODEL.RGB_ENCODER.type = "clip"
_C.MODEL.RGB_ENCODER.precision = "ambient"
_C.MODEL.RGB_ENCODER.model_dir = ""
_C.MODEL.RGB_ENCODER.output_size = 512
_C.MODEL.RGB_ENCODER.cls_residual_mlp_enabled = False
_C.MODEL.RGB_ENCODER.cls_residual_mlp_hidden_dim = 768
_C.MODEL.RGB_ENCODER.cls_residual_mlp_zero_init = True

# Stage-0 RAE-NWM bridge. Prediction-only and optional RGB fusion are disabled
# unless an experiment supplies all external assets and their checksums.
_C.MODEL.RAENWM = CN()
_C.MODEL.RAENWM.enabled = False
_C.MODEL.RAENWM.emit_patch_latents = False
_C.MODEL.RAENWM.predict_cls_token = False
_C.MODEL.RAENWM.config_path = "configs/nwm/raenwm_mp3d.yaml"
_C.MODEL.RAENWM.checkpoint_path = ""
_C.MODEL.RAENWM.checkpoint_sha256 = (
    "392fe02045f7c11f006e1efb822914eee5826b8c6fdb2553c87e7de7c1c4df36"
)
_C.MODEL.RAENWM.head_checkpoint_path = ""
_C.MODEL.RAENWM.head_checkpoint_sha256 = (
    "a4d396021b8bf670c44565c383db1c9288c3bd8988624fa0a59064b6a81dc6e2"
)
_C.MODEL.RAENWM.stat_path = ""
_C.MODEL.RAENWM.stat_sha256 = (
    "84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59"
)
_C.MODEL.RAENWM.context_size = 4
_C.MODEL.RAENWM.max_buffer_size = 64
_C.MODEL.RAENWM.metric_waypoint_spacing = 0.24975892673356762
_C.MODEL.RAENWM.pos_eps = 1.0e-3
_C.MODEL.RAENWM.yaw_eps = 1.0e-3
_C.MODEL.RAENWM.static_run_k = 3
_C.MODEL.RAENWM.max_horizon = 64.0
_C.MODEL.RAENWM.num_steps = 10
_C.MODEL.RAENWM.final_only_euler = False
_C.MODEL.RAENWM.noise_seed = 0
_C.MODEL.RAENWM.rgb_fusion_enabled = False
_C.MODEL.RAENWM.rgb_fusion_type = "residual_gate"
_C.MODEL.RAENWM.rgb_fusion_alpha = 1.0
_C.MODEL.RAENWM.rgb_fusion_zero_init = True
_C.MODEL.RAENWM.rgb_fusion_gate_bias_init = -8.0
_C.MODEL.RAENWM.rgb_fusion_trainable = False

# Predicted Top-5 q1 lookahead with a jointly trained E24 residual scorer.
# Every switch is off by default so legacy R1 behavior is unchanged.
_C.MODEL.ACTIVE_LOOKAHEAD = CN()
_C.MODEL.ACTIVE_LOOKAHEAD.enabled = False
_C.MODEL.ACTIVE_LOOKAHEAD.source = "dino_cwp_nwm"
_C.MODEL.ACTIVE_LOOKAHEAD.offline_topk = 5
_C.MODEL.ACTIVE_LOOKAHEAD.base_checkpoint_path = ""
_C.MODEL.ACTIVE_LOOKAHEAD.base_checkpoint_sha256 = ""
_C.MODEL.ACTIVE_LOOKAHEAD.base_iteration = 14200
_C.MODEL.ACTIVE_LOOKAHEAD.e24_joint_init_path = ""
_C.MODEL.ACTIVE_LOOKAHEAD.e24_joint_init_sha256 = ""
_C.MODEL.ACTIVE_LOOKAHEAD.e24_source_base_manifest_sha256 = ""
_C.MODEL.ACTIVE_LOOKAHEAD.e24_head_lr = 5.0e-6
_C.MODEL.ACTIVE_LOOKAHEAD.e24_loss_weight = 1.0
_C.MODEL.ACTIVE_LOOKAHEAD.e24_train_delta_scale = 1.0
_C.MODEL.ACTIVE_LOOKAHEAD.e24_action_warmup_iters = 400
_C.MODEL.ACTIVE_LOOKAHEAD.e24_replay_micro_batch_size = 2
_C.MODEL.ACTIVE_LOOKAHEAD.e24_replay_storage = "cpu_fp16"
_C.MODEL.ACTIVE_LOOKAHEAD.e24_head_gradient_clip_norm = 10.0
_C.MODEL.ACTIVE_LOOKAHEAD.top5_cls_condition_hidden_dim = 128
_C.MODEL.ACTIVE_LOOKAHEAD.top5_cls_zero_init = True
_C.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_checkpoint_path = ""
_C.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_checkpoint_sha256 = ""
_C.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_none_threshold = 0.3
_C.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_context_strategy = "fixed_initial"
_C.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_heading_policy = "face_motion"
_C.MODEL.ACTIVE_LOOKAHEAD.checkpoint_format_version = "etpr1-e24-joint-v1"
_C.MODEL.ACTIVE_LOOKAHEAD.smoke_freeze_check = False
_C.MODEL.ACTIVE_LOOKAHEAD.diagnostics_enabled = True

_C.MODEL.DEPTH_ENCODER = CN()
_C.MODEL.DEPTH_ENCODER.cnn_type = "VlnResnetDepthEncoder"
_C.MODEL.DEPTH_ENCODER.output_size = 128
# type of resnet to use
_C.MODEL.DEPTH_ENCODER.backbone = "resnet50"
# path to DDPPO resnet weights
_C.MODEL.DEPTH_ENCODER.ddppo_checkpoint = (
    "data/ddppo-models/gibson-2plus-resnet50.pth"
)

_C.MODEL.STATE_ENCODER = CN()
_C.MODEL.STATE_ENCODER.hidden_size = 512
_C.MODEL.STATE_ENCODER.rnn_type = "GRU"

_C.MODEL.SEQ2SEQ = CN()
_C.MODEL.SEQ2SEQ.use_prev_action = False

_C.MODEL.PROGRESS_MONITOR = CN()
_C.MODEL.PROGRESS_MONITOR.use = False
_C.MODEL.PROGRESS_MONITOR.alpha = 1.0  # loss multiplier


def purge_keys(config: CN, keys: List[str]) -> None:
    for k in keys:
        del config[k]
        config.register_deprecated_key(k)


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values. Initialized from the
    habitat_baselines default config. Overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
    """
    config = CN()
    config.merge_from_other_cfg(habitat_baselines.config.default._C)
    purge_keys(config, ["SIMULATOR_GPU_ID", "TEST_EPISODE_COUNT"])
    config.merge_from_other_cfg(_C.clone())
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        prev_task_config = ""
        for config_path in config_paths:
            config.merge_from_file(config_path)
            if config.BASE_TASK_CONFIG_PATH != prev_task_config:
                config.TASK_CONFIG = get_task_config(
                    config.BASE_TASK_CONFIG_PATH
                )
                prev_task_config = config.BASE_TASK_CONFIG_PATH
    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    config.freeze()
    return config
