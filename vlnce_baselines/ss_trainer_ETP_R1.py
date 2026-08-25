import gc
import os
import sys
import random
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import numbers
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.runtime_compat import (
    batch_obs_compat as batch_obs,
    get_active_obs_transforms_compat as get_active_obs_transforms,
    get_env_class,
)
from vlnce_baselines.common.amp_utils import step_amp_optimizer
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import (
    GraphMap,
    MAX_DIST,
    heading_from_quaternion,
)
from vlnce_baselines.nwm.rgb_fusion import (
    RaeNwmRgbFusionAdapter,
    apply_rgb_fusion_to_current_candidates,
    clone_wp_outputs_candidate_rgb,
)
from vlnce_baselines.nwm.active_lookahead.base_freeze import (
    capture_base_tensor_manifest,
    compare_base_tensor_manifests,
)
from vlnce_baselines.nwm.active_lookahead.dino_cwp_future import (
    PREDICTED_FUTURE_DIAGNOSTIC_NAMES,
    load_dino_cwp_predictor,
    summarize_predicted_future_diagnostics,
)
from vlnce_baselines.nwm.active_lookahead.joint_e24 import (
    E24JointTrainModule,
    attach_e24_joint_targets,
    build_e24_joint_step,
    collate_e24_joint_packs,
    e24_joint_batch_to_device,
    e24_joint_denominators,
    e24_joint_diagnostic_totals,
    forward_e24_joint_batch,
    joint_action_scale,
    load_e24_joint_head,
    make_e24_joint_dummy_batch,
    normalized_e24_joint_loss,
    slice_e24_joint_batch,
)
from vlnce_baselines.nwm.active_lookahead.offline_objective import (
    OfflineDecisionLossConfig,
)
from vlnce_baselines.nwm.active_lookahead.offline_checkpoint import sha256_file
from vlnce_baselines.nwm.active_lookahead.online_e24 import (
    stop_isolated_e24_actions,
)
from vlnce_baselines.nwm.active_lookahead.residual_head import (
    InterleavedCrossModalTopKFutureLogitResidualHead,
)
from vlnce_baselines.models.checkpoint_utils import (
    navigation_state_dict,
    report_navigation_incompatible_keys,
    validate_rgb_checkpoint_metadata,
)
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from vlnce_baselines.common.online_checkpoint import (
    atomic_torch_save,
    latest_complete_checkpoint_pair,
    latest_checkpoint_path,
    prune_training_states,
)
from vlnce_baselines.common.checkpoint_sync import launch_checkpoint_sync
from torch.nn.utils.rnn import pad_sequence
import cv2
from collections import OrderedDict


def _load_adamw_optimizer_state(
    optimizer,
    optimizer_state,
    use_fused_adamw,
):
    """Load AdamW state while preserving the current implementation mode."""
    saved_param_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_param_groups, list):
        raise ValueError("AdamW optimizer state is missing param_groups")

    use_fused_adamw = bool(use_fused_adamw)
    for param_group in saved_param_groups:
        param_group["fused"] = use_fused_adamw
        if use_fused_adamw:
            param_group["foreach"] = None

    optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["fused"] = use_fused_adamw
        if use_fused_adamw:
            param_group["foreach"] = None

    migrated_tensors = 0
    if use_fused_adamw:
        for parameter, state in optimizer.state.items():
            for name, value in tuple(state.items()):
                if (
                    torch.is_tensor(value)
                    and value.device != parameter.device
                ):
                    state[name] = value.to(device=parameter.device)
                    migrated_tensors += 1
    return migrated_tensors


@baseline_registry.register_trainer(name="SS-ETP-R1")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.illegal_episodes_count = 0
        self.raenwm_runtime = None
        self.raenwm_rgb_fusion_adapter = None
        self.last_raenwm_prediction = None
        self.last_raenwm_rgb_fusion_diagnostics = None
        self._raenwm_head_state_override = None
        self._raenwm_context_source_logged = False
        self.e24_joint_head = None
        self.e24_joint_metadata = None
        self.dino_cwp_future_predictor = None
        self.dino_cwp_future_metadata = None
        self._e24_joint_training = False
        self._e24_joint_replay_packs = []
        self._e24_joint_iteration = 0
        self._e24_joint_start_iteration = 0
        self._e24_future_diagnostic_totals = defaultdict(float)
        self._e24_joint_frozen_manifest = None

    def _active_lookahead_config(self):
        return getattr(getattr(self.config, "MODEL", None), "ACTIVE_LOOKAHEAD", None)

    def _active_lookahead_enabled(self):
        cfg = self._active_lookahead_config()
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    def _e24_joint_head_state_module(self):
        module = self.e24_joint_head
        if module is None:
            return None
        module = getattr(module, "module", module)
        return getattr(module, "head", module)

    def _e24_joint_wrapper_state_module(self):
        module = self.e24_joint_head
        return getattr(module, "module", module) if module is not None else None

    def _e24_joint_train_module(self):
        return self.e24_joint_head

    def _e24_predicted_future_enabled(self):
        return self._active_lookahead_enabled()

    def _initialize_dino_cwp_future_predictor(self):
        if not self._active_lookahead_enabled():
            return None
        cfg = self._active_lookahead_config()
        if str(cfg.source).strip().lower() != "dino_cwp_nwm":
            raise ValueError("ACTIVE_LOOKAHEAD.source must be dino_cwp_nwm")
        if str(cfg.dino_cwp_context_strategy) != "fixed_initial":
            raise ValueError("predicted q1 requires fixed_initial context")
        if str(cfg.dino_cwp_heading_policy) != "face_motion":
            raise ValueError("predicted q1 requires face_motion heading")
        if self.dino_cwp_future_predictor is None:
            model, metadata = load_dino_cwp_predictor(
                cfg.dino_cwp_checkpoint_path,
                expected_sha256=cfg.dino_cwp_checkpoint_sha256,
                device=self.device,
            )
            self.dino_cwp_future_predictor = model
            self.dino_cwp_future_metadata = metadata
        return self.dino_cwp_future_predictor

    def _e24_joint_frozen_modules(self):
        policy_net = getattr(self.policy.net, "module", self.policy.net)
        runtime = self.raenwm_runtime
        predictor = None if runtime is None else runtime.predictor
        bundle = None if predictor is None else predictor.bundle
        return {
            "rae_dino_encoder": getattr(policy_net.rgb_encoder, "rae", None),
            "nwm_body": None if bundle is None else bundle.model,
            "nwm_heads": None if predictor is None else predictor.heads,
            "dino_cwp": self.dino_cwp_future_predictor,
            "waypoint_predictor": self.waypoint_predictor,
        }

    def _validate_e24_joint_provenance(self, checkpoint):
        cfg = self._active_lookahead_config()
        if checkpoint.get("e24_joint_format_version") != str(
            cfg.checkpoint_format_version
        ):
            raise ValueError("joint checkpoint has the wrong format version")
        provenance = checkpoint.get("e24_joint_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("joint checkpoint is missing provenance")
        expected = {
            "base_checkpoint_sha256": str(cfg.base_checkpoint_sha256),
            "e24_init_checkpoint_sha256": str(cfg.e24_joint_init_sha256),
            "dino_cwp_checkpoint_sha256": str(cfg.dino_cwp_checkpoint_sha256),
            "nwm_checkpoint_sha256": str(self.config.MODEL.RAENWM.checkpoint_sha256),
            "nwm_heads_sha256": str(self.config.MODEL.RAENWM.head_checkpoint_sha256),
            "nwm_stat_sha256": str(self.config.MODEL.RAENWM.stat_sha256),
            "source": "dino_cwp_nwm",
            "context_strategy": "fixed_initial",
            "heading_policy": "face_motion",
            "topk": 5,
            "base_iteration": int(cfg.base_iteration),
            "action_warmup_iters": int(cfg.e24_action_warmup_iters),
            "sample_ratio_iteration_offset": int(
                self.config.IL.sample_ratio_iteration_offset
            ),
            "sample_ratio_zero_threshold": float(
                self.config.IL.sample_ratio_zero_threshold
            ),
            "none_threshold": float(cfg.dino_cwp_none_threshold),
            "delta_scale": float(cfg.e24_train_delta_scale),
            "e24_source_base_manifest_sha256": str(
                cfg.e24_source_base_manifest_sha256
            ),
        }
        mismatches = {
            key: (provenance.get(key), value)
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise ValueError(f"joint checkpoint provenance mismatch: {mismatches}")
        return provenance

    def _e24_joint_provenance(self):
        cfg = self._active_lookahead_config()
        return {
            "base_checkpoint_sha256": str(cfg.base_checkpoint_sha256),
            "base_iteration": int(cfg.base_iteration),
            "e24_init_checkpoint_sha256": str(cfg.e24_joint_init_sha256),
            "e24_source_base_manifest_sha256": str(
                cfg.e24_source_base_manifest_sha256
            ),
            "dino_cwp_checkpoint_sha256": str(cfg.dino_cwp_checkpoint_sha256),
            "nwm_checkpoint_sha256": str(self.config.MODEL.RAENWM.checkpoint_sha256),
            "nwm_heads_sha256": str(
                self.config.MODEL.RAENWM.head_checkpoint_sha256
            ),
            "nwm_stat_sha256": str(self.config.MODEL.RAENWM.stat_sha256),
            "source": "dino_cwp_nwm",
            "q0_contract": "temporary_action_same_island_navmesh",
            "q1_contract": "nwm_predicted_no_simulator_query",
            "context_strategy": "fixed_initial",
            "heading_policy": "face_motion",
            "topk": 5,
            "none_threshold": float(cfg.dino_cwp_none_threshold),
            "action_warmup_iters": int(cfg.e24_action_warmup_iters),
            "sample_ratio_iteration_offset": int(
                self.config.IL.sample_ratio_iteration_offset
            ),
            "sample_ratio_zero_threshold": float(
                self.config.IL.sample_ratio_zero_threshold
            ),
            "delta_scale": float(cfg.e24_train_delta_scale),
        }

    def _initialize_e24_joint_head(self, checkpoint=None):
        if not self._active_lookahead_enabled():
            self.e24_joint_head = None
            return None
        cfg = self._active_lookahead_config()
        if int(cfg.offline_topk) != 5:
            raise ValueError("ACTIVE_LOOKAHEAD.offline_topk must be 5")
        joint_state = None if checkpoint is None else checkpoint.get(
            "e24_joint_state_dict"
        )
        if joint_state is not None:
            self._validate_e24_joint_provenance(checkpoint)
            metadata = checkpoint.get("e24_joint_metadata")
            if not isinstance(metadata, dict) or not isinstance(
                metadata.get("model_kwargs"), dict
            ):
                raise ValueError("joint checkpoint lacks E24 model metadata")
            head = InterleavedCrossModalTopKFutureLogitResidualHead(
                **metadata["model_kwargs"]
            ).to(self.device)
            wrapper = E24JointTrainModule(head).to(self.device)
            wrapper.load_state_dict(joint_state, strict=True)
            self.e24_joint_metadata = dict(metadata)
        else:
            head, metadata = load_e24_joint_head(
                cfg.e24_joint_init_path,
                device=self.device,
                expected_checkpoint_sha256=cfg.e24_joint_init_sha256,
                expected_source_base_manifest_sha256=(
                    cfg.e24_source_base_manifest_sha256
                ),
                topk=int(cfg.offline_topk),
            )
            wrapper = E24JointTrainModule(head).to(self.device)
            self.e24_joint_metadata = metadata
        self._e24_joint_training = torch.is_grad_enabled()
        if self._e24_joint_training:
            wrapper.train()
            if self.config.GPU_NUMBERS > 1:
                wrapper = DDP(
                    wrapper,
                    device_ids=[self.device.index],
                    output_device=self.device.index,
                    find_unused_parameters=False,
                    broadcast_buffers=False,
                )
        else:
            wrapper.eval()
            for parameter in wrapper.parameters():
                parameter.requires_grad_(False)
        self.e24_joint_head = wrapper
        self._e24_joint_start_iteration = 0
        return wrapper

    def _raenwm_enabled(self):
        model_config = getattr(self.config, "MODEL", None)
        raenwm_config = getattr(model_config, "RAENWM", None)
        return bool(getattr(raenwm_config, "enabled", False))

    def _raenwm_rgb_fusion_enabled(self):
        return self._raenwm_enabled() and bool(
            getattr(self.config.MODEL.RAENWM, "rgb_fusion_enabled", False)
        )

    def _raenwm_rgb_fusion_trainable(self):
        return self._raenwm_rgb_fusion_enabled() and bool(
            getattr(self.config.MODEL.RAENWM, "rgb_fusion_trainable", False)
        )

    def _initialize_raenwm_rgb_fusion_adapter(self):
        if not self._raenwm_rgb_fusion_enabled():
            self.raenwm_rgb_fusion_adapter = None
            return None
        raenwm_config = self.config.MODEL.RAENWM
        fusion_type = str(
            getattr(raenwm_config, "rgb_fusion_type", "residual_gate")
        ).strip().lower()
        if fusion_type != "residual_gate":
            raise ValueError(
                f"Unsupported MODEL.RAENWM.rgb_fusion_type: {fusion_type}"
            )
        encoder_type = str(self.config.MODEL.RGB_ENCODER.type).lower()
        output_size = int(self.config.MODEL.RGB_ENCODER.output_size)
        if encoder_type != "rae_dinov2" or output_size != 768:
            raise ValueError(
                "RAE-NWM RGB fusion requires MODEL.RGB_ENCODER.type="
                "rae_dinov2 and output_size=768"
            )
        if self.raenwm_rgb_fusion_adapter is None:
            self.raenwm_rgb_fusion_adapter = RaeNwmRgbFusionAdapter(
                input_dim=768,
                hidden_dim=768,
                zero_init=bool(
                    getattr(raenwm_config, "rgb_fusion_zero_init", True)
                ),
                alpha=float(
                    getattr(raenwm_config, "rgb_fusion_alpha", 1.0)
                ),
            ).to(self.device)
        trainable = self._raenwm_rgb_fusion_trainable()
        self.raenwm_rgb_fusion_adapter.train(trainable)
        for parameter in self.raenwm_rgb_fusion_adapter.parameters():
            parameter.requires_grad_(trainable)
        return self.raenwm_rgb_fusion_adapter

    def _broadcast_raenwm_rgb_fusion_state(self):
        adapter = self._raenwm_rgb_fusion_state_module()
        if (
            adapter is None
            or not self._raenwm_rgb_fusion_trainable()
            or int(getattr(self, "world_size", 1)) <= 1
        ):
            return False
        if not distr.is_available() or not distr.is_initialized():
            raise RuntimeError("distributed RGB fusion requires torch.distributed")
        for tensor in list(adapter.parameters()) + list(adapter.buffers()):
            distr.broadcast(tensor.data, src=0)
        return True

    def _synchronize_raenwm_rgb_fusion_gradients(self):
        adapter = self._raenwm_rgb_fusion_state_module()
        world_size = int(getattr(self, "world_size", 1))
        if (
            adapter is None
            or not self._raenwm_rgb_fusion_trainable()
            or world_size <= 1
        ):
            return False
        if not distr.is_available() or not distr.is_initialized():
            raise RuntimeError("distributed RGB fusion requires torch.distributed")
        for parameter in adapter.parameters():
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            distr.all_reduce(parameter.grad, op=distr.ReduceOp.SUM)
            parameter.grad.div_(world_size)
        return True

    def _raenwm_rgb_fusion_state_module(self):
        adapter = getattr(self, "raenwm_rgb_fusion_adapter", None)
        return getattr(adapter, "module", adapter) if adapter is not None else None

    def _raenwm_heads_state_dict(self):
        runtime = getattr(self, "raenwm_runtime", None)
        predictor = getattr(runtime, "predictor", None)
        heads = getattr(predictor, "heads", None)
        if heads is not None:
            return getattr(heads, "module", heads).state_dict()
        return getattr(self, "_raenwm_head_state_override", None)

    def _load_raenwm_rgb_fusion_from_checkpoint(
        self, checkpoint, *, allow_missing
    ):
        adapter = self._raenwm_rgb_fusion_state_module()
        if adapter is None:
            return None
        state_dict = checkpoint.get("raenwm_rgb_fusion_adapter_state_dict")
        if state_dict is None:
            if allow_missing:
                logger.info(
                    "Starting a new single-GPU RGB-fusion training run from "
                    "a checkpoint without fusion weights"
                )
                return None
            raise ValueError(
                "Checkpoint is missing required "
                "raenwm_rgb_fusion_adapter_state_dict"
            )
        return adapter.load_state_dict(state_dict, strict=True)

    def _initialize_raenwm_runtime(self, num_envs):
        if not self._raenwm_enabled():
            self.raenwm_runtime = None
            self.last_raenwm_prediction = None
            return None
        if self.raenwm_runtime is None:
            from vlnce_baselines.nwm.runtime import NwmPredictionRuntime

            self.raenwm_runtime = NwmPredictionRuntime(
                self.config.MODEL.RAENWM,
                self.device,
                head_state_dict_override=self._raenwm_head_state_override,
            )
            self._raenwm_head_state_override = None
        self.raenwm_runtime.reset(num_envs)
        self.last_raenwm_prediction = None
        self._raenwm_context_source_logged = False
        return self.raenwm_runtime

    def _build_raenwm_preview_queries(
        self, cur_pos, cur_ori, candidate_previews
    ):
        from vlnce_baselines.nwm.runtime import NwmQuery

        queries = []
        for env_index, previews in enumerate(candidate_previews):
            grouped_positions = OrderedDict()
            for preview in previews:
                if preview.target_kind not in ("new_ghost", "existing_ghost"):
                    continue
                grouped_positions.setdefault(str(preview.target_vp), []).append(
                    np.asarray(preview.position, dtype=np.float32)
                )
            for ghost_vp, positions in grouped_positions.items():
                queries.append(NwmQuery(
                    env_index=env_index,
                    query_id=ghost_vp,
                    current_position=np.asarray(
                        cur_pos[env_index], dtype=np.float32
                    ),
                    current_yaw=float(
                        heading_from_quaternion(cur_ori[env_index])
                    ),
                    target_position=np.mean(positions, axis=0).astype(np.float32),
                ))
        return queries

    def _run_raenwm_rgb_fusion_prediction(
        self,
        front_latents,
        cur_pos,
        cur_ori,
        candidate_previews,
        wp_outputs,
    ):
        runtime = self.raenwm_runtime
        self.last_raenwm_rgb_fusion_diagnostics = None
        if runtime is None:
            return None
        if front_latents is None:
            raise RuntimeError(
                "RAE-NWM is enabled but waypoint output has no pano_rae_latents"
            )
        yaws = [heading_from_quaternion(value) for value in cur_ori]
        runtime.update_contexts(front_latents, cur_pos, yaws)
        prediction = runtime.predict(
            self._build_raenwm_preview_queries(
                cur_pos, cur_ori, candidate_previews
            )
        )
        self.last_raenwm_prediction = prediction
        self.last_raenwm_rgb_fusion_diagnostics = (
            apply_rgb_fusion_to_current_candidates(
                wp_outputs,
                candidate_previews,
                prediction,
                self.raenwm_rgb_fusion_adapter,
            )
        )
        return prediction

    def _raenwm_rgb_fusion_applied_last_step(self):
        diagnostics = self.last_raenwm_rgb_fusion_diagnostics or []
        return any(
            int((item or {}).get("fused_candidate_count", 0)) > 0
            for item in diagnostics
        )

    def _run_raenwm_prediction(
        self,
        front_latents,
        cur_pos,
        cur_ori,
        cand_vp,
        cand_pos,
    ):
        runtime = self.raenwm_runtime
        if runtime is None:
            return None
        if front_latents is None:
            raise RuntimeError(
                "RAE-NWM is enabled but waypoint output has no pano_rae_latents"
            )
        yaws = [heading_from_quaternion(orientation) for orientation in cur_ori]
        runtime.update_contexts(front_latents, cur_pos, yaws)

        from vlnce_baselines.nwm.runtime import NwmQuery

        queries = []
        for env_index, (env_vp, env_pos) in enumerate(zip(cand_vp, cand_pos)):
            for query_id, target_position in zip(env_vp, env_pos):
                queries.append(
                    NwmQuery(
                        env_index=env_index,
                        query_id=str(query_id),
                        current_position=np.asarray(cur_pos[env_index], dtype=np.float32),
                        current_yaw=float(yaws[env_index]),
                        target_position=np.asarray(target_position, dtype=np.float32),
                    )
                )
        prediction = runtime.predict(queries)
        self.last_raenwm_prediction = prediction
        if not self._raenwm_context_source_logged and not prediction.meta.get("empty", False):
            logger.info(
                "RAE-NWM prediction-only bridge active: records=%d skipped=%s "
                "pred_latent_shape=%s pred_cls_shape=%s",
                len(prediction.meta.get("records", [])),
                dict(prediction.meta.get("skipped", {})),
                tuple(prediction.pred_latent.shape),
                tuple(prediction.pred_cls.shape),
            )
            self._raenwm_context_source_logged = True
        return prediction

    def _record_e24_source_contexts(self, *, stepk, cur_vp):
        if not self._active_lookahead_enabled() or self.raenwm_runtime is None:
            return
        for env_index, (front_vp, gmap) in enumerate(zip(cur_vp, self.gmaps)):
            snapshot = self.raenwm_runtime.source_context_snapshot(
                env_index,
                source_front_vp=str(front_vp),
                source_high_level_step=int(stepk),
            )
            if snapshot is not None:
                gmap.record_raenwm_source_context(snapshot)

    def _create_grad_scaler(self):
        init_scale = self.config.IL.amp_init_scale
        if (
            isinstance(init_scale, bool)
            or not isinstance(init_scale, numbers.Real)
            or not math.isfinite(init_scale)
            or init_scale <= 0
        ):
            raise ValueError(
                "IL.amp_init_scale must be a finite positive number, "
                f"got {init_scale!r}"
            )
        return GradScaler(init_scale=float(init_scale))

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def _launch_checkpoint_sync(self, checkpoint_path):
        enabled = bool(
            getattr(self.config.IL, "checkpoint_sync_enabled", False)
        )
        if not enabled:
            return
        destination = str(
            getattr(self.config.IL, "checkpoint_sync_destination", "")
        ).strip()
        if not destination:
            raise ValueError(
                "IL.checkpoint_sync_destination must be set when "
                "IL.checkpoint_sync_enabled=True"
            )
        try:
            pid = launch_checkpoint_sync(checkpoint_path, destination)
        except Exception:
            logger.exception(
                "Failed to launch asynchronous checkpoint sync for %s",
                checkpoint_path,
            )
            return
        logger.info(
            "Launched asynchronous checkpoint sync: pid=%d source=%s "
            "destination=%s",
            pid,
            checkpoint_path,
            destination,
        )

    def save_checkpoint(
        self,
        iteration: int,
        episode_iterator_state=None,
    ):
        state_dict, rgb_encoder_meta = navigation_state_dict(
            self.policy, self.config
        )
        resumable = bool(
            getattr(self.config.IL, "resumable_checkpoints", False)
        )
        checkpoint = {
            "state_dict": state_dict,
            "rgb_encoder": rgb_encoder_meta,
            "config": self.config,
            "iteration": iteration,
        }
        if self._active_lookahead_enabled():
            e24_wrapper = self._e24_joint_wrapper_state_module()
            if e24_wrapper is None:
                raise RuntimeError("active lookahead has no E24 module to save")
            checkpoint.update(
                {
                    "e24_joint_format_version": str(
                        self._active_lookahead_config().checkpoint_format_version
                    ),
                    "e24_joint_state_dict": e24_wrapper.state_dict(),
                    "e24_joint_metadata": dict(self.e24_joint_metadata or {}),
                    "e24_joint_provenance": self._e24_joint_provenance(),
                }
            )
        fusion_adapter = self._raenwm_rgb_fusion_state_module()
        if fusion_adapter is not None:
            checkpoint["raenwm_rgb_fusion_adapter_state_dict"] = (
                fusion_adapter.state_dict()
            )
        heads_state_dict = self._raenwm_heads_state_dict()
        if heads_state_dict is not None:
            checkpoint["raenwm_heads_state_dict"] = heads_state_dict
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"
        )
        if resumable:
            if episode_iterator_state is None:
                raise ValueError(
                    "Resumable SFT checkpoints require episode iterator state"
                )
            training_state = {
                "format_version": (
                    3 if self._active_lookahead_enabled() else 2
                ),
                "iteration": iteration,
                "model_checkpoint": os.path.basename(checkpoint_path),
                "optim_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "episode_iterator_state": episode_iterator_state,
            }
            training_state_path = os.path.join(
                self.config.CHECKPOINT_FOLDER,
                "train_states",
                f"train_state.iter{iteration}.pth",
            )
            atomic_torch_save(checkpoint, checkpoint_path)
            atomic_torch_save(training_state, training_state_path)
            removed = prune_training_states(
                os.path.dirname(training_state_path),
                int(self.config.IL.keep_last_train_states),
                int(self.config.IL.keep_train_state_every_n_iters),
            )
            if removed:
                logger.info(
                    "Pruned old SFT training states: %s",
                    ", ".join(path.name for path in removed),
                )
            self._launch_checkpoint_sync(checkpoint_path)
            return

        save_training_state = (
            not self.config.ONLY_LAST_SAVEALL
            or iteration == self.config.IL.iters
        )
        if save_training_state:
            checkpoint.update(
                {
                    "optim_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                }
            )
            if hasattr(self, "scaler"):
                checkpoint["scaler_state"] = self.scaler.state_dict()

        torch.save(
            obj=checkpoint,
            f=checkpoint_path,
        )
        self._launch_checkpoint_sync(checkpoint_path)

    def _capture_episode_iterator_state(self):
        # Training rollouts pause workers as their episodes finish. Restore
        # every worker before snapshotting, otherwise interval-boundary
        # checkpoints incorrectly record an empty VectorEnv.
        self.envs.resume_all()
        environment_states = self.envs.call(
            ["get_episode_iterator_state"] * self.envs.num_envs
        )
        local_state = {
            "rank": int(self.local_rank),
            "num_envs": int(self.envs.num_envs),
            "environments": environment_states,
        }
        if self.world_size > 1:
            rank_states = [None for _ in range(self.world_size)]
            distr.all_gather_object(rank_states, local_state)
        else:
            rank_states = [local_state]
        return {
            "format_version": 1,
            "world_size": int(self.world_size),
            "ranks": rank_states,
        }

    def _restore_episode_iterator_state(self, state):
        if not isinstance(state, dict):
            raise TypeError(
                "SFT episode iterator checkpoint state must be a dictionary"
            )
        if state.get("format_version") != 1:
            raise ValueError(
                "Unsupported SFT episode iterator checkpoint format: "
                f"{state.get('format_version')!r}"
            )

        saved_world_size = state.get("world_size")
        if saved_world_size != self.world_size:
            raise ValueError(
                "Cannot preserve SFT episode order after changing the number "
                f"of training ranks: checkpoint={saved_world_size}, "
                f"current={self.world_size}"
            )

        rank_states = state.get("ranks")
        if not isinstance(rank_states, list):
            raise ValueError(
                "SFT episode iterator checkpoint is missing rank states"
            )
        if len(rank_states) != self.world_size:
            raise ValueError(
                "SFT episode iterator checkpoint has the wrong number of "
                f"rank states: checkpoint={len(rank_states)}, "
                f"expected={self.world_size}"
            )
        by_rank = {
            rank_state.get("rank"): rank_state
            for rank_state in rank_states
            if isinstance(rank_state, dict)
        }
        expected_ranks = set(range(self.world_size))
        if set(by_rank) != expected_ranks:
            raise ValueError(
                "SFT episode iterator checkpoint has the wrong ranks: "
                f"checkpoint={sorted(by_rank)}, "
                f"expected={sorted(expected_ranks)}"
            )

        all_ranks_empty = all(
            rank_state.get("num_envs") == 0
            and rank_state.get("environments") == []
            for rank_state in by_rank.values()
        )
        if all_ranks_empty:
            logger.warning(
                "SFT checkpoint contains the legacy empty episode-order "
                "snapshot; model, optimizer, scheduler, scaler, and "
                "iteration are restored, but episode iteration starts from "
                "the newly constructed environment queues"
            )
            return

        local_state = by_rank[self.local_rank]
        environment_states = local_state.get("environments")
        saved_num_envs = local_state.get("num_envs")
        if (
            saved_num_envs != self.envs.num_envs
            or not isinstance(environment_states, list)
            or len(environment_states) != self.envs.num_envs
        ):
            raise ValueError(
                "Cannot preserve SFT episode order after changing the number "
                f"of environments on rank {self.local_rank}: "
                f"checkpoint={saved_num_envs}, current={self.envs.num_envs}"
            )

        self.envs.call(
            ["set_episode_iterator_state"] * self.envs.num_envs,
            [{"state": item} for item in environment_states],
        )
        logger.info(
            "Restored exact SFT episode iterator state for rank %d "
            "across %d environment(s)",
            self.local_rank,
            self.envs.num_envs,
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        print(f"init camera information: resize_config:{resize_config}, crop_config:{crop_config}, new_camera_heading:{camera_orientations}")
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            device_id = int(self.config.TORCH_GPU_IDS[self.local_rank])
            self.device = torch.device("cuda", device_id)
            self.config.defrost()
            self.config.TORCH_GPU_ID = device_id
            self.config.freeze()
            torch.cuda.set_device(device_id)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        allow_missing_fusion_checkpoint: bool = False,
    ):
        start_iter = 0
        ckpt_dict = None
        ckpt_path = None
        training_state = None
        if self._raenwm_enabled():
            config.defrost()
            config.MODEL.RAENWM.emit_patch_latents = True
            config.freeze()
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        logger.info(f"-------------------Load pretrain weight: {config.MODEL.pretrained_path}-------------------")
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict']) 
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self._initialize_raenwm_rgb_fusion_adapter()
        if (
            self._raenwm_rgb_fusion_enabled()
            and not load_from_ckpt
            and not allow_missing_fusion_checkpoint
        ):
            raise ValueError(
                "RGB fusion evaluation/inference requires a checkpoint with "
                "raenwm_rgb_fusion_adapter_state_dict"
            )
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            device_id = self.device.index
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[device_id],
                output_device=device_id, find_unused_parameters=False, broadcast_buffers=False)

        if load_from_ckpt:
            if config.IL.is_requeue:
                if config.IL.resumable_checkpoints:
                    ckpt_path, training_state_path = latest_complete_checkpoint_pair(
                        config.CHECKPOINT_FOLDER
                    )
                    training_state = torch.load(
                        training_state_path, map_location="cpu"
                    )
                else:
                    ckpt_path = latest_checkpoint_path(config.CHECKPOINT_FOLDER)
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            if self._active_lookahead_enabled():
                if ckpt_dict.get("e24_joint_state_dict") is None:
                    actual_base_sha = sha256_file(ckpt_path)
                    if actual_base_sha != str(
                        self._active_lookahead_config().base_checkpoint_sha256
                    ):
                        raise ValueError(
                            "active-lookahead base checkpoint SHA256 mismatch: "
                            f"expected={self._active_lookahead_config().base_checkpoint_sha256} "
                            f"actual={actual_base_sha}"
                        )
                else:
                    self._validate_e24_joint_provenance(ckpt_dict)
            if self._raenwm_enabled() and not self._active_lookahead_enabled():
                self._raenwm_head_state_override = ckpt_dict.get(
                    "raenwm_heads_state_dict"
                )

        if self._active_lookahead_enabled():
            if not self._raenwm_enabled() or not self._raenwm_rgb_fusion_trainable():
                raise ValueError(
                    "active lookahead requires enabled RAE-NWM and trainable RGB fusion"
                )
            self._initialize_dino_cwp_future_predictor()
            if (
                config.IL.is_requeue
                and (
                    ckpt_dict is None
                    or ckpt_dict.get("e24_joint_state_dict") is None
                )
            ):
                raise ValueError(
                    "active-lookahead requeue requires E24 state in the model checkpoint"
                )
            self._initialize_e24_joint_head(ckpt_dict)
        
        param_optimizer = list(self.policy.named_parameters())
        if self._raenwm_rgb_fusion_trainable():
            param_optimizer.extend(
                (f"raenwm_rgb_fusion_adapter.{name}", parameter)
                for name, parameter in self.raenwm_rgb_fusion_adapter.named_parameters()
            )
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01, 'lr': float(self.config.IL.lr),
            'name': 'navigation_decay'},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': float(self.config.IL.lr),
            'name': 'navigation_no_decay'}
        ]
        if self._active_lookahead_enabled() and self._e24_joint_training:
            optimizer_grouped_parameters.append(
                {
                    'params': list(self.e24_joint_head.parameters()),
                    'weight_decay': 0.01,
                    'lr': float(self._active_lookahead_config().e24_head_lr),
                    'name': 'e24',
                }
            )

        use_fused_adamw = bool(
            getattr(self.config.IL, "use_fused_adamw", False)
        )
        if use_fused_adamw and not torch.cuda.is_available():
            raise ValueError("Fused AdamW requires a CUDA training device")
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.IL.lr,
            fused=use_fused_adamw,
        )
        logger.info(
            "AdamW implementation: %s",
            "fused" if use_fused_adamw else "default",
        )
        num_warmup_steps = self.config.IL.warmup_iters
        num_training_steps = self.config.IL.iters
        min_lr_ratio = self.config.IL.min_lr_ratio

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            return decayed_lr_multiplier
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        # self.scheduler.step()

        if load_from_ckpt:
            validate_rgb_checkpoint_metadata(ckpt_dict, config)
            if config.IL.is_requeue:
                if training_state is None:
                    training_state = ckpt_dict
                required_training_state = {
                    "optim_state",
                    "scheduler_state",
                    "scaler_state",
                    "iteration",
                }
                missing_training_state = sorted(
                    required_training_state.difference(training_state)
                )
                if missing_training_state:
                    raise ValueError(
                        "SFT resume training state is incomplete; missing "
                        f"{missing_training_state}: {ckpt_path}"
                    )
                start_iter = training_state["iteration"]
                if ckpt_dict.get("iteration") != start_iter:
                    raise ValueError(
                        "SFT model and training-state iteration mismatch: "
                        f"model={ckpt_dict.get('iteration')} "
                        f"state={start_iter}"
                    )
                if config.IL.resumable_checkpoints:
                    referenced_model = training_state.get(
                        "model_checkpoint"
                    )
                    if referenced_model != os.path.basename(ckpt_path):
                        raise ValueError(
                            "SFT training state references the wrong model: "
                            f"{referenced_model!r} != "
                            f"{os.path.basename(ckpt_path)!r}"
                        )
            else:
                start_iter = 0

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            elif 'module' not in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS > 1:
                new_state_dict = OrderedDict()
                for k, v in ckpt_dict['state_dict'].items():
                    if k.startswith("net."):
                        name = k.replace("net.", "net.module.", 1)
                        new_state_dict[name] = v
                    else:
                        new_state_dict[k] = v
                incompatible_keys = self.policy.load_state_dict(new_state_dict, strict=False)
            else:
                incompatible_keys = self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            self._load_raenwm_rgb_fusion_from_checkpoint(
                ckpt_dict,
                allow_missing=bool(allow_missing_fusion_checkpoint),
            )
            
            if self.local_rank < 1:
                report_navigation_incompatible_keys(
                    incompatible_keys,
                    config,
                )

            if config.IL.is_requeue:
                migrated_optimizer_tensors = _load_adamw_optimizer_state(
                    self.optimizer,
                    training_state["optim_state"],
                    use_fused_adamw,
                )
                logger.info(
                    "Loaded AdamW state with fused=%s; migrated %d state "
                    "tensor(s) to parameter devices",
                    use_fused_adamw,
                    migrated_optimizer_tensors,
                )
                self.scheduler.load_state_dict(
                    training_state["scheduler_state"]
                )
                self.scaler.load_state_dict(
                    training_state["scaler_state"]
                )
                training_state_format = training_state.get(
                    "format_version", 1
                )
                if training_state_format not in (1, 2, 3):
                    raise ValueError(
                        "Unsupported SFT training-state format: "
                        f"{training_state_format!r}"
                    )
                if self._active_lookahead_enabled():
                    if training_state_format != 3:
                        raise ValueError(
                            "active-lookahead requeue requires training-state format 3"
                        )
                    saved_groups = training_state["optim_state"].get(
                        "param_groups", []
                    )
                    current_groups = self.optimizer.param_groups
                    if len(saved_groups) != 3 or len(current_groups) != 3:
                        raise ValueError(
                            "active-lookahead requeue requires exactly three optimizer groups"
                        )
                    if [group.get("name") for group in saved_groups] != [
                        "navigation_decay",
                        "navigation_no_decay",
                        "e24",
                    ]:
                        raise ValueError(
                            "active-lookahead optimizer group names do not match the joint contract"
                        )
                episode_iterator_state = training_state.get(
                    "episode_iterator_state"
                )
                if episode_iterator_state is None:
                    if training_state_format >= 2:
                        raise ValueError(
                            "SFT training state format 2 is incomplete; "
                            "missing episode_iterator_state"
                        )
                    logger.warning(
                        "SFT checkpoint predates episode-order state; model, "
                        "optimizer, scheduler, scaler, and iteration were "
                        "restored, but episode iteration starts from the newly "
                        "constructed environment queues"
                    )
                else:
                    self._restore_episode_iterator_state(
                        episode_iterator_state
                    )
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        self._broadcast_raenwm_rgb_fusion_state()
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(
        self,
        batch_gmap_vp_ids,
        batch_no_vp_left,
        is_train,
        current_goal_distances=None,
    ):
        teacher_actions = []
        cur_episodes = None
        if self.config.IL.expert_policy == 'ndtw':
            cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            if current_goal_distances is None:
                curr_dis_to_goal = self.envs.call_at(
                    i,
                    "current_dist_to_goal",
                    {"is_train": is_train},
                )
            else:
                curr_dis_to_goal = current_goal_distances[i]
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = []
                    ghost_dis_to_goal = []
                    for vp, positions in gmap.ghost_real_pos.items():
                        cached_distances = gmap.ghost_goal_dists.get(vp)
                        if (
                            cached_distances is not None
                            and len(cached_distances) == len(positions)
                            and all(
                                distance is not None
                                for distance in cached_distances
                            )
                        ):
                            chosen_pos, chosen_distance = random.choice(
                                list(zip(positions, cached_distances))
                            )
                        else:
                            chosen_pos = random.choice(positions)
                            chosen_distance = self.envs.call_at(
                                i,
                                "point_dist_to_goal",
                                {"pos": chosen_pos, "is_train": is_train},
                            )
                        ghost_vp_pos.append((vp, chosen_pos))
                        ghost_dis_to_goal.append(chosen_distance)
                    target_ghost_vp = ghost_vp_pos[
                        np.argmin(ghost_dis_to_goal)
                    ][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
        return torch.tensor(
            teacher_actions,
            device=self.device,
            dtype=torch.long,
        )

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []
        
        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool_)
            cand_idxes[obs['cand_img_idxes'][i]] = True

            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])

            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))

        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori, task_type):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        batch_gmap_task_embeddings = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            gmap_step_ids_tensor = torch.LongTensor(gmap_step_ids)
            batch_gmap_step_ids.append(gmap_step_ids_tensor)
            batch_gmap_task_embeddings.append(torch.full_like(gmap_step_ids_tensor, task_type))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_task_embeddings = pad_sequence(batch_gmap_task_embeddings, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left, 'gmap_task_embeddings': batch_gmap_task_embeddings
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]
        return envs, batch

    def train(self):
        self._set_config()
        torch.backends.cudnn.benchmark = bool(
            getattr(self.config.IL, "cudnn_benchmark", False)
        )
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        self.scaler = self._create_grad_scaler()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
            allow_missing_fusion_checkpoint=(
                self._raenwm_rgb_fusion_trainable()
                and not self.config.IL.is_requeue
            ),
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        logger.info('Traning Starts... GOOD LUCK!')

        if self.config.local_rank < 1:
            config_path = os.path.join(self.config.CHECKPOINT_FOLDER, "config.yaml")
            with open(config_path, "w") as f:
                f.write(self.config.dump())
            logger.info(f"Configuration saved to {config_path}")
        
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            schedule_iteration = idx + int(
                self.config.IL.sample_ratio_iteration_offset
            )
            sample_ratio = self.config.IL.sample_ratio ** (
                schedule_iteration // self.config.IL.decay_interval + 1
            )
            if sample_ratio <= float(self.config.IL.sample_ratio_zero_threshold):
                sample_ratio = 0.0
            self._e24_joint_iteration = int(idx)
            logger.info(f"sample ratio: {sample_ratio}")
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            episode_iterator_state = None
            if self.config.IL.resumable_checkpoints:
                episode_iterator_state = (
                    self._capture_episode_iterator_state()
                )

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                current_lr = self.optimizer.param_groups[0]['lr']
                writer.add_scalar('train/lr', current_lr, cur_iter)
                logger.info(loss_str)
                logger.info(f"lr: {current_lr}")
                self.save_checkpoint(
                    cur_iter,
                    episode_iterator_state=episode_iterator_state,
                )
        
    def _e24_joint_loss_config(self):
        return OfflineDecisionLossConfig(
            signed_weight=0.25,
            present_signed_weight=1.0,
            absent_signed_weight=1.0,
            final_weight=1.0,
            pair_weight=0.5,
            regularization_weight=0.001,
            absent_noop_weight=0.05,
            correct_row_weight=2.0,
            wrong_row_weight=1.0,
        )

    def _flush_e24_future_diagnostics(self):
        summary = self._aggregate_e24_future_diagnostics()
        if summary is None:
            return
        for name, value in summary.items():
            self.logs[f"E24_future_{name}"].append(value)

    def _aggregate_e24_future_diagnostics(self):
        if not self._e24_predicted_future_enabled():
            return None
        values = torch.tensor(
            [self._e24_future_diagnostic_totals[name] for name in PREDICTED_FUTURE_DIAGNOSTIC_NAMES],
            dtype=torch.float64,
            device=self.device,
        )
        if self.world_size > 1:
            distr.all_reduce(values, op=distr.ReduceOp.SUM)
        totals = dict(zip(PREDICTED_FUTURE_DIAGNOSTIC_NAMES, values.tolist()))
        return summarize_predicted_future_diagnostics(totals)

    def _backward_e24_joint_replay(self):
        if not getattr(self, "_e24_joint_training", False):
            return
        self._flush_e24_future_diagnostics()
        active_cfg = self._active_lookahead_config()
        batch = collate_e24_joint_packs(self._e24_joint_replay_packs)
        self._e24_joint_replay_packs = []
        loss_config = self._e24_joint_loss_config()
        local_denominators = (
            {name: 0.0 for name in ("signed", "decision_weight", "regularization", "absent_noop")}
            if batch is None
            else e24_joint_denominators(batch, loss_config)
        )
        denominator_names = tuple(local_denominators)
        denominator_tensor = torch.tensor(
            [local_denominators[name] for name in denominator_names],
            dtype=torch.float64,
            device=self.device,
        )
        if self.world_size > 1:
            distr.all_reduce(denominator_tensor, op=distr.ReduceOp.SUM)
        global_denominators = {
            name: float(value)
            for name, value in zip(denominator_names, denominator_tensor.tolist())
        }
        micro_batch = int(active_cfg.e24_replay_micro_batch_size)
        local_rows = 0 if batch is None else int(batch["owner_embeddings"].shape[0])
        local_rounds = int(math.ceil(local_rows / micro_batch)) if local_rows else 0
        rounds_tensor = torch.tensor(local_rounds, dtype=torch.long, device=self.device)
        if self.world_size > 1:
            distr.all_reduce(rounds_tensor, op=distr.ReduceOp.MAX)
        replay_rounds = int(rounds_tensor.item())
        if replay_rounds == 0:
            for name in (
                "loss", "signed_loss", "final_loss", "pair_loss",
                "regularization_loss", "absent_noop_loss", "valid_decisions",
                "teacher_in_top5_rate", "future_valid_rate", "fix", "harm",
                "delta_mean", "delta_abs_mean", "delta_saturation_rate",
                "replay_seconds",
            ):
                self.logs[f"E24_{name}"].append(0.0)
            return

        replay_started = time.perf_counter()
        dummy = make_e24_joint_dummy_batch(topk=int(active_cfg.offline_topk))
        train_module = self._e24_joint_train_module()
        head_dtype = next(train_module.parameters()).dtype
        metric_totals = defaultdict(float)
        for replay_index in range(replay_rounds):
            start = replay_index * micro_batch
            real_micro = batch is not None and start < local_rows
            if real_micro:
                micro = slice_e24_joint_batch(
                    batch, start, min(start + micro_batch, local_rows)
                )
            else:
                micro = dummy
            micro = e24_joint_batch_to_device(
                micro, self.device, dtype=head_dtype
            )
            sync_context = (
                train_module.no_sync()
                if isinstance(train_module, DDP) and replay_index + 1 < replay_rounds
                else nullcontext()
            )
            with sync_context:
                # The retained E24 head uses indexed writes into dense FP32
                # buffers. CUDA autocast can make the indexed source FP16 and
                # violates PyTorch's exact-dtype index_put contract. Keep the
                # small replay microbatch in FP32; navigation still uses AMP
                # and both losses share the same GradScaler/optimizer step.
                with torch.autocast(device_type=self.device.type, enabled=False):
                    result, deltas = forward_e24_joint_batch(
                        train_module, micro, loss_config=loss_config
                    )
                    replay_loss = normalized_e24_joint_loss(
                        result,
                        global_denominators=global_denominators,
                        config=loss_config,
                        world_size=self.world_size,
                        loss_weight=float(active_cfg.e24_loss_weight),
                    )
                self.scaler.scale(replay_loss).backward()
            metric_totals["loss"] += float(replay_loss.detach().cpu())
            signed_count = result.teacher_present_count + result.teacher_absent_count
            metric_totals["signed_numerator"] += (
                float(result.signed_loss.detach().cpu()) * signed_count
            )
            metric_totals["final_numerator"] += (
                float(result.final_loss.detach().cpu()) * result.decision_weight_sum
            )
            metric_totals["pair_numerator"] += (
                float(result.pair_loss.detach().cpu()) * result.decision_weight_sum
            )
            metric_totals["regularization_numerator"] += (
                float(result.regularization_loss.detach().cpu())
                * result.regularization_candidate_count
            )
            metric_totals["absent_noop_numerator"] += (
                float(result.absent_noop_loss.detach().cpu())
                * result.absent_noop_candidate_count
            )
            if real_micro:
                for name, value in e24_joint_diagnostic_totals(
                    micro,
                    deltas,
                    delta_max=float(self._e24_joint_head_state_module().delta_max),
                ).items():
                    metric_totals[name] += value

        diagnostic_names = (
            "teacher_eligible", "teacher_in_top5", "future_slots", "future_valid",
            "fix", "harm", "delta_sum", "delta_abs_sum", "delta_count",
            "delta_saturated",
        )
        diagnostics = torch.tensor(
            [metric_totals[name] for name in diagnostic_names],
            dtype=torch.float64,
            device=self.device,
        )
        if self.world_size > 1:
            distr.all_reduce(diagnostics, op=distr.ReduceOp.SUM)
        global_metrics = dict(zip(diagnostic_names, diagnostics.tolist()))

        loss_names = (
            "loss", "signed_numerator", "final_numerator", "pair_numerator",
            "regularization_numerator", "absent_noop_numerator",
        )
        losses = torch.tensor(
            [metric_totals[name] for name in loss_names],
            dtype=torch.float64,
            device=self.device,
        )
        if self.world_size > 1:
            distr.all_reduce(losses, op=distr.ReduceOp.SUM)
        global_losses = dict(zip(loss_names, losses.tolist()))
        self.logs["E24_loss"].append(global_losses["loss"] / self.world_size)
        for name, denominator_name in (
            ("signed", "signed"),
            ("final", "decision_weight"),
            ("pair", "decision_weight"),
            ("regularization", "regularization"),
            ("absent_noop", "absent_noop"),
        ):
            self.logs[f"E24_{name}_loss"].append(
                global_losses[f"{name}_numerator"]
                / max(1.0, global_denominators[denominator_name])
            )
        self.logs["E24_valid_decisions"].append(global_denominators["signed"])
        self.logs["E24_teacher_in_top5_rate"].append(
            global_metrics["teacher_in_top5"]
            / max(1.0, global_metrics["teacher_eligible"])
        )
        self.logs["E24_future_valid_rate"].append(
            global_metrics["future_valid"] / max(1.0, global_metrics["future_slots"])
        )
        self.logs["E24_fix"].append(global_metrics["fix"])
        self.logs["E24_harm"].append(global_metrics["harm"])
        self.logs["E24_delta_mean"].append(
            global_metrics["delta_sum"] / max(1.0, global_metrics["delta_count"])
        )
        self.logs["E24_delta_abs_mean"].append(
            global_metrics["delta_abs_sum"] / max(1.0, global_metrics["delta_count"])
        )
        self.logs["E24_delta_saturation_rate"].append(
            global_metrics["delta_saturated"]
            / max(1.0, global_metrics["delta_count"])
        )
        self.logs["E24_replay_seconds"].append(time.perf_counter() - replay_started)

    @staticmethod
    def _joint_parameter_grad_norm(parameters):
        squared = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().square().sum()
            squared = value if squared is None else squared + value
        return 0.0 if squared is None else float(squared.sqrt().cpu())


    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        joint_training = getattr(self, "_e24_joint_training", False)
        if joint_training:
            self.e24_joint_head.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        self.sap_loss = 0.
        log_cuda_memory = bool(
            getattr(self.config.IL, "log_cuda_memory", False)
        )
        if log_cuda_memory:
            torch.cuda.reset_peak_memory_stats(self.device)
        accumulation_steps = int(
            getattr(self.config.IL, "gradient_accumulation_steps", 1)
        )
        if accumulation_steps < 1:
            raise ValueError(
                "IL.gradient_accumulation_steps must be at least 1"
            )
        for idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            self._e24_joint_replay_packs = []
            self._e24_future_diagnostic_totals = defaultdict(float)
            for accumulation_idx in range(accumulation_steps):
                should_sync = (
                    self.world_size <= 1
                    or accumulation_idx == accumulation_steps - 1
                )
                sync_context = (
                    nullcontext()
                    if should_sync
                    else self.policy.net.no_sync()
                )
                with sync_context:
                    self.loss = 0.
                    with autocast():
                        self.rollout('train', ml_weight, sample_ratio)
                    self.scaler.scale(
                        self.loss / accumulation_steps
                    ).backward()
            self._backward_e24_joint_replay()
            if joint_training:
                self.scaler.unscale_(self.optimizer)
            self._synchronize_raenwm_rgb_fusion_gradients()
            if joint_training:
                e24_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.e24_joint_head.parameters(),
                    float(
                        self._active_lookahead_config().e24_head_gradient_clip_norm
                    ),
                )
                self.logs["E24_grad_norm"].append(float(e24_grad_norm.cpu()))
            step_amp_optimizer(
                self.scaler,
                self.optimizer,
                self.scheduler,
            )
            if joint_training:
                self._e24_joint_iteration += 1
            if (
                self._active_lookahead_enabled()
                and bool(self._active_lookahead_config().smoke_freeze_check)
                and self._e24_joint_frozen_manifest is not None
            ):
                comparison = compare_base_tensor_manifests(
                    self._e24_joint_frozen_manifest,
                    capture_base_tensor_manifest(self._e24_joint_frozen_modules()),
                )
                if not comparison["exact_match"]:
                    raise RuntimeError(
                        f"frozen active-lookahead tensors changed: {comparison}"
                    )

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
        if log_cuda_memory and self.local_rank < 1:
            mib = 1024 ** 2
            logger.info(
                "CUDA memory MiB: allocated=%.1f reserved=%.1f "
                "peak_allocated=%.1f peak_reserved=%.1f",
                torch.cuda.memory_allocated(self.device) / mib,
                torch.cuda.memory_reserved(self.device) / mib,
                torch.cuda.max_memory_allocated(self.device) / mib,
                torch.cuda.max_memory_reserved(self.device) / mib,
            )
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        # self.config.TASK_CONFIG.TASK.MEASUREMENTS.append('POSITION_INFER')
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            self.config.VIDEO_DIR = self.config.VIDEO_DIR + "_" + self.config.EVAL.SPLIT
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return

        if self.config.EVAL.fast_eval:
            episodes_allowed = self.traj[::5]
        elif self.config.EVAL.EPISODE_ID:
            episodes_allowed = self.config.EVAL.EPISODE_ID
        else:
            episodes_allowed = self.traj
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=episodes_allowed,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=not self.config.EVAL.PRETRAINED_ONLY,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self._e24_future_diagnostic_totals = defaultdict(float)
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        evaluation_started = time.perf_counter()
        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        evaluation_elapsed_seconds = time.perf_counter() - evaluation_started

        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        lookahead_diagnostics = self._aggregate_e24_future_diagnostics()
        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
            )
            with open(fname, "w") as f:
                json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)
                if lookahead_diagnostics is not None:
                    diagnostic_path = os.path.join(
                        self.config.RESULTS_DIR,
                        f"lookahead_ckpt_{checkpoint_index}_{split}.json",
                    )
                    temporary_path = diagnostic_path + ".tmp"
                    payload = {
                        "format_version": "etpr1-active-lookahead-diagnostics-v1",
                        "checkpoint_path": str(checkpoint_path),
                        "checkpoint_index": int(checkpoint_index),
                        "split": str(split),
                        "episodes": int(total),
                        "elapsed_seconds": float(evaluation_elapsed_seconds),
                        "oracle_q1_calls": float(
                            lookahead_diagnostics.get("oracle_q1_requested", 0.0)
                        ),
                        "metrics": lookahead_diagnostics,
                    }
                    with open(temporary_path, "w") as f:
                        json.dump(payload, f, indent=2, sort_keys=True)
                    os.replace(temporary_path, diagnostic_path)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)
            print(f"Episodes evaluated: {total}")

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]

        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            device_id = int(self.config.TORCH_GPU_IDS[self.local_rank])
            self.device = torch.device("cuda", device_id)
            torch.cuda.set_device(device_id)
            self.config.defrost()
            self.config.TORCH_GPU_ID = device_id
            self.config.freeze()
        self.traj = self.collect_infer_traj()
        
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        self.envs.resume_all()
        observations = self.envs.reset()
 
        instr_max_len = self.config.IL.max_text_len
        instr_pad_id = 1
        if self.config.MODEL.task_type == 'r2r':
            task_type = 1
        elif self.config.MODEL.task_type == 'rxr':
            task_type = 2
        else:
            print("self.config.MODEL.task_type Error")
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause) 
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_task_encoding = batch['txt_task_encoding']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_task_encoding=all_txt_task_encoding,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        
        not_done_index = list(range(self.envs.num_envs)) 
        have_real_pos = (
            mode == 'train'
            or bool(self.config.VIDEO_OPTION)
            or self._active_lookahead_enabled()
        )
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost, 
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        self._initialize_raenwm_runtime(self.envs.num_envs)
        if (
            self._active_lookahead_enabled()
            and bool(self._active_lookahead_config().smoke_freeze_check)
            and self._e24_joint_frozen_manifest is None
        ):
            self._e24_joint_frozen_manifest = capture_base_tensor_manifest(
                self._e24_joint_frozen_modules()
            )

        for stepk in range(self.max_len): 
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks
            txt_embeds = all_txt_embeds
            
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug), 
            )
            raenwm_front_latents = None
            if self.raenwm_runtime is not None:
                pano_latents = wp_outputs.pop("pano_rae_latents", None)
                if pano_latents is None:
                    raise RuntimeError(
                        "RAE-NWM is enabled but waypoint output has no "
                        "pano_rae_latents"
                    )
                raenwm_front_latents = pano_latents[:, 0].detach()

            fusion_enabled = self._raenwm_rgb_fusion_enabled()
            if not fusion_enabled:
                # Preserve the prediction-only path's original call order.
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs.update({'mode': 'panorama'})
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                avg_pano_embeds = torch.sum(
                    pano_embeds * pano_masks.unsqueeze(2), 1
                ) / torch.sum(pano_masks, 1, keepdim=True)

            current_goal_distances = None
            cand_goal_dists = None
            candidate_q0_records = None
            if (
                mode == 'train'
                or self.config.VIDEO_OPTION
                or self._active_lookahead_enabled()
            ):
                navigation_states = self.envs.call(
                    ["get_navigation_state"] * self.envs.num_envs,
                    [
                        {
                            "angles": wp_outputs['cand_angles'][i],
                            "forwards": wp_outputs['cand_distances'][i],
                            "include_current_goal_distance": mode == 'train',
                            "include_candidate_goal_distances": (
                                mode == 'train'
                                and self.config.IL.expert_policy == 'spl'
                            ),
                        }
                        for i in range(self.envs.num_envs)
                    ],
                )
                cur_pos = [
                    state["position"] for state in navigation_states
                ]
                cur_ori = [
                    state["orientation"] for state in navigation_states
                ]
                cand_real_pos = [
                    state["candidate_positions"]
                    for state in navigation_states
                ]
                candidate_q0_records = [
                    state["candidate_q0_records"]
                    for state in navigation_states
                ]
                if mode == 'train':
                    current_goal_distances = [
                        state["current_goal_distance"]
                        for state in navigation_states
                    ]
                    if self.config.IL.expert_policy == 'spl':
                        cand_goal_dists = [
                            state["candidate_goal_distances"]
                            for state in navigation_states
                        ]
            else:
                cur_pos, cur_ori = self.get_pos_ori()
                cand_real_pos = [None] * self.envs.num_envs

            cur_vp, cand_vp, cand_pos = [], [], []
            candidate_previews = []
            batch_candidate_to_ghost = []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
                if fusion_enabled:
                    candidate_previews.append(
                        self.gmaps[i].preview_candidate_mapping(
                            cur_vp_i, cur_pos[i], cand_vp_i, cand_pos_i
                        )
                    )

            if fusion_enabled:
                raw_wp_outputs = clone_wp_outputs_candidate_rgb(wp_outputs)
                self._run_raenwm_rgb_fusion_prediction(
                    raenwm_front_latents,
                    cur_pos,
                    cur_ori,
                    candidate_previews,
                    wp_outputs,
                )
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs.update({'mode': 'panorama'})
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                node_pano_embeds, node_pano_masks = pano_embeds, pano_masks
                if self._raenwm_rgb_fusion_applied_last_step():
                    raw_vp_inputs = self._vp_feature_variable(raw_wp_outputs)
                    raw_vp_inputs.update({'mode': 'panorama'})
                    node_pano_embeds, node_pano_masks = self.policy.net(
                        **raw_vp_inputs
                    )
                avg_pano_embeds = torch.sum(
                    node_pano_embeds * node_pano_masks.unsqueeze(2), 1
                ) / torch.sum(node_pano_masks, 1, keepdim=True)
            else:
                self._run_raenwm_prediction(
                    raenwm_front_latents,
                    cur_pos,
                    cur_ori,
                    cand_vp,
                    cand_pos,
                )
            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1] 
                candidate_to_ghost = self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                        cur_vp[i], cur_pos[i], cur_embeds,
                                        cand_vp[i], cand_pos[i], cand_embeds,
                                        cand_real_pos[i],
                                        None if cand_goal_dists is None
                                        else cand_goal_dists[i],
                                        candidate_preview=(
                                            candidate_previews[i]
                                            if fusion_enabled else None
                                        ))
                batch_candidate_to_ghost.append(candidate_to_ghost)

            if self._active_lookahead_enabled():
                if candidate_q0_records is None:
                    raise RuntimeError("active lookahead did not receive candidate q0 records")
                for i, gmap in enumerate(self.gmaps):
                    gmap.record_persistent_q0_candidates(
                        batch_candidate_to_ghost[i],
                        cand_pos[i],
                        cand_real_pos[i],
                        candidate_q0_records[i],
                        wp_outputs['cand_img_idxes'][i],
                        wp_outputs['cand_distances'][i],
                        source_front_vp=str(cur_vp[i]),
                        source_high_level_step=int(stepk),
                    )
                self._record_e24_source_contexts(stepk=stepk, cur_vp=cur_vp)

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori, task_type)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds, 
                'txt_masks': txt_masks, 
            })
            no_vp_left = nav_inputs.pop('no_vp_left') 
            nav_outs = self.policy.net(**nav_inputs)
            nav_logits = nav_outs['global_logits']
            nav_probs = F.softmax(nav_logits, 1)

            active_deltas = None
            e24_joint_pack = None
            if self._active_lookahead_enabled():
                (
                    active_deltas,
                    _active_query_counts,
                    e24_joint_pack,
                    e24_future_diagnostics,
                ) = build_e24_joint_step(
                    self,
                    nav_inputs=nav_inputs,
                    nav_outs=nav_outs,
                    txt_embeds=txt_embeds,
                    txt_masks=txt_masks,
                )
                for name, value in e24_future_diagnostics.items():
                    self._e24_future_diagnostic_totals[name] += float(value)

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(
                    nav_inputs['gmap_vp_ids'],
                    no_vp_left,
                    mode == 'train',
                    current_goal_distances=current_goal_distances,
                )
            if mode == 'train': 
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)
                if e24_joint_pack is not None:
                    e24_joint_pack = attach_e24_joint_targets(
                        e24_joint_pack, teacher_actions, no_vp_left
                    )
                    self._e24_joint_replay_packs.append(
                        e24_joint_pack.to_cpu_fp16()
                    )

            # determine action
            if feedback == 'sample':
                if not self._active_lookahead_enabled():
                    a_t = torch.distributions.Categorical(nav_probs).sample().detach()
                elif active_deltas is None:
                    a_t = nav_logits.argmax(dim=-1)
                else:
                    action_scale = joint_action_scale(
                        self._e24_joint_iteration,
                        self._e24_joint_start_iteration,
                        int(
                            self._active_lookahead_config().e24_action_warmup_iters
                        ),
                    )
                    adjusted_actions = stop_isolated_e24_actions(
                        nav_logits,
                        active_deltas * action_scale,
                        nav_inputs['gmap_vp_ids'],
                    )
                    base_actions = nav_logits.detach().argmax(dim=-1)
                    self._e24_future_diagnostic_totals['action_rows'] += float(
                        adjusted_actions.numel()
                    )
                    self._e24_future_diagnostic_totals['action_flips'] += float(
                        (adjusted_actions != base_actions).sum()
                    )
                    a_t = adjusted_actions
                a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)

            elif feedback == 'argmax':
                a_t = (
                    nav_logits.argmax(dim=-1)
                    if active_deltas is None
                    else stop_isolated_e24_actions(
                        nav_logits,
                        active_deltas,
                        nav_inputs['gmap_vp_ids'],
                    )
                )
                if active_deltas is not None:
                    base_actions = nav_logits.detach().argmax(dim=-1)
                    self._e24_future_diagnostic_totals['action_rows'] += float(
                        a_t.numel()
                    )
                    self._e24_future_diagnostic_totals['action_flips'] += float(
                        (a_t != base_actions).sum()
                    )
            else:
                raise NotImplementedError
            navigation_control = torch.stack(
                (nav_probs[:, 0].float(), a_t.float()),
                dim=1,
            ).detach().cpu().numpy()
            cpu_a_t = navigation_control[:, 1].astype(np.int64)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = float(
                    navigation_control[i, 0]
                )

            # make equiv action
            env_actions = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING) 
            for i, gmap in enumerate(self.gmaps):
                if cpu_a_t[i] == 0 or stepk == self.max_len - 1 or no_vp_left[i]: 
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]

                    if self.config.IL.back_algo == 'control': 
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout
                            },
                            'vis_info': vis_info,
                        }
                    )
                    
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp) 
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float64)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    metric['high_level_step'] = stepk
                    if ep_id in self.stat_eps:
                        print("ERROR!!!!!!!!!! ", ep_id)
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        if self.raenwm_runtime is not None:
                            self.raenwm_runtime.pause_at(i)
                        all_txt_ids = torch.cat((all_txt_ids[:i], all_txt_ids[i + 1:]), dim=0)
                        all_txt_task_encoding = torch.cat((all_txt_task_encoding[:i], all_txt_task_encoding[i + 1:]), dim=0)
                        all_txt_masks = torch.cat((all_txt_masks[:i], all_txt_masks[i + 1:]), dim=0)
                        all_txt_embeds = torch.cat((all_txt_embeds[:i], all_txt_embeds[i + 1:]), dim=0)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID, \
                                                                        max_length=instr_max_len, pad_id=instr_pad_id, task_type=task_type)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train':
            loss = ml_weight * loss / total_actions 
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
