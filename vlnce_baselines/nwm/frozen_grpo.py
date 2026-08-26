"""Frozen native-CLS RGB injection and Top-5 lookahead for GRPO."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import torch

from vlnce_baselines.models.graph_utils import heading_from_quaternion
from vlnce_baselines.nwm.active_lookahead.base_freeze import (
    capture_base_tensor_manifest,
    compare_base_tensor_manifests,
)
from vlnce_baselines.nwm.active_lookahead.dino_cwp_future import (
    load_dino_cwp_predictor,
)
from vlnce_baselines.nwm.active_lookahead.joint_e24 import (
    E24JointTrainModule,
    build_e24_joint_step,
)
from vlnce_baselines.nwm.active_lookahead.native_cls_adapter import (
    Top5NativeClsAdapter,
)
from vlnce_baselines.nwm.active_lookahead.offline_checkpoint import sha256_file
from vlnce_baselines.nwm.active_lookahead.residual_head import (
    InterleavedCrossModalTopKFutureLogitResidualHead,
)
from vlnce_baselines.nwm.rgb_fusion import (
    RaeNwmRgbFusionAdapter,
    apply_rgb_fusion_to_current_candidates,
)


class FrozenLookaheadController:
    """Own frozen SFT lookahead modules while proxying live GRPO state."""

    def __init__(self, trainer, checkpoint: Mapping):
        self.trainer = trainer
        self.config = trainer.config
        self.device = trainer.device
        self.raenwm_runtime = None
        self.last_prediction = None
        self.last_rgb_diagnostics = None
        self._pending_generator_state = None
        self._frozen_manifest = None

        self._validate_config()
        self._validate_checkpoint(checkpoint)
        self.rgb_fusion_adapter = self._load_rgb_adapter(checkpoint)
        self.e24_joint_head = self._load_e24_wrapper(checkpoint)
        self.dino_cwp_future_predictor, self.dino_cwp_future_metadata = (
            load_dino_cwp_predictor(
                self._active_lookahead_config().dino_cwp_checkpoint_path,
                expected_sha256=(
                    self._active_lookahead_config().dino_cwp_checkpoint_sha256
                ),
                device=self.device,
            )
        )
        self.e24_joint_metadata = dict(checkpoint["e24_joint_metadata"])
        self.e24_joint_provenance = dict(checkpoint["e24_joint_provenance"])
        self.e24_joint_format_version = str(
            checkpoint["e24_joint_format_version"]
        )
        self._freeze_modules()

    @property
    def envs(self):
        return self.trainer.envs

    @property
    def gmaps(self):
        return self.trainer.gmaps

    def _active_lookahead_config(self):
        return self.config.MODEL.ACTIVE_LOOKAHEAD

    def _e24_joint_head_state_module(self):
        return self.e24_joint_head.head

    def _e24_joint_train_module(self):
        return self.e24_joint_head

    def _validate_config(self):
        active = self._active_lookahead_config()
        nwm = self.config.MODEL.RAENWM
        if not bool(active.enabled):
            raise ValueError("frozen GRPO lookahead requires ACTIVE_LOOKAHEAD.enabled")
        if str(active.source).strip().lower() != "dino_cwp_nwm":
            raise ValueError("frozen GRPO lookahead requires dino_cwp_nwm")
        if int(active.offline_topk) != 5:
            raise ValueError("frozen GRPO lookahead requires Top-5")
        if not bool(nwm.enabled) or not bool(nwm.rgb_fusion_enabled):
            raise ValueError("frozen GRPO lookahead requires RGB NWM fusion")
        if not bool(nwm.predict_cls_token) or int(nwm.token_count) != 257:
            raise ValueError("frozen GRPO lookahead requires native 257-token NWM")
        for name in ("train_rgb_fusion", "train_top5_e24"):
            if bool(getattr(self.config.GRPO, name, False)):
                raise ValueError(f"first-stage frozen GRPO requires GRPO.{name}=False")
        if int(self.config.GRPO.update_epochs) != 1:
            raise ValueError(
                "first-stage frozen GRPO requires update_epochs=1 so the "
                "rollout Top-5 context is consumed once"
            )

    def _validate_checkpoint(self, checkpoint: Mapping):
        required = {
            "state_dict",
            "raenwm_rgb_fusion_adapter_state_dict",
            "e24_joint_state_dict",
            "e24_joint_metadata",
            "e24_joint_provenance",
            "e24_joint_format_version",
        }
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(
                f"frozen GRPO source checkpoint is incomplete; missing {missing}"
            )
        active = self._active_lookahead_config()
        if checkpoint["e24_joint_format_version"] != str(
            active.checkpoint_format_version
        ):
            raise ValueError("frozen GRPO source has the wrong joint format")
        provenance = checkpoint["e24_joint_provenance"]
        if not isinstance(provenance, Mapping):
            raise ValueError("frozen GRPO source lacks joint provenance")
        expected = {
            "base_checkpoint_sha256": str(active.base_checkpoint_sha256),
            "e24_init_checkpoint_sha256": str(
                active.e24_joint_init_sha256
            ),
            "e24_source_base_manifest_sha256": str(
                active.e24_source_base_manifest_sha256
            ),
            "nwm_checkpoint_sha256": str(
                self.config.MODEL.RAENWM.checkpoint_sha256
            ),
            "nwm_stat_sha256": str(self.config.MODEL.RAENWM.stat_sha256),
            "dino_cwp_checkpoint_sha256": str(
                active.dino_cwp_checkpoint_sha256
            ),
            "source": "dino_cwp_nwm",
            "context_strategy": "fixed_initial",
            "heading_policy": "face_motion",
            "topk": 5,
            "predict_cls_token": True,
            "token_count": 257,
            "nwm_inference_config_sha256": sha256_file(
                self.config.MODEL.RAENWM.config_path
            ),
            "nwm_num_steps": int(self.config.MODEL.RAENWM.num_steps),
            "none_threshold": float(active.dino_cwp_none_threshold),
            "delta_scale": float(active.e24_train_delta_scale),
        }
        mismatches = {
            key: (provenance.get(key), value)
            for key, value in expected.items()
            if provenance.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"frozen GRPO source provenance mismatch: {mismatches}"
            )
        saved_config = checkpoint.get("config")
        saved_task = str(
            getattr(getattr(saved_config, "MODEL", None), "task_type", "")
        ).lower()
        current_task = str(self.config.MODEL.task_type).lower()
        if saved_task != current_task:
            raise ValueError(
                "frozen GRPO source task mismatch: "
                f"checkpoint={saved_task!r} current={current_task!r}"
            )
        if current_task == "rxr" and provenance.get("task_type") != "rxr":
            raise ValueError("RxR frozen GRPO source lacks RxR provenance")

    def _load_rgb_adapter(self, checkpoint: Mapping):
        nwm = self.config.MODEL.RAENWM
        adapter = RaeNwmRgbFusionAdapter(
            input_dim=768,
            hidden_dim=768,
            zero_init=bool(nwm.rgb_fusion_zero_init),
            alpha=float(nwm.rgb_fusion_alpha),
            gate_bias_init=float(nwm.rgb_fusion_gate_bias_init),
        ).to(self.device)
        adapter.load_state_dict(
            checkpoint["raenwm_rgb_fusion_adapter_state_dict"], strict=True
        )
        return adapter

    def _load_e24_wrapper(self, checkpoint: Mapping):
        metadata = checkpoint["e24_joint_metadata"]
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("model_kwargs"), Mapping
        ):
            raise ValueError("frozen GRPO source lacks E24 model metadata")
        head = InterleavedCrossModalTopKFutureLogitResidualHead(
            **dict(metadata["model_kwargs"])
        ).to(self.device)
        adapter = Top5NativeClsAdapter(
            feature_dim=768,
            condition_hidden_dim=int(
                self._active_lookahead_config().top5_cls_condition_hidden_dim
            ),
            zero_init=bool(
                self._active_lookahead_config().top5_cls_zero_init
            ),
        )
        wrapper = E24JointTrainModule(head, cls_adapter=adapter).to(self.device)
        wrapper.load_state_dict(checkpoint["e24_joint_state_dict"], strict=True)
        return wrapper

    def _freeze_modules(self):
        for module in (
            self.rgb_fusion_adapter,
            self.e24_joint_head,
            self.dino_cwp_future_predictor,
        ):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def initialize_runtime(self, num_envs: int):
        if self.raenwm_runtime is None:
            from vlnce_baselines.nwm.runtime import NwmPredictionRuntime

            self.raenwm_runtime = NwmPredictionRuntime(
                self.config.MODEL.RAENWM,
                self.device,
            )
        self.raenwm_runtime.reset(int(num_envs))
        if self._pending_generator_state is not None:
            self.raenwm_runtime.generator.set_state(
                self._pending_generator_state
            )
            self._pending_generator_state = None
        self.last_prediction = None
        self.last_rgb_diagnostics = None
        return self.raenwm_runtime

    def set_pending_generator_state(self, state):
        self._pending_generator_state = state

    def generator_state(self):
        if self.raenwm_runtime is None:
            return self._pending_generator_state
        return self.raenwm_runtime.generator.get_state().cpu()

    def _build_preview_queries(self, cur_pos, cur_ori, candidate_previews):
        from vlnce_baselines.nwm.runtime import NwmQuery

        queries = []
        for env_index, previews in enumerate(candidate_previews):
            grouped = OrderedDict()
            for preview in previews:
                if preview.target_kind not in ("new_ghost", "existing_ghost"):
                    continue
                grouped.setdefault(str(preview.target_vp), []).append(
                    np.asarray(preview.position, dtype=np.float32)
                )
            for ghost_vp, positions in grouped.items():
                queries.append(NwmQuery(
                    env_index=env_index,
                    query_id=ghost_vp,
                    current_position=np.asarray(cur_pos[env_index], dtype=np.float32),
                    current_yaw=float(heading_from_quaternion(cur_ori[env_index])),
                    target_position=np.mean(positions, axis=0).astype(np.float32),
                ))
        return queries

    def inject_rgb(
        self,
        front_latents,
        front_cls,
        cur_pos,
        cur_ori,
        candidate_previews,
        wp_outputs,
    ):
        if self.raenwm_runtime is None:
            raise RuntimeError("frozen GRPO NWM runtime is not initialized")
        if front_latents is None or front_cls is None:
            raise RuntimeError("frozen GRPO native NWM requires front CLS and patch")
        yaws = [heading_from_quaternion(value) for value in cur_ori]
        self.raenwm_runtime.update_contexts(
            front_latents,
            cur_pos,
            yaws,
            raw_front_cls=front_cls,
        )
        prediction = self.raenwm_runtime.predict(
            self._build_preview_queries(cur_pos, cur_ori, candidate_previews)
        )
        self.last_prediction = prediction
        self.last_rgb_diagnostics = apply_rgb_fusion_to_current_candidates(
            wp_outputs,
            candidate_previews,
            prediction,
            self.rgb_fusion_adapter,
        )
        return prediction

    def rgb_was_applied(self):
        return any(
            int((item or {}).get("fused_candidate_count", 0)) > 0
            for item in (self.last_rgb_diagnostics or [])
        )

    def record_source_contexts(self, *, stepk, cur_vp):
        for env_index, (front_vp, gmap) in enumerate(zip(cur_vp, self.gmaps)):
            snapshot = self.raenwm_runtime.source_context_snapshot(
                env_index,
                source_front_vp=str(front_vp),
                source_high_level_step=int(stepk),
            )
            if snapshot is not None:
                gmap.record_raenwm_source_context(snapshot)

    def build_frozen_deltas(self, *, nav_inputs, nav_outs, txt_embeds, txt_masks):
        delta, query_counts, _pack, diagnostics = build_e24_joint_step(
            self,
            nav_inputs=nav_inputs,
            nav_outs=nav_outs,
            txt_embeds=txt_embeds,
            txt_masks=txt_masks,
        )
        return delta.detach(), query_counts, diagnostics

    def pause_at(self, index: int):
        if self.raenwm_runtime is not None:
            self.raenwm_runtime.pause_at(int(index))

    def checkpoint_payload(self):
        return {
            "e24_joint_format_version": self.e24_joint_format_version,
            "e24_joint_state_dict": self.e24_joint_head.state_dict(),
            "e24_joint_metadata": dict(self.e24_joint_metadata),
            "e24_joint_provenance": dict(self.e24_joint_provenance),
            "raenwm_rgb_fusion_adapter_state_dict": (
                self.rgb_fusion_adapter.state_dict()
            ),
        }

    def capture_frozen_manifest(self):
        runtime = self.raenwm_runtime
        predictor = None if runtime is None else runtime.predictor
        bundle = None if predictor is None else predictor.bundle
        self._frozen_manifest = capture_base_tensor_manifest({
            "rgb_fusion": self.rgb_fusion_adapter,
            "top5_e24": self.e24_joint_head,
            "dino_cwp": self.dino_cwp_future_predictor,
            "nwm_body": None if bundle is None else bundle.model,
        })
        return self._frozen_manifest

    def assert_frozen(self):
        for name, module in (
            ("rgb_fusion", self.rgb_fusion_adapter),
            ("top5_e24", self.e24_joint_head),
            ("dino_cwp", self.dino_cwp_future_predictor),
        ):
            for parameter in module.parameters():
                if parameter.requires_grad or parameter.grad is not None:
                    raise RuntimeError(f"frozen GRPO module changed gradient state: {name}")
        if self._frozen_manifest is not None:
            current = capture_base_tensor_manifest({
                "rgb_fusion": self.rgb_fusion_adapter,
                "top5_e24": self.e24_joint_head,
                "dino_cwp": self.dino_cwp_future_predictor,
                "nwm_body": self.raenwm_runtime.predictor.bundle.model,
            })
            comparison = compare_base_tensor_manifests(
                self._frozen_manifest, current
            )
            if not comparison["exact_match"]:
                raise RuntimeError(
                    f"frozen GRPO lookahead tensors changed: {comparison}"
                )
        return True
