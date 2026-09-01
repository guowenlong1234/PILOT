from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from vlnce_baselines.nwm.assets import validate_external_asset
from vlnce_baselines.nwm.etp_adapter import (
    NwmEtpAdapter,
    RaeEtpAdapterConfig,
    RaeGhostInputRequest,
    RaeLatentTargetRequest,
)
from vlnce_baselines.nwm.predictor import RaeNwmHeadPredictor, RaeNwmPredictor
from vlnce_baselines.nwm.raenwm_core.models import (
    pack_cls_patch,
    unpack_cls_patch,
)
from vlnce_baselines.nwm.types import NwmPrediction
from vlnce_baselines.nwm.low_level_context import (
    LOW_LEVEL_CONTEXT_SOURCE,
    context_metadata_from_config,
    normalize_context_source,
)


def _load_tensor_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _canonical_stat_tensor(value, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"RAE latent {name} must be a tensor")
    value = value.detach().to(dtype=torch.float32, device="cpu")
    if value.ndim == 1 and value.shape[0] == 768:
        value = value.view(1, 768, 1, 1)
    elif value.ndim == 3 and value.shape[0] == 768:
        value = value.unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 768:
        raise ValueError(
            f"RAE latent {name} must broadcast as [1,768,H,W], got {tuple(value.shape)}"
        )
    if value.shape[0] != 1 or value.shape[2] not in (1, 16) or value.shape[3] not in (1, 16):
        raise ValueError(
            f"RAE latent {name} must broadcast to [N,768,16,16], got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"RAE latent {name} contains NaN or infinity")
    return value.contiguous()


class RaeNwmLatentNormalizer(nn.Module):
    """Apply Stage-1 RAE patch statistics without entering policy checkpoints."""

    def __init__(self, stat_path, eps: float = 1.0e-5):
        super().__init__()
        stats = _load_tensor_dict(stat_path)
        if not isinstance(stats, dict):
            raise ValueError("RAE latent stat file must contain a dict")
        raw_mean = stats.get("mean")
        raw_var = stats.get("var")
        if raw_mean is None and raw_var is None:
            raise ValueError("RAE latent stat file has neither mean nor variance")
        var = (
            torch.ones((1, 768, 1, 1), dtype=torch.float32)
            if raw_var is None
            else _canonical_stat_tensor(raw_var, "var")
        )
        mean = (
            torch.zeros_like(var)
            if raw_mean is None
            else _canonical_stat_tensor(raw_mean, "mean")
        )
        try:
            torch.broadcast_shapes(mean.shape, var.shape, (1, 768, 16, 16))
        except RuntimeError as exc:
            raise ValueError(
                "RAE latent mean and variance do not broadcast to "
                "[N,768,16,16]"
            ) from exc
        if (var < 0).any():
            raise ValueError("RAE latent variance must be non-negative")
        self.eps = float(eps)
        if not np.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("RAE latent normalization eps must be positive")
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("var", var, persistent=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.normalize_patch(latent)

    def normalize_patch(self, latent: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(latent) or latent.ndim != 4:
            shape = tuple(latent.shape) if torch.is_tensor(latent) else None
            raise ValueError(f"RAE patch latent must have shape [N,768,16,16], got {shape}")
        if tuple(latent.shape[1:]) != (768, 16, 16):
            raise ValueError(
                f"RAE patch latent must have shape [N,768,16,16], got {tuple(latent.shape)}"
            )
        latent = latent.float()
        if not torch.isfinite(latent).all():
            raise FloatingPointError("RAE patch latent contains NaN or infinity")
        mean = self.mean.to(device=latent.device)
        var = self.var.to(device=latent.device)
        output = (latent - mean) / torch.sqrt(var + self.eps)
        if not torch.isfinite(output).all():
            raise FloatingPointError("normalized RAE patch latent contains NaN or infinity")
        return output

    def _cls_stats(self, reference: torch.Tensor):
        mean = self.mean.to(device=reference.device)
        var = self.var.to(device=reference.device)
        if mean.ndim == 4:
            mean = mean.mean(dim=(2, 3))
        if var.ndim == 4:
            var = var.mean(dim=(2, 3))
        return mean, var

    @staticmethod
    def _validate_cls(cls: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(cls) or cls.ndim != 2:
            shape = tuple(cls.shape) if torch.is_tensor(cls) else None
            raise ValueError(
                f"RAE CLS latent must have shape [N,768], got {shape}"
            )
        if int(cls.shape[1]) != 768:
            raise ValueError(
                f"RAE CLS latent must have shape [N,768], got {tuple(cls.shape)}"
            )
        cls = cls.float()
        if not torch.isfinite(cls).all():
            raise FloatingPointError("RAE CLS latent contains NaN or infinity")
        return cls

    def normalize_cls(self, cls: torch.Tensor) -> torch.Tensor:
        cls = self._validate_cls(cls)
        mean, var = self._cls_stats(cls)
        output = (cls - mean) / torch.sqrt(var + self.eps)
        if not torch.isfinite(output).all():
            raise FloatingPointError("normalized RAE CLS contains NaN or infinity")
        return output

    def denormalize_cls(self, cls: torch.Tensor) -> torch.Tensor:
        cls = self._validate_cls(cls)
        mean, var = self._cls_stats(cls)
        output = cls * torch.sqrt(var + self.eps) + mean
        if not torch.isfinite(output).all():
            raise FloatingPointError("denormalized RAE CLS contains NaN or infinity")
        return output


def build_native_cls_prediction(
    pred_tokens: torch.Tensor,
    *,
    normalizer: RaeNwmLatentNormalizer,
    pred_rgb=None,
    meta=None,
) -> NwmPrediction:
    """Expose native sequence output without mixing normalized and raw spaces."""

    if not torch.is_tensor(pred_tokens) or pred_tokens.ndim != 3:
        shape = tuple(pred_tokens.shape) if torch.is_tensor(pred_tokens) else None
        raise ValueError(
            f"native NWM output must have shape [N,257,768], got {shape}"
        )
    if tuple(pred_tokens.shape[1:]) != (257, 768):
        raise ValueError(
            "native NWM output must have shape [N,257,768], got "
            f"{tuple(pred_tokens.shape)}"
        )
    if not torch.isfinite(pred_tokens).all():
        raise FloatingPointError("native NWM output contains NaN or infinity")
    pred_cls_normalized, pred_patch = unpack_cls_patch(
        pred_tokens, latent_size=16
    )
    pred_cls_raw = normalizer.denormalize_cls(pred_cls_normalized)
    return NwmPrediction(
        pred_latent=pred_patch,
        pred_tokens=pred_tokens,
        pred_cls_normalized=pred_cls_normalized,
        pred_cls_raw=pred_cls_raw,
        pred_cls=pred_cls_raw,
        pred_rgb=pred_rgb,
        confidence=None,
        conf_logit=None,
        meta=dict(meta or {}),
    )


@dataclass(frozen=True)
class NwmQuery:
    env_index: int
    query_id: str
    current_position: np.ndarray
    current_yaw: float
    target_position: np.ndarray


class NwmPredictionRuntime:
    """Prediction-only online bridge; it never mutates navigation features."""

    def __init__(self, config, device, head_state_dict_override=None):
        self.config = config
        self.device = torch.device(device)
        self.predict_cls_token = bool(
            getattr(config, "predict_cls_token", False)
        )
        self.context_source = normalize_context_source(
            getattr(config, "context_source", None)
        )
        self.context_metadata = context_metadata_from_config(config)
        if (
            self.context_source == LOW_LEVEL_CONTEXT_SOURCE
            and not self.predict_cls_token
        ):
            raise ValueError(
                "low-level movement context currently requires native CLS NWM"
            )
        if int(config.context_size) != 4:
            raise ValueError("Stage-0 RAE-NWM requires context_size=4")
        if int(config.num_steps) != 10:
            raise ValueError("Stage-0 RAE-NWM requires num_steps=10")
        if bool(config.final_only_euler):
            raise ValueError("Stage-0 RAE-NWM requires final_only_euler=False")
        checkpoint_path = validate_external_asset(
            config.checkpoint_path,
            config.checkpoint_sha256,
            "RAE-NWM checkpoint",
        )
        head_checkpoint_path = None
        if self.predict_cls_token:
            if head_state_dict_override is not None:
                raise ValueError(
                    "native CLS NWM must not receive an external head state"
                )
            if str(getattr(config, "head_checkpoint_path", "")).strip():
                raise ValueError(
                    "native CLS NWM must not configure head_checkpoint_path"
                )
        else:
            head_checkpoint_path = validate_external_asset(
                config.head_checkpoint_path,
                config.head_checkpoint_sha256,
                "RAE-NWM head checkpoint",
            )
        stat_path = validate_external_asset(
            config.stat_path,
            config.stat_sha256,
            "RAE patch statistics",
        )
        self.normalizer = RaeNwmLatentNormalizer(stat_path).to(self.device)
        self.adapter = NwmEtpAdapter(
            RaeEtpAdapterConfig(
                context_size=int(config.context_size),
                max_buffer_size=int(config.max_buffer_size),
                image_size=224,
                metric_waypoint_spacing=float(config.metric_waypoint_spacing),
                action_min=(-64.0, -64.0),
                action_max=(64.0, 64.0),
                pos_eps=float(config.pos_eps),
                yaw_eps=float(config.yaw_eps),
                static_run_k=int(config.static_run_k),
                min_horizon=1.0,
                max_horizon=float(config.max_horizon),
            )
        )
        fork_devices = []
        if self.device.type == "cuda":
            fork_devices = [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(int(config.noise_seed))
            predictor_kwargs = dict(
                config_path=config.config_path,
                checkpoint_path=checkpoint_path,
                device=self.device,
                enable_decoder=False,
                torch_compile=False,
                num_steps=int(config.num_steps),
                final_only_euler=bool(config.final_only_euler),
                use_external_context_latents=True,
            )
            if self.predict_cls_token:
                self.predictor = RaeNwmPredictor(**predictor_kwargs)
            else:
                self.predictor = RaeNwmHeadPredictor(
                    **predictor_kwargs,
                    head_checkpoint_path=head_checkpoint_path,
                    strict_heads=True,
                    heads_trainable=False,
                    token_head_trainable=False,
                    confidence_head_trainable=False,
                    head_state_dict_override=head_state_dict_override,
                )
        loaded_native_mode = bool(
            self.predictor.config.get("predict_cls_token", False)
        )
        if loaded_native_mode != self.predict_cls_token:
            raise ValueError(
                "MODEL.RAENWM.predict_cls_token does not match NWM config: "
                f"{self.predict_cls_token} vs {loaded_native_mode}"
            )
        if self.predict_cls_token:
            expected = {
                "context_size": 4,
                "image_size": 224,
                "latent_dim": 768,
                "token_count": 257,
            }
            for name, value in expected.items():
                if int(self.predictor.config.get(name, -1)) != value:
                    raise ValueError(
                        f"native CLS NWM requires {name}={value}"
                    )
            if int(getattr(config, "token_count", -1)) != 257:
                raise ValueError(
                    "MODEL.RAENWM.token_count must be 257 in native CLS mode"
                )
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(int(config.noise_seed))
        self.last_batch = None
        self.last_prediction: Optional[NwmPrediction] = None

    def reset(self, num_envs: int) -> None:
        self.adapter.reset(num_envs)
        self.last_batch = None
        self.last_prediction = None

    def pause_at(self, env_index: int) -> None:
        self.adapter.pause_at(env_index)

    def update_contexts(
        self,
        raw_front_latents,
        positions,
        yaws,
        raw_front_cls=None,
    ) -> None:
        if self.context_source == LOW_LEVEL_CONTEXT_SOURCE:
            raise RuntimeError(
                "low-level movement context forbids high-level update_contexts"
            )
        if len(raw_front_latents) != len(positions) or len(positions) != len(yaws):
            raise ValueError("front latent, position, and yaw counts must match")
        if len(self.adapter.buffers) != len(positions):
            raise RuntimeError("RAE-NWM buffer count does not match active environments")
        normalized_patch = self.normalizer.normalize_patch(raw_front_latents)
        if self.predict_cls_token:
            if raw_front_cls is None:
                raise ValueError("native CLS NWM requires raw front CLS features")
            if len(raw_front_cls) != len(positions):
                raise ValueError("front CLS and position counts must match")
            normalized_cls = self.normalizer.normalize_cls(raw_front_cls)
            normalized = pack_cls_patch(normalized_cls, normalized_patch)
        else:
            if raw_front_cls is not None:
                raise ValueError("patch-only NWM does not accept raw front CLS")
            normalized = normalized_patch
        for env_index in range(len(positions)):
            self.adapter.update_context(
                env_index=env_index,
                rgb=None,
                position=positions[env_index],
                yaw=float(yaws[env_index]),
                latent=normalized[env_index],
            )

    def apply_low_level_context_events(
        self,
        events,
        *,
        raw_patch_latents: torch.Tensor,
        raw_cls: torch.Tensor,
    ) -> None:
        if self.context_source != LOW_LEVEL_CONTEXT_SOURCE:
            raise RuntimeError(
                "low-level context events cannot be applied in high-level mode"
            )
        if not isinstance(events, (list, tuple)):
            raise TypeError("low-level context events must be a sequence")
        frame_events = [
            event for event in events
            if isinstance(event, Mapping) and event.get("type") == "frame"
        ]
        if int(raw_patch_latents.shape[0]) != len(frame_events):
            raise ValueError("low-level patch count does not match frame events")
        if int(raw_cls.shape[0]) != len(frame_events):
            raise ValueError("low-level CLS count does not match frame events")
        normalized_patch = self.normalizer.normalize_patch(raw_patch_latents)
        normalized_cls = self.normalizer.normalize_cls(raw_cls)
        normalized_tokens = pack_cls_patch(normalized_cls, normalized_patch)
        if tuple(normalized_tokens.shape[1:]) != (257, 768):
            raise ValueError(
                "low-level native context must encode as [N,257,768]"
            )
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError("low-level context event must be a mapping")
            event_type = event.get("type")
            env_index = int(event.get("env_index", -1))
            if event_type == "reset":
                self.adapter.clear_at(env_index)
            elif event_type == "frame":
                frame_index = int(event.get("frame_index", -1))
                if frame_index < 0 or frame_index >= len(frame_events):
                    raise ValueError("low-level frame_index is out of range")
                self.adapter.update_context(
                    env_index=env_index,
                    rgb=None,
                    position=event.get("position"),
                    yaw=float(event.get("yaw")),
                    latent=normalized_tokens[frame_index],
                )
            else:
                raise ValueError(
                    f"Unknown low-level context event type: {event_type!r}"
                )

    def source_context_snapshot(
        self,
        env_index: int,
        *,
        source_front_vp: str,
        source_high_level_step: int,
    ):
        """Return a detached CPU copy of the normalized four-frame context."""

        return self.adapter.source_context_snapshot(
            env_index,
            source_front_vp=source_front_vp,
            source_high_level_step=source_high_level_step,
            context_metadata=self.context_metadata,
        )

    def predict_latent_targets(
        self,
        requests: Sequence[RaeLatentTargetRequest],
        *,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> NwmPrediction:
        """Predict q0/q1 from saved contexts through the owned frozen predictor."""

        self.last_batch = self.adapter.build_raenwm_latent_batch(
            requests, device=self.device
        )
        self.last_prediction = self._predict_batch(
            self.last_batch, initial_noise=initial_noise
        )
        return self.last_prediction

    def predict(
        self,
        queries: Sequence[NwmQuery],
        *,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> NwmPrediction:
        requests = [
            RaeGhostInputRequest(
                env_index=int(query.env_index),
                ghost_vp=str(query.query_id),
                current_position=np.asarray(query.current_position, dtype=np.float32),
                current_yaw=float(query.current_yaw),
                ghost_position=np.asarray(query.target_position, dtype=np.float32),
            )
            for query in queries
        ]
        self.last_batch = self.adapter.build_raenwm_batch(requests, device=self.device)
        self.last_prediction = self._predict_batch(
            self.last_batch, initial_noise=initial_noise
        )
        return self.last_prediction

    def _predict_batch(self, batch, *, initial_noise=None) -> NwmPrediction:
        if self.predict_cls_token:
            prediction = self.predictor.predict_time_from_etp_batch(
                batch,
                return_rgb=False,
                generator=self.generator,
                initial_noise=initial_noise,
            )
            if prediction.pred_latent is None:
                return prediction
            return build_native_cls_prediction(
                prediction.pred_latent,
                normalizer=self.normalizer,
                pred_rgb=prediction.pred_rgb,
                meta=prediction.meta,
            )
        return self.predictor.predict_time_with_heads_from_etp_batch(
            batch,
            return_rgb=False,
            generator=self.generator,
            initial_noise=initial_noise,
        )
