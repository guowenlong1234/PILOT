from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from vlnce_baselines.nwm.assets import validate_external_asset
from vlnce_baselines.nwm.etp_adapter import (
    NwmEtpAdapter,
    RaeEtpAdapterConfig,
    RaeGhostInputRequest,
)
from vlnce_baselines.nwm.predictor import RaeNwmHeadPredictor
from vlnce_baselines.nwm.types import NwmPrediction


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
            self.predictor = RaeNwmHeadPredictor(
                config_path=config.config_path,
                checkpoint_path=checkpoint_path,
                device=self.device,
                enable_decoder=False,
                torch_compile=False,
                num_steps=int(config.num_steps),
                final_only_euler=bool(config.final_only_euler),
                use_external_context_latents=True,
                head_checkpoint_path=head_checkpoint_path,
                strict_heads=True,
                heads_trainable=False,
                token_head_trainable=False,
                confidence_head_trainable=False,
                head_state_dict_override=head_state_dict_override,
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

    def update_contexts(self, raw_front_latents, positions, yaws) -> None:
        if len(raw_front_latents) != len(positions) or len(positions) != len(yaws):
            raise ValueError("front latent, position, and yaw counts must match")
        if len(self.adapter.buffers) != len(positions):
            raise RuntimeError("RAE-NWM buffer count does not match active environments")
        normalized = self.normalizer(raw_front_latents)
        for env_index in range(len(positions)):
            self.adapter.update_context(
                env_index=env_index,
                rgb=None,
                position=positions[env_index],
                yaw=float(yaws[env_index]),
                latent=normalized[env_index],
            )

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
        self.last_prediction = self.predictor.predict_time_with_heads_from_etp_batch(
            self.last_batch,
            return_rgb=False,
            generator=self.generator,
            initial_noise=initial_noise,
        )
        return self.last_prediction
