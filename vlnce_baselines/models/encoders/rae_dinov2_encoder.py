from contextlib import nullcontext
from collections.abc import Mapping
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


_COMPUTE_DTYPES = {
    "ambient": None,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
}


def resolve_rae_compute_dtype(precision: str) -> Optional[torch.dtype]:
    normalized = str(precision).lower()
    if normalized not in _COMPUTE_DTYPES:
        supported = ", ".join(sorted(_COMPUTE_DTYPES))
        raise ValueError(
            f"Unsupported RAE/DINOv2 precision {precision!r}; "
            f"expected one of: {supported}"
        )
    return _COMPUTE_DTYPES[normalized]


def _as_batched_chw_rgb_tensor(rgb_observations) -> torch.Tensor:
    """Match ETPNav's navigation-side RGB layout conversion."""
    if isinstance(rgb_observations, Image.Image):
        rgb_observations = np.asarray(
            rgb_observations.convert("RGB")
        ).copy()
    if isinstance(rgb_observations, np.ndarray):
        if not rgb_observations.flags.writeable:
            rgb_observations = rgb_observations.copy()
        rgb_observations = torch.from_numpy(rgb_observations)
    if not torch.is_tensor(rgb_observations):
        raise TypeError(
            "rgb_observations must be a tensor, numpy array, or PIL image, "
            f"got {type(rgb_observations).__name__}"
        )

    if rgb_observations.ndim == 3:
        if (
            rgb_observations.shape[0] in (1, 3, 4)
            and rgb_observations.shape[-1] not in (1, 3, 4)
        ):
            rgb_observations = rgb_observations.unsqueeze(0)
        else:
            rgb_observations = rgb_observations.unsqueeze(0).permute(
                0, 3, 1, 2
            )
    elif rgb_observations.ndim == 4:
        if (
            rgb_observations.shape[-1] in (1, 3, 4)
            and rgb_observations.shape[1] not in (1, 3)
        ):
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
        elif rgb_observations.shape[1] in (1, 3, 4):
            pass
        elif rgb_observations.shape[-1] in (1, 3, 4):
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "RGB tensor must have a channel dimension of 1, 3, or 4, "
                f"got {tuple(rgb_observations.shape)}"
            )
    else:
        raise ValueError(
            f"RGB tensor must be 3D or 4D, got {tuple(rgb_observations.shape)}"
        )

    if rgb_observations.shape[1] == 4:
        rgb_observations = rgb_observations[:, :3]
    elif rgb_observations.shape[1] == 1:
        rgb_observations = rgb_observations.repeat(1, 3, 1, 1)
    elif rgb_observations.shape[1] != 3:
        raise ValueError(
            "RGB tensor must have 3 channels after conversion, "
            f"got {tuple(rgb_observations.shape)}"
        )

    return rgb_observations.contiguous()


def normalize_for_rae_input(rgb_observations: torch.Tensor) -> torch.Tensor:
    """Convert common RGB ranges to ETPNav's pre-encoder [0, 1] range."""
    is_floating_point = torch.is_floating_point(rgb_observations)
    rgb_observations = rgb_observations.to(dtype=torch.float32)
    if rgb_observations.numel() > 0:
        if not is_floating_point:
            rgb_observations = rgb_observations / 255.0
        else:
            detached = rgb_observations.detach()
            max_value = detached.max()
            min_value = detached.min()
            if max_value > 2.0:
                rgb_observations = rgb_observations / 255.0
            elif min_value < 0.0:
                rgb_observations = (rgb_observations + 1.0) * 0.5
    return rgb_observations.clamp(0.0, 1.0)


def _resize_for_rae(
    rgb_observations: torch.Tensor,
    size: int,
) -> torch.Tensor:
    if rgb_observations.shape[-2:] == (size, size):
        return rgb_observations
    try:
        return F.interpolate(
            rgb_observations,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    except TypeError:
        return F.interpolate(
            rgb_observations,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )


def prepare_rae_rgb_tensor(
    rgb_observations,
    device,
    size: int = 224,
) -> torch.Tensor:
    rgb_observations = _as_batched_chw_rgb_tensor(rgb_observations)
    rgb_observations = rgb_observations.to(device=device)
    rgb_observations = normalize_for_rae_input(rgb_observations)
    return _resize_for_rae(rgb_observations, int(size))


class ClsResidualMlp(nn.Module):
    """ETPNav's zero-initialized residual adapter for navigation CLS."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        zero_init: bool = True,
    ):
        super().__init__()
        input_dim = int(input_dim)
        hidden_dim = int(
            hidden_dim if hidden_dim is not None else input_dim
        )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        if zero_init:
            last_layer = self.layers[-1]
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, cls_features: torch.Tensor) -> torch.Tensor:
        return self.layers(cls_features)


class RaeDinov2RgbEncoder(nn.Module):
    """Frozen DINOv2 backbone plus ETPNav's trainable CLS residual MLP.

    RAE's spatial ``stat.pt`` statistics intentionally do not participate in
    this navigation CLS path. They normalize patch latents in ETPNav, while
    ``encode_with_cls`` returns the original 768-dimensional CLS token.
    """

    output_size = 768

    def __init__(
        self,
        model_dir,
        device,
        precision: str = "ambient",
        cls_residual_mlp_enabled: bool = False,
        cls_residual_mlp_hidden_dim: Optional[int] = None,
        cls_residual_mlp_zero_init: bool = True,
        compile_backbone: bool = False,
        async_finite_checks: bool = False,
        retain_intermediate_states: bool = True,
    ):
        super().__init__()
        self.precision = str(precision).lower()
        self.compute_dtype = resolve_rae_compute_dtype(self.precision)
        self.encoder_input_size = 224
        self.async_finite_checks = bool(async_finite_checks)
        self.retain_intermediate_states = bool(retain_intermediate_states)

        self.backbone = Dinov2WithRegistersModel.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        self.backbone.layernorm.elementwise_affine = False
        self.backbone.layernorm.weight = None
        self.backbone.layernorm.bias = None
        self.backbone.requires_grad_(False)

        processor = AutoImageProcessor.from_pretrained(
            model_dir,
            local_files_only=True,
        )
        image_mean = torch.as_tensor(
            processor.image_mean,
            dtype=torch.float32,
        )
        image_std = torch.as_tensor(
            processor.image_std,
            dtype=torch.float32,
        )
        if tuple(image_mean.shape) != (3,) or tuple(image_std.shape) != (3,):
            raise ValueError(
                "RAE/DINOv2 processor must provide three-channel image mean/std"
            )
        if (
            not torch.isfinite(image_mean).all()
            or not torch.isfinite(image_std).all()
        ):
            raise ValueError("RAE/DINOv2 processor image mean/std must be finite")
        if not torch.all(image_std > 0):
            raise ValueError("RAE/DINOv2 processor image std must be positive")
        self.register_buffer("image_mean", image_mean.view(1, 3, 1, 1))
        self.register_buffer("image_std", image_std.view(1, 3, 1, 1))

        self.cls_residual_mlp = None
        if cls_residual_mlp_enabled:
            self.cls_residual_mlp = ClsResidualMlp(
                input_dim=self.output_size,
                hidden_dim=cls_residual_mlp_hidden_dim,
                zero_init=cls_residual_mlp_zero_init,
            )

        self.to(device)
        if self.compute_dtype is not None:
            self.backbone.to(dtype=self.compute_dtype)
        self.train(False)
        if compile_backbone:
            from .compiled_visual import compile_visual_backbone
            compile_visual_backbone(self.backbone)

    @property
    def is_blind(self):
        return False

    def train(self, mode: bool = True):
        # Keep the frozen foundation model deterministic while allowing the
        # residual MLP to follow the surrounding policy's train/eval mode.
        super().train(mode)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        return self

    def _encode_rgb(self, rgb_observations) -> torch.Tensor:
        pixels = prepare_rae_rgb_tensor(
            rgb_observations,
            device=self.image_mean.device,
            size=self.encoder_input_size,
        )
        device_type = self.image_mean.device.type
        pixels = (pixels - self.image_mean) / self.image_std

        if self.compute_dtype is None:
            autocast_context = nullcontext()
            backbone_pixels = pixels
        elif self.compute_dtype == torch.float32:
            autocast_context = torch.autocast(
                device_type=device_type,
                enabled=False,
            )
            backbone_pixels = pixels.to(dtype=self.compute_dtype)
        else:
            autocast_context = torch.autocast(
                device_type=device_type,
                dtype=self.compute_dtype,
            )
            backbone_pixels = pixels.to(dtype=self.compute_dtype)

        with torch.no_grad(), autocast_context:
            hidden_state = self.backbone(
                backbone_pixels,
                output_hidden_states=self.retain_intermediate_states,
            ).last_hidden_state
        features = hidden_state[:, 0].float()
        if tuple(features.shape) != (pixels.shape[0], self.output_size):
            raise ValueError(
                "RAE/DINOv2 backbone returned CLS shape "
                f"{tuple(features.shape)}, expected "
                f"{(pixels.shape[0], self.output_size)}"
            )
        return features

    def _encode_rgb_with_patch_latents(self, rgb_observations):
        pixels = prepare_rae_rgb_tensor(
            rgb_observations,
            device=self.image_mean.device,
            size=self.encoder_input_size,
        )
        device_type = self.image_mean.device.type
        pixels = (pixels - self.image_mean) / self.image_std

        if self.compute_dtype is None:
            autocast_context = nullcontext()
            backbone_pixels = pixels
        elif self.compute_dtype == torch.float32:
            autocast_context = torch.autocast(
                device_type=device_type,
                enabled=False,
            )
            backbone_pixels = pixels.to(dtype=self.compute_dtype)
        else:
            autocast_context = torch.autocast(
                device_type=device_type,
                dtype=self.compute_dtype,
            )
            backbone_pixels = pixels.to(dtype=self.compute_dtype)

        with torch.no_grad(), autocast_context:
            hidden_state = self.backbone(
                backbone_pixels,
                output_hidden_states=self.retain_intermediate_states,
            ).last_hidden_state
        expected_tokens = 1 + 4 + 16 * 16
        expected_shape = (pixels.shape[0], expected_tokens, self.output_size)
        if tuple(hidden_state.shape) != expected_shape:
            raise ValueError(
                "RAE/DINOv2-with-registers returned token shape "
                f"{tuple(hidden_state.shape)}, expected {expected_shape}"
            )
        features = hidden_state[:, 0].float()
        patch_latents = (
            hidden_state[:, 5:]
            .float()
            .transpose(1, 2)
            .reshape(pixels.shape[0], self.output_size, 16, 16)
            .contiguous()
        )
        return features, patch_latents

    def _apply_cls_residual_mlp(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        features = features.float()
        if self.cls_residual_mlp is None:
            return features
        return features + self.cls_residual_mlp(features)

    def _check_finite(self, value, message):
        finite = torch.isfinite(value).all()
        if value.is_cuda and getattr(self, "async_finite_checks", False):
            # Detect bad values on the same CUDA stream without synchronizing
            # the host after every frozen encoder / residual-MLP forward.
            torch._assert_async(finite, message)
        elif not finite:
            raise FloatingPointError(message)

    def forward(self, observations: Mapping[str, torch.Tensor]):
        if not isinstance(observations, Mapping):
            raise TypeError("RAE/DINOv2 observations must be a mapping")
        if "rgb_features" in observations:
            features = observations["rgb_features"]
        elif "rgb" in observations:
            features = self._encode_rgb(observations["rgb"])
        else:
            raise ValueError(
                "RAE/DINOv2 observations must contain 'rgb' or 'rgb_features'"
            )

        if not torch.is_tensor(features) or features.ndim < 1:
            shape = tuple(features.shape) if torch.is_tensor(features) else None
            raise ValueError(
                f"RAE/DINOv2 CLS features must be a tensor, got shape {shape}"
            )
        if features.shape[-1] != self.output_size:
            raise ValueError(
                "RAE/DINOv2 CLS last dimension must be "
                f"{self.output_size}, got {features.shape[-1]}"
            )
        self._check_finite(features, "RAE/DINOv2 CLS contains NaN or infinity")

        output = self._apply_cls_residual_mlp(features)
        self._check_finite(output, "RAE/DINOv2 CLS contains NaN or infinity")
        return output

    def forward_with_patch_latents(
        self,
        observations: Mapping[str, torch.Tensor],
    ):
        _raw_cls, nav_cls, patch_latents = (
            self.forward_with_raw_cls_and_patch_latents(observations)
        )
        return nav_cls, patch_latents

    def forward_raw_cls_and_patch_latents(
        self,
        observations: Mapping[str, torch.Tensor],
    ):
        """Return frozen raw CLS/patch without touching the navigation MLP.

        Low-level world-model context collection runs under ``no_grad`` inside
        the navigation rollout's outer autocast context.  Calling the
        trainable residual MLP there can populate autocast's weight cache with
        detached casts and disconnect the later navigation forward.  Keep the
        raw context contract on an API that cannot execute that MLP.
        """

        if not isinstance(observations, Mapping):
            raise TypeError("RAE/DINOv2 observations must be a mapping")
        if "rgb_features" in observations:
            raw_features = observations["rgb_features"]
            patch_latents = observations.get("rgb_patch_latents")
            if patch_latents is None:
                raise ValueError(
                    "rgb_patch_latents is required with precomputed rgb_features "
                    "when RAE-NWM is enabled"
                )
        elif "rgb" in observations:
            raw_features, patch_latents = self._encode_rgb_with_patch_latents(
                observations["rgb"]
            )
        else:
            raise ValueError(
                "RAE/DINOv2 observations must contain 'rgb' or 'rgb_features'"
            )

        if tuple(patch_latents.shape[1:]) != (self.output_size, 16, 16):
            raise ValueError(
                "RAE/DINOv2 patch latents must have shape [N,768,16,16], "
                f"got {tuple(patch_latents.shape)}"
            )
        if not torch.isfinite(patch_latents).all():
            raise FloatingPointError("RAE/DINOv2 patch latents contain NaN or infinity")
        raw_features = raw_features.float()
        if tuple(raw_features.shape[-1:]) != (self.output_size,):
            raise ValueError(
                "RAE/DINOv2 raw CLS last dimension must be "
                f"{self.output_size}, got {raw_features.shape[-1]}"
            )
        if not torch.isfinite(raw_features).all():
            raise FloatingPointError("RAE/DINOv2 raw CLS contains NaN or infinity")
        return raw_features, patch_latents.float()

    def forward_with_raw_cls_and_patch_latents(
        self,
        observations: Mapping[str, torch.Tensor],
    ):
        """Return frozen raw CLS/patch plus the trainable navigation CLS."""

        raw_features, patch_latents = self.forward_raw_cls_and_patch_latents(
            observations
        )
        nav_features = self._apply_cls_residual_mlp(raw_features)
        if not torch.isfinite(nav_features).all():
            raise FloatingPointError("RAE/DINOv2 CLS contains NaN or infinity")
        return raw_features, nav_features, patch_latents


# Keep the old import name available for external callers while using the
# ETPNav-compatible class and constructor everywhere in this repository.
RaeDinov2ClsEncoder = RaeDinov2RgbEncoder
