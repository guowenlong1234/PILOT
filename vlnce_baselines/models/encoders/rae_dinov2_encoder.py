import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2WithRegistersModel


def _prepare_cls_stat(
    stat: torch.Tensor,
    cls: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(stat):
        raise ValueError(f"RAE {name} must be a tensor, got {type(stat).__name__}")
    if not torch.isfinite(stat).all():
        raise ValueError(f"RAE {name} must contain only finite values")

    channels = cls.shape[1]
    if stat.ndim == 1 and stat.shape[0] == channels:
        prepared = stat.unsqueeze(0)
    elif stat.ndim == 2 and tuple(stat.shape) == (1, channels):
        prepared = stat
    elif stat.ndim == 3 and stat.shape[0] == channels:
        prepared = stat.mean(dim=(1, 2)).unsqueeze(0)
    elif (
        stat.ndim == 4
        and stat.shape[0] == 1
        and stat.shape[1] == channels
    ):
        prepared = stat.mean(dim=(2, 3))
    else:
        raise ValueError(
            f"RAE {name} shape {tuple(stat.shape)} is incompatible with "
            f"CLS shape {tuple(cls.shape)}"
        )

    prepared = prepared.to(device=cls.device, dtype=cls.dtype)
    if not torch.isfinite(prepared).all():
        raise ValueError(f"RAE {name} must contain only finite values")
    return prepared


def normalize_rae_cls(
    cls: torch.Tensor,
    mean: torch.Tensor | None,
    var: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Normalize a batched CLS vector with RAE spatial latent statistics."""
    if not torch.is_tensor(cls) or cls.ndim != 2:
        shape = tuple(cls.shape) if torch.is_tensor(cls) else None
        raise ValueError(f"RAE CLS must have shape [B, C], got {shape}")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise ValueError(f"RAE normalization eps must be a finite non-negative number, got {eps}")
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError(f"RAE normalization eps must be a finite non-negative number, got {eps}")

    cls = cls.float()
    prepared_mean = 0.0 if mean is None else _prepare_cls_stat(mean, cls, "mean")
    prepared_var = _prepare_cls_stat(var, cls, "var")
    normalized = (cls - prepared_mean) / torch.sqrt(prepared_var + eps)
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("RAE/DINOv2 CLS contains NaN or infinity")
    return normalized


class RaeDinov2ClsEncoder(nn.Module):
    output_size = 768

    def __init__(self, model_dir, stat_path, device):
        super().__init__()
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
        image_mean = torch.as_tensor(processor.image_mean, dtype=torch.float32)
        image_std = torch.as_tensor(processor.image_std, dtype=torch.float32)
        if tuple(image_mean.shape) != (3,) or tuple(image_std.shape) != (3,):
            raise ValueError(
                "RAE/DINOv2 processor must provide three-channel image mean/std"
            )
        if not torch.isfinite(image_mean).all() or not torch.isfinite(image_std).all():
            raise ValueError("RAE/DINOv2 processor image mean/std must be finite")
        if not torch.all(image_std > 0):
            raise ValueError("RAE/DINOv2 processor image std must be positive")
        self.register_buffer("image_mean", image_mean.view(1, 3, 1, 1))
        self.register_buffer("image_std", image_std.view(1, 3, 1, 1))

        stats = torch.load(stat_path, map_location="cpu")
        if not isinstance(stats, Mapping):
            raise ValueError("RAE stat file must contain a mapping")
        latent_var = stats.get("var")
        if not torch.is_tensor(latent_var):
            raise ValueError("RAE stat file must contain tensor key 'var'")
        self.register_buffer("latent_var", latent_var.float())
        latent_mean = stats.get("mean")
        if latent_mean is not None and not torch.is_tensor(latent_mean):
            raise ValueError("RAE stat key 'mean' must be a tensor or None")
        self.register_buffer(
            "latent_mean",
            None if latent_mean is None else latent_mean.float(),
        )

        self.to(device)
        self.train(False)

    @property
    def is_blind(self):
        return False

    def train(self, mode=True):
        super().train(False)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, observations):
        if not isinstance(observations, Mapping) or "rgb" not in observations:
            raise ValueError("RAE/DINOv2 observations must contain an 'rgb' tensor")
        rgb = observations["rgb"]
        if not torch.is_tensor(rgb) or rgb.ndim != 4:
            shape = tuple(rgb.shape) if torch.is_tensor(rgb) else None
            raise ValueError(f"RAE/DINOv2 requires BHWC RGB, got {shape}")
        if rgb.shape[-1] != 3:
            raise ValueError(
                f"RAE/DINOv2 requires BHWC RGB with 3 channels, got {tuple(rgb.shape)}"
            )
        if tuple(rgb.shape[1:3]) != (224, 224):
            raise ValueError(
                f"RAE/DINOv2 requires native 224x224 RGB, got {tuple(rgb.shape)}"
            )
        if rgb.dtype != torch.uint8:
            raise ValueError(f"RAE/DINOv2 requires uint8 RGB, got {rgb.dtype}")

        rgb = rgb.permute(0, 3, 1, 2).contiguous()
        rgb = rgb.to(device=self.image_mean.device, dtype=torch.float32).div(255.0)
        rgb = (rgb - self.image_mean) / self.image_std
        hidden_state = self.backbone(rgb).last_hidden_state
        cls = hidden_state[:, 0].float()
        if tuple(cls.shape) != (rgb.shape[0], self.output_size):
            raise ValueError(
                f"RAE/DINOv2 backbone returned CLS shape {tuple(cls.shape)}, "
                f"expected {(rgb.shape[0], self.output_size)}"
            )
        cls = normalize_rae_cls(
            cls,
            self.latent_mean,
            self.latent_var,
            eps=1e-5,
        )
        if not torch.isfinite(cls).all():
            raise FloatingPointError("RAE/DINOv2 CLS contains NaN or infinity")
        return cls
