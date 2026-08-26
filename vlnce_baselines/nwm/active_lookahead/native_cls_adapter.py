"""Trainable adapter for native NWM q1 CLS tokens."""

from __future__ import annotations

import torch
from torch import nn


def expand_native_cls_condition(condition: torch.Tensor) -> torch.Tensor:
    """Expand normalized NWM `[dx,dy,dtheta,rel_t]` to the fixed 5-D form."""

    if not torch.is_tensor(condition) or condition.ndim < 1:
        raise TypeError("native CLS condition must be a torch.Tensor")
    if int(condition.shape[-1]) != 4:
        raise ValueError(
            "native CLS condition must end with [dx,dy,dtheta,rel_t], got "
            f"{tuple(condition.shape)}"
        )
    dx, dy, dtheta, rel_t = condition.unbind(dim=-1)
    return torch.stack(
        (dx, dy, torch.sin(dtheta), torch.cos(dtheta), rel_t),
        dim=-1,
    )


class Top5NativeClsAdapter(nn.Module):
    """Condition and residually adjust only q1's native CLS token."""

    def __init__(
        self,
        *,
        feature_dim: int = 768,
        condition_hidden_dim: int = 128,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.condition_hidden_dim = int(condition_hidden_dim)
        if self.feature_dim <= 0 or self.condition_hidden_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        self.cls_norm = nn.LayerNorm(self.feature_dim)
        self.condition = nn.Sequential(
            nn.Linear(5, self.condition_hidden_dim),
            nn.GELU(),
            nn.Linear(self.condition_hidden_dim, self.condition_hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                self.feature_dim + self.condition_hidden_dim,
                self.feature_dim,
            ),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        if zero_init:
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        pred_cls_raw: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if pred_cls_raw.shape[:-1] != condition.shape[:-1]:
            raise ValueError("CLS and condition leading dimensions must match")
        if int(pred_cls_raw.shape[-1]) != self.feature_dim:
            raise ValueError(
                f"CLS feature dim must be {self.feature_dim}, got "
                f"{pred_cls_raw.shape[-1]}"
            )
        expanded = expand_native_cls_condition(condition).to(
            device=pred_cls_raw.device,
            dtype=pred_cls_raw.dtype,
        )
        condition_features = self.condition(expanded)
        delta = self.fusion(
            torch.cat((self.cls_norm(pred_cls_raw), condition_features), dim=-1)
        )
        adapted = pred_cls_raw + delta
        return adapted, {"cls_delta_norm": delta.detach().norm(dim=-1)}

    def adapt_tokens(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.ndim < 3 or int(tokens.shape[-2]) != 257:
            raise ValueError(
                "native q1 tokens must have 257 tokens, got "
                f"{tuple(tokens.shape)}"
            )
        adapted_cls, diagnostics = self(tokens[..., 0, :], condition)
        adapted = torch.cat((adapted_cls.unsqueeze(-2), tokens[..., 1:, :]), dim=-2)
        return adapted, diagnostics
