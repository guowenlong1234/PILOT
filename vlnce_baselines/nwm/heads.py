from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn


def _as_tuple2(value, name: str) -> Tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"{name} must be a length-2 tuple/list, got {value!r}")


@dataclass
class NwmHeadConfig:
    input_dim: int = 768
    output_dim: int = 768
    grid_size: Tuple[int, int] = (16, 16)
    pos_embed_type: str = "learned_2d"
    norm_eps: float = 1e-6
    activation: str = "gelu"
    token_head_trainable: bool = True
    confidence_head_trainable: bool = True
    token_pooler_type: str = "query_attention"
    token_pool_dim: int = 768
    token_num_queries: int = 1
    token_head_depth: int = 1
    token_head_num_heads: int = 8
    token_head_mlp_ratio: float = 4.0
    token_hidden_dim: int = 768
    token_projection_depth: int = 2
    token_dropout: float = 0.0
    confidence_pooler_type: str = "query_attention"
    confidence_pool_dim: int = 512
    confidence_num_queries: int = 1
    confidence_head_depth: int = 1
    confidence_head_num_heads: int = 8
    confidence_head_mlp_ratio: float = 4.0
    confidence_condition_dim: int = 128
    confidence_hidden_dim: int = 512
    confidence_classifier_depth: int = 2
    confidence_dropout: float = 0.0
    token_cosine_weight: float = 1.0
    token_smooth_l1_weight: float = 0.25
    confidence_mean_error_weight: float = 0.8
    confidence_topk_error_weight: float = 0.2
    confidence_topk_ratio: float = 0.2
    confidence_label_epsilon: float = 1e-6

    def __post_init__(self):
        self.input_dim = int(self.input_dim)
        self.output_dim = int(self.output_dim)
        self.grid_size = _as_tuple2(self.grid_size, "grid_size")
        self.token_pool_dim = int(self.token_pool_dim)
        self.token_num_queries = int(self.token_num_queries)
        self.token_head_depth = int(self.token_head_depth)
        self.token_head_num_heads = int(self.token_head_num_heads)
        self.token_head_mlp_ratio = float(self.token_head_mlp_ratio)
        self.token_hidden_dim = int(self.token_hidden_dim)
        self.token_projection_depth = int(self.token_projection_depth)
        self.token_dropout = float(self.token_dropout)
        self.confidence_pool_dim = int(self.confidence_pool_dim)
        self.confidence_num_queries = int(self.confidence_num_queries)
        self.confidence_head_depth = int(self.confidence_head_depth)
        self.confidence_head_num_heads = int(self.confidence_head_num_heads)
        self.confidence_head_mlp_ratio = float(self.confidence_head_mlp_ratio)
        self.confidence_condition_dim = int(self.confidence_condition_dim)
        self.confidence_hidden_dim = int(self.confidence_hidden_dim)
        self.confidence_classifier_depth = int(self.confidence_classifier_depth)
        self.confidence_dropout = float(self.confidence_dropout)
        self.token_cosine_weight = float(self.token_cosine_weight)
        self.token_smooth_l1_weight = float(self.token_smooth_l1_weight)
        self.confidence_mean_error_weight = float(self.confidence_mean_error_weight)
        self.confidence_topk_error_weight = float(self.confidence_topk_error_weight)
        self.confidence_topk_ratio = float(self.confidence_topk_ratio)
        self.confidence_label_epsilon = float(self.confidence_label_epsilon)
        self._validate()

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None):
        return cls(**dict(data or {}))

    def to_dict(self) -> Dict[str, Any]:
        values = dict(self.__dict__)
        values["grid_size"] = list(self.grid_size)
        return values

    def _validate(self):
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.grid_size[0] <= 0 or self.grid_size[1] <= 0:
            raise ValueError("grid_size entries must be positive")
        if self.token_pooler_type not in {"mean", "query_attention"}:
            raise ValueError(
                "token_pooler_type must be mean or query_attention, "
                f"got {self.token_pooler_type}"
            )
        if self.confidence_pooler_type not in {"mean", "query_attention"}:
            raise ValueError(
                "confidence_pooler_type must be mean or query_attention, "
                f"got {self.confidence_pooler_type}"
            )
        if self.pos_embed_type not in {"learned_2d", "none"}:
            raise ValueError(
                "pos_embed_type must be learned_2d or none, "
                f"got {self.pos_embed_type}"
            )
        for name in (
            "token_pool_dim",
            "token_num_queries",
            "token_head_num_heads",
            "token_hidden_dim",
            "confidence_pool_dim",
            "confidence_num_queries",
            "confidence_head_num_heads",
            "confidence_condition_dim",
            "confidence_hidden_dim",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("token_head_mlp_ratio", "confidence_head_mlp_ratio"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.token_pool_dim % self.token_head_num_heads != 0:
            raise ValueError(
                "token_pool_dim must be divisible by token_head_num_heads"
            )
        if self.confidence_pool_dim % self.confidence_head_num_heads != 0:
            raise ValueError(
                "confidence_pool_dim must be divisible by confidence_head_num_heads"
            )
        for name in (
            "token_head_depth",
            "token_projection_depth",
            "confidence_head_depth",
            "confidence_classifier_depth",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("token_dropout", "confidence_dropout"):
            value = getattr(self, name)
            if value < 0.0 or value >= 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.confidence_topk_ratio <= 0.0 or self.confidence_topk_ratio > 1.0:
            raise ValueError("confidence_topk_ratio must be in (0, 1]")
        for name in (
            "token_cosine_weight",
            "token_smooth_l1_weight",
            "confidence_mean_error_weight",
            "confidence_topk_error_weight",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.token_cosine_weight + self.token_smooth_l1_weight <= 0.0:
            raise ValueError("at least one token loss weight must be positive")
        if self.confidence_mean_error_weight + self.confidence_topk_error_weight <= 0.0:
            raise ValueError("at least one confidence error weight must be positive")
        if self.confidence_label_epsilon <= 0.0:
            raise ValueError("confidence_label_epsilon must be positive")
        if not isinstance(self.token_head_trainable, bool):
            raise ValueError("token_head_trainable must be bool")
        if not isinstance(self.confidence_head_trainable, bool):
            raise ValueError("confidence_head_trainable must be bool")


class NwmOutputHead:
    """Backward-compatible extension boundary name."""

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Use NwmOutputHeads for token/confidence prediction.")


def patch_tokens_from_latent(pred_patch, config: NwmHeadConfig):
    if pred_patch.dim() == 4:
        _batch, channels, height, width = pred_patch.shape
        if channels != config.input_dim:
            raise ValueError(
                f"pred_patch channel dim must be {config.input_dim}, got {channels}"
            )
        if (height, width) != config.grid_size:
            raise ValueError(
                "pred_patch spatial shape must match "
                f"grid_size {config.grid_size}, got {(height, width)}"
            )
        return pred_patch.flatten(2).transpose(1, 2).contiguous()
    if pred_patch.dim() == 3:
        _batch, tokens, channels = pred_patch.shape
        expected_tokens = config.grid_size[0] * config.grid_size[1]
        if tokens != expected_tokens:
            raise ValueError(
                "pred_patch token count must match "
                f"grid_size {config.grid_size}, got {tokens}"
            )
        if channels != config.input_dim:
            raise ValueError(
                f"pred_patch feature dim must be {config.input_dim}, got {channels}"
            )
        return pred_patch
    raise ValueError(
        "pred_patch must have shape [B, C, H, W] or [B, N, C], "
        f"got {tuple(pred_patch.shape)}"
    )


def _activation(name: str):
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
    dropout: float,
    activation: str,
):
    layers = []
    current_dim = input_dim
    for _ in range(max(1, int(depth)) - 1):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class QueryAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        activation: str,
        norm_eps: float,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.query_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.key_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = _make_mlp(
            dim,
            hidden_dim,
            dim,
            depth=2,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, query, tokens):
        normed_tokens = self.key_norm(tokens)
        attended, _ = self.attn(
            self.query_norm(query),
            normed_tokens,
            normed_tokens,
            need_weights=False,
        )
        query = query + attended
        query = query + self.mlp(self.mlp_norm(query))
        return query


@dataclass
class NwmHeadOutputs:
    pred_cls: Any
    conf_logit: Any
    confidence: Any


class PatchTokenPooler(nn.Module):
    def __init__(
        self,
        config: NwmHeadConfig,
        *,
        pooler_type: str,
        pool_dim: int,
        num_queries: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.config = config
        self.pooler_type = pooler_type
        self.input_norm = nn.LayerNorm(config.input_dim, eps=config.norm_eps)
        if config.input_dim != pool_dim:
            self.input_proj = nn.Linear(config.input_dim, pool_dim)
        else:
            self.input_proj = nn.Identity()
        if config.pos_embed_type == "learned_2d":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, config.grid_size[0] * config.grid_size[1], pool_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_parameter("pos_embed", None)
        if pooler_type == "query_attention":
            self.query = nn.Parameter(torch.zeros(1, num_queries, pool_dim))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.blocks = nn.ModuleList(
                QueryAttentionBlock(
                    pool_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    config.activation,
                    config.norm_eps,
                )
                for _ in range(depth)
            )
            self.output_norm = nn.LayerNorm(pool_dim, eps=config.norm_eps)
        else:
            self.query = None
            self.blocks = nn.ModuleList()
            self.output_norm = nn.Identity()

    def forward(self, pred_patch):
        tokens = patch_tokens_from_latent(pred_patch, self.config)
        tokens = self.input_proj(self.input_norm(tokens))
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed
        if self.pooler_type == "mean":
            return tokens.mean(dim=1)
        query = self.query.expand(tokens.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, tokens)
        return self.output_norm(query).mean(dim=1)


class NwmTokenHead(nn.Module):
    def __init__(self, config: NwmHeadConfig):
        super().__init__()
        self.pooler = PatchTokenPooler(
            config,
            pooler_type=config.token_pooler_type,
            pool_dim=config.token_pool_dim,
            num_queries=config.token_num_queries,
            depth=config.token_head_depth,
            num_heads=config.token_head_num_heads,
            mlp_ratio=config.token_head_mlp_ratio,
            dropout=config.token_dropout,
        )
        self.projection = _make_mlp(
            config.token_pool_dim,
            config.token_hidden_dim,
            config.output_dim,
            config.token_projection_depth,
            config.token_dropout,
            config.activation,
        )

    def forward(self, pred_patch):
        return self.projection(self.pooler(pred_patch))


class NwmConfidenceHead(nn.Module):
    def __init__(self, config: NwmHeadConfig):
        super().__init__()
        self.pooler = PatchTokenPooler(
            config,
            pooler_type=config.confidence_pooler_type,
            pool_dim=config.confidence_pool_dim,
            num_queries=config.confidence_num_queries,
            depth=config.confidence_head_depth,
            num_heads=config.confidence_head_num_heads,
            mlp_ratio=config.confidence_head_mlp_ratio,
            dropout=config.confidence_dropout,
        )
        self.condition_mlp = _make_mlp(
            5,
            config.confidence_condition_dim,
            config.confidence_condition_dim,
            depth=2,
            dropout=config.confidence_dropout,
            activation=config.activation,
        )
        self.classifier = _make_mlp(
            config.confidence_pool_dim + config.confidence_condition_dim,
            config.confidence_hidden_dim,
            1,
            config.confidence_classifier_depth,
            config.confidence_dropout,
            config.activation,
        )

    def forward(self, pred_patch, condition):
        if condition.dim() != 2 or condition.shape[1] != 4:
            raise ValueError(f"condition must have shape [B, 4], got {tuple(condition.shape)}")
        patch_feature = self.pooler(pred_patch)
        dtheta = condition[:, 2]
        condition_expanded = torch.stack(
            [
                condition[:, 0],
                condition[:, 1],
                torch.sin(dtheta),
                torch.cos(dtheta),
                condition[:, 3],
            ],
            dim=-1,
        )
        condition_feature = self.condition_mlp(
            condition_expanded.to(patch_feature.dtype)
        )
        return self.classifier(
            torch.cat([patch_feature, condition_feature], dim=-1)
        ).squeeze(-1)


class NwmOutputHeads(nn.Module):
    def __init__(self, config: NwmHeadConfig | Mapping[str, Any] | None = None):
        super().__init__()
        if isinstance(config, NwmHeadConfig):
            self.config = config
        else:
            self.config = NwmHeadConfig.from_mapping(config)
        self.token_head = NwmTokenHead(self.config)
        self.confidence_head = NwmConfidenceHead(self.config)
        self.set_trainable(
            self.config.token_head_trainable,
            self.config.confidence_head_trainable,
        )

    def set_trainable(
        self,
        token_head_trainable: bool,
        confidence_head_trainable: bool,
    ):
        for param in self.token_head.parameters():
            param.requires_grad = bool(token_head_trainable)
        for param in self.confidence_head.parameters():
            param.requires_grad = bool(confidence_head_trainable)

    def forward(self, pred_patch, condition):
        pred_cls = self.token_head(pred_patch)
        conf_logit = self.confidence_head(pred_patch, condition)
        return NwmHeadOutputs(
            pred_cls=pred_cls,
            conf_logit=conf_logit,
            confidence=torch.sigmoid(conf_logit),
        )


def trainable_head_parameters(heads: NwmOutputHeads):
    return [param for param in heads.parameters() if param.requires_grad]


def token_head_loss(pred_cls, gt_cls, config: NwmHeadConfig):
    import torch.nn.functional as F

    cosine = 1.0 - F.cosine_similarity(pred_cls, gt_cls, dim=-1)
    pred_norm = F.normalize(pred_cls, dim=-1)
    gt_norm = F.normalize(gt_cls, dim=-1)
    smooth_l1 = F.smooth_l1_loss(
        pred_norm,
        gt_norm,
        reduction="none",
    ).mean(dim=-1)
    return (
        config.token_cosine_weight * cosine
        + config.token_smooth_l1_weight * smooth_l1
    ).mean()


def patch_cosine_error(pred_patch, gt_patch, config: NwmHeadConfig):
    import torch.nn.functional as F

    pred_tokens = F.normalize(patch_tokens_from_latent(pred_patch, config), dim=-1)
    gt_tokens = F.normalize(patch_tokens_from_latent(gt_patch, config), dim=-1)
    patch_error = 1.0 - (pred_tokens * gt_tokens).sum(dim=-1)
    topk = max(
        1,
        int(
            torch.ceil(
                torch.tensor(
                    patch_error.shape[1] * config.confidence_topk_ratio,
                    device=patch_error.device,
                )
            ).item()
        ),
    )
    topk_error = patch_error.topk(topk, dim=1).values.mean(dim=1)
    mean_error = patch_error.mean(dim=1)
    return (
        config.confidence_mean_error_weight * mean_error
        + config.confidence_topk_error_weight * topk_error
    )


def confidence_labels_from_patch_error(error, e_good, e_bad, config: NwmHeadConfig):
    import math

    e_good = torch.as_tensor(e_good, device=error.device, dtype=error.dtype)
    e_bad = torch.as_tensor(e_bad, device=error.device, dtype=error.dtype)
    logit_08 = math.log(0.8 / 0.2)
    denom = torch.clamp(
        e_bad - e_good,
        min=float(config.confidence_label_epsilon),
    )
    scale = denom / (2.0 * logit_08)
    mid = (e_good + e_bad) / 2.0
    return torch.sigmoid((mid - error) / scale)


def confidence_head_loss(conf_logit, confidence_label):
    import torch.nn.functional as F

    return F.binary_cross_entropy_with_logits(conf_logit, confidence_label)
