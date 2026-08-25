from typing import Optional

import torch
from torch import nn

from .types import FutureHeadOutput


class _InterleavedCrossModalBlock(nn.Module):
    """Let candidate queries read text, patches, and global CLS in one round."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_attention_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.text_attention = nn.MultiheadAttention(
            hidden_dim, num_attention_heads, dropout=dropout, batch_first=True
        )
        self.patch_attention = nn.MultiheadAttention(
            hidden_dim, num_attention_heads, dropout=dropout, batch_first=True
        )
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.patch_norm = nn.LayerNorm(hidden_dim)
        self.cls_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.cls_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        text_h: torch.Tensor,
        patch_h: torch.Tensor,
        cls_h: torch.Tensor,
        *,
        text_padding_mask: Optional[torch.Tensor],
        patch_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        text_update, _ = self.text_attention(
            queries,
            text_h,
            text_h,
            key_padding_mask=text_padding_mask,
            need_weights=False,
        )
        queries = self.text_norm(queries + text_update)
        patch_update, _ = self.patch_attention(
            queries,
            patch_h,
            patch_h,
            key_padding_mask=patch_padding_mask,
            need_weights=False,
        )
        queries = self.patch_norm(queries + patch_update)
        cls_expanded = cls_h[:, None, :].expand_as(queries)
        cls_gate = torch.sigmoid(
            self.cls_gate(
                torch.cat(
                    (queries, cls_expanded, queries * cls_expanded), dim=-1
                )
            )
        )
        queries = self.cls_norm(queries + cls_gate * cls_expanded)
        return self.ffn_norm(queries + self.ffn(queries))


class InterleavedCrossModalTopKFutureLogitResidualHead(nn.Module):
    """Retained E24 scorer with full per-candidate fusion and TopK comparison.

    Each round reads the complete instruction, future patch tokens, and the
    dedicated future CLS token.  The resulting candidate summaries then
    interact across TopK and are fed back into the next fusion round.  Frozen
    q0 geometry and base log-probability enter before the first round, rather
    than being appended only after semantic compression.
    """

    supports_joint_topk = True
    supports_candidate_geometry = True

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        num_queries: int = 8,
        num_attention_heads: int = 12,
        num_layers: int = 1,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        delta_max: float = 1.0,
        fusion_layers: int = 3,
        delta_centering: str = "none",
        score_context: str = "base_bounded_margin_relative",
        round_weight_sharing: str = "independent",
        residual_confidence_gate: str = "none",
    ) -> None:
        super().__init__()
        if num_layers != 1:
            raise ValueError("E24 scorer requires num_layers=1")
        if min(input_dim, hidden_dim, num_queries, ffn_dim, fusion_layers) <= 0:
            raise ValueError("E17 scorer dimensions and fusion_layers must be positive")
        if hidden_dim % num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if delta_max <= 0:
            raise ValueError("delta_max must be positive")
        retained = {
            "delta_centering": (delta_centering, "none"),
            "score_context": (
                score_context,
                "base_bounded_margin_relative",
            ),
            "round_weight_sharing": (round_weight_sharing, "independent"),
            "residual_confidence_gate": (residual_confidence_gate, "none"),
        }
        for name, (actual, expected) in retained.items():
            if actual != expected:
                raise ValueError(
                    f"E24 scorer requires {name}={expected!r}, got {actual!r}"
                )
        self.delta_max = float(delta_max)
        self.fusion_layers = int(fusion_layers)
        self.delta_centering = str(delta_centering)
        self.score_context = str(score_context)
        self.round_weight_sharing = str(round_weight_sharing)
        self.residual_confidence_gate = str(residual_confidence_gate)
        self.owner_norm = nn.LayerNorm(input_dim)
        self.owner_proj = nn.Linear(input_dim, hidden_dim)
        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.future_proj = nn.Linear(input_dim, hidden_dim)
        self.geometry_proj = nn.Linear(3, hidden_dim, bias=False)
        self.base_context_proj = nn.Linear(1, hidden_dim, bias=False)
        self.learned_queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.normal_(self.learned_queries, std=0.02)
        self.fusion_blocks = nn.ModuleList(
            [
                _InterleavedCrossModalBlock(
                    hidden_dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(self.fusion_layers)
            ]
        )
        self.candidate_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_attention_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                for _ in range(self.fusion_layers)
            ]
        )
        self.feedback_proj = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(self.fusion_layers - 1)]
        )
        self.feedback_norm = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(self.fusion_layers - 1)]
        )
        # Candidate features, frozen-winner comparison, bounded row margin,
        # and explicit winner flag. This shape is part of the E24 checkpoint.
        score_input_dim = hidden_dim * 5 + 4
        self.score_mlp = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def forward_topk_from_log_probs(
        self,
        owner_embeddings: torch.Tensor,
        text_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        base_ghost_log_probs: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        *,
        text_token_mask: Optional[torch.Tensor] = None,
        future_token_mask: Optional[torch.Tensor] = None,
        candidate_geometry: Optional[torch.Tensor] = None,
    ) -> FutureHeadOutput:
        if owner_embeddings.ndim != 3 or future_tokens.ndim != 4:
            raise ValueError("E17 owner/future inputs must have shape [B,K,D]/[B,K,T,D]")
        batch, topk, _ = owner_embeddings.shape
        if future_tokens.shape[:2] != (batch, topk) or future_tokens.shape[2] < 2:
            raise ValueError("E17 future input requires CLS plus patch tokens for every TopK slot")
        if base_ghost_log_probs.shape != (batch, topk):
            raise ValueError("E17 base log probabilities must have shape [B,K]")
        if candidate_valid_mask.shape != (batch, topk):
            raise ValueError("E17 candidate mask must have shape [B,K]")
        if candidate_geometry is None or candidate_geometry.shape != (batch, topk, 3):
            raise ValueError("E17 scorer requires candidate q0 geometry with shape [B,K,3]")
        if text_tokens.ndim not in (3, 4) or text_tokens.shape[0] != batch:
            raise ValueError("E17 text must have shape [B,L,D] or [B,K,L,D]")
        if text_tokens.ndim == 4 and text_tokens.shape[1] != topk:
            raise ValueError("candidate-specific E17 text must have K slots")

        valid = candidate_valid_mask.detach().to(
            device=owner_embeddings.device, dtype=torch.bool
        )
        base_lp = base_ghost_log_probs.detach().to(
            device=owner_embeddings.device, dtype=owner_embeddings.dtype
        )
        geometry = candidate_geometry.detach().to(
            device=owner_embeddings.device, dtype=owner_embeddings.dtype
        )
        if not bool(torch.isfinite(base_lp[valid]).all()):
            raise ValueError("valid E17 base log probabilities must be finite")
        if not bool(torch.isfinite(geometry[valid]).all()):
            raise ValueError("valid E17 candidate geometry must be finite")

        rows, slots = valid.nonzero(as_tuple=True)
        zero = next(self.parameters()).reshape(-1)[0] * 0.0
        raw_dense = owner_embeddings.new_zeros(batch, topk) + zero
        hidden_dim = self.owner_proj.out_features
        summary_dense = owner_embeddings.new_zeros(batch, topk, hidden_dim) + zero
        if not rows.numel():
            return FutureHeadOutput(raw_dense, raw_dense, summary_dense, base_lp)

        owner_h = self.owner_proj(self.owner_norm(owner_embeddings.detach()))
        geometry_h = self.geometry_proj(geometry)
        base_h = self.base_context_proj(base_lp.unsqueeze(-1))
        selected_owner = owner_h[rows, slots]
        selected_queries = (
            self.learned_queries[None, :, :]
            + selected_owner[:, None, :]
            + geometry_h[rows, slots, None, :]
            + base_h[rows, slots, None, :]
        )

        text_h = self.text_proj(text_tokens.detach())
        selected_text = text_h[rows, slots] if text_h.ndim == 4 else text_h[rows]
        if text_token_mask is None:
            selected_text_mask = None
        else:
            selected_text_mask = (
                text_token_mask[rows, slots]
                if text_token_mask.ndim == 3
                else text_token_mask[rows]
            )
        future_h = self.future_proj(future_tokens.detach())
        selected_future = future_h[rows, slots]
        selected_cls = selected_future[:, 0]
        selected_patches = selected_future[:, 1:]
        selected_patch_mask = None
        if future_token_mask is not None:
            selected_future_mask = future_token_mask[rows, slots]
            selected_cls = selected_cls * selected_future_mask[:, 0, None].to(
                device=selected_cls.device, dtype=selected_cls.dtype
            )
            selected_patch_mask = selected_future_mask[:, 1:]

        active_rows = valid.any(dim=1).nonzero(as_tuple=False).flatten()
        for layer_index in range(self.fusion_layers):
            fusion = self.fusion_blocks[layer_index]
            candidate_layer = self.candidate_layers[layer_index]
            selected_queries = fusion(
                selected_queries,
                selected_text,
                selected_patches,
                selected_cls,
                text_padding_mask=_padding_mask(selected_text_mask, selected_text),
                patch_padding_mask=_padding_mask(
                    selected_patch_mask, selected_patches
                ),
            )
            local_summary = selected_queries.mean(dim=1)
            round_dense = owner_embeddings.new_zeros(batch, topk, hidden_dim) + zero
            round_dense[rows, slots] = local_summary
            interacted = candidate_layer(
                round_dense[active_rows], src_key_padding_mask=~valid[active_rows]
            )
            summary_dense = owner_embeddings.new_zeros(batch, topk, hidden_dim) + zero
            summary_dense[active_rows] = interacted
            if layer_index + 1 < self.fusion_layers:
                feedback = self.feedback_proj[layer_index](summary_dense[rows, slots])
                selected_queries = self.feedback_norm[layer_index](
                    selected_queries + feedback[:, None, :]
                )

        final_summary = summary_dense[rows, slots]
        score_parts = [
            final_summary,
            selected_owner,
            final_summary * selected_owner,
            base_lp[rows, slots, None],
        ]
        valid_base_lp = base_lp.masked_fill(~valid, -torch.inf)
        base_slots = valid_base_lp.argmax(dim=1)
        base_summary_by_row = summary_dense[
            torch.arange(batch, device=summary_dense.device), base_slots
        ]
        selected_base_summary = base_summary_by_row[rows]
        selected_base_lp = valid_base_lp[
            torch.arange(batch, device=base_lp.device), base_slots
        ][rows]
        base_competitors = valid_base_lp.clone()
        base_competitors[
            torch.arange(batch, device=base_lp.device), base_slots
        ] = -torch.inf
        runner_up_lp = base_competitors.max(dim=1).values
        winner_margin = selected_base_lp - runner_up_lp[rows]
        winner_margin = torch.where(
            valid.sum(dim=1)[rows] > 1,
            winner_margin,
            torch.zeros_like(winner_margin),
        )
        winner_margin = winner_margin / (1.0 + winner_margin)
        selected_is_winner = (slots == base_slots[rows]).to(dtype=base_lp.dtype)
        score_parts.extend(
            (
                final_summary - selected_base_summary,
                final_summary * selected_base_summary,
                (base_lp[rows, slots] - selected_base_lp)[:, None],
                winner_margin[:, None],
                selected_is_winner[:, None],
            )
        )
        score_input = torch.cat(score_parts, dim=-1)
        selected_raw = self.score_mlp(score_input).squeeze(-1)
        raw_dense = owner_embeddings.new_zeros(batch, topk) + zero
        raw_dense[rows, slots] = selected_raw
        delta = self.delta_max * torch.tanh(raw_dense)
        return FutureHeadOutput(
            raw_dense,
            delta.masked_fill(~valid, 0.0),
            summary_dense,
            base_lp,
        )


def _padding_mask(valid_mask: Optional[torch.Tensor], values: torch.Tensor):
    if valid_mask is None:
        return None
    if valid_mask.shape != values.shape[:2]:
        raise ValueError("token mask must match [N,T]")
    valid_mask = valid_mask.detach().to(device=values.device, dtype=torch.bool)
    if torch.any(~valid_mask.any(dim=1)):
        raise ValueError("each attention row must contain at least one valid token")
    return ~valid_mask
