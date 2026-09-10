"""Batched prediction residuals with optional persistent ghost node state."""

import math

import torch
from torch import nn


class _ConcatResidualMlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, zero_init):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        if zero_init:
            nn.init.zeros_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

    def forward(self, value):
        return self.layers(value)


class GhostConcatFusionAdapter(nn.Module):
    """Learn a residual from concatenated graph observation and raw prediction.

    Hidden layers never compress the concatenated input; only the output maps
    back to the navigation dimension. There is no gate or feature subtraction.
    """

    def __init__(self, input_dim=768, hidden_dim=1536, zero_init=True, alpha=1.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.hidden_dim < 2 * self.input_dim:
            raise ValueError("hidden_dim must be >= 2 * input_dim (no bottleneck)")
        if not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite")
        self.residual = _ConcatResidualMlp(
            self.input_dim, self.hidden_dim, bool(zero_init)
        )

    def forward(self, observed, pred_raw):
        if not torch.is_tensor(observed) or not torch.is_tensor(pred_raw):
            raise TypeError("observed and pred_raw must be tensors")
        if observed.ndim != 2 or pred_raw.ndim != 2:
            raise ValueError("observed and pred_raw must have shape [N, D]")
        if observed.shape != pred_raw.shape:
            raise ValueError("observed and pred_raw must have the same shape")
        if observed.shape[-1] != self.input_dim:
            raise ValueError("feature dimension must match input_dim")
        if observed.device != pred_raw.device or observed.dtype != pred_raw.dtype:
            raise ValueError("observed and pred_raw must have the same device/dtype")

        valid = torch.isfinite(observed).all(-1) & torch.isfinite(pred_raw).all(-1)
        if self.alpha == 0.0:
            # Multiplying NaN by zero would not disable injection.
            return observed, {
                "valid_mask": valid.detach(),
                "fusion_delta_norm": observed.new_zeros(observed.shape[0]),
            }
        safe_observed = torch.where(valid[:, None], observed, torch.zeros_like(observed))
        safe_prediction = torch.where(valid[:, None], pred_raw, torch.zeros_like(pred_raw))
        delta = self.alpha * self.residual(torch.cat([safe_observed, safe_prediction], -1))
        proposed = safe_observed + delta
        valid = valid & torch.isfinite(proposed).all(-1)
        fused = torch.where(valid[:, None], proposed, observed)
        safe_delta = torch.where(valid[:, None], delta, torch.zeros_like(delta))
        return fused, {
            "valid_mask": valid.detach(),
            "fusion_delta_norm": safe_delta.detach().float().norm(dim=-1),
        }


def _empty_diagnostics(features):
    zero = features.new_zeros((), dtype=torch.float64)
    return {name: zero.clone() for name in (
        "eligible_candidate_count", "fused_candidate_count",
        "invalid_prediction_count", "fusion_delta_norm_sum",
        "fusion_delta_norm_square_sum", "fusion_delta_norm_count",
        "persistent_writeback_count", "persistent_retained_state_count",
        "persistent_state_norm_sum", "persistent_state_norm_count",
    )}


def apply_ghost_concat_to_graph(nav_inputs, prediction, adapter, gmaps=None):
    """Return graph inputs and optionally commit valid residuals to live ghosts.

    CPU metadata identifies rows across all environments. GPU gathers, masks,
    one adapter forward, and an out-of-place index_copy process them together.
    Persistent writes use one batched validity transfer to CPU, with no
    per-ghost scalar synchronization. No raw prediction cache is retained.
    Diagnostics are device scalars; the caller decides when to synchronize.
    """
    features = nav_inputs["gmap_img_fts"]
    if features.ndim != 3:
        raise ValueError("gmap_img_fts must have shape [B, L, D]")
    diagnostics = _empty_diagnostics(features)
    persistent = {}
    if gmaps is not None:
        if len(gmaps) != features.shape[0]:
            raise ValueError("gmaps length must match graph batch size")
        persistent = {i: graph for i, graph in enumerate(gmaps)
                      if graph.ghost_concat_memory_mode == "persistent_node_state"}
    retained = sum(len(graph.ghost_concat_state_vps) for graph in persistent.values())
    diagnostics["persistent_retained_state_count"] = features.new_tensor(retained).detach()
    pred_raw = getattr(prediction, "pred_cls_raw", None)
    if pred_raw is None:
        return nav_inputs, diagnostics
    if not torch.is_tensor(pred_raw):
        pred_raw = torch.as_tensor(pred_raw)
    records = list((getattr(prediction, "meta", None) or {}).get("records", []))
    if pred_raw.ndim != 2 or pred_raw.shape[-1] != features.shape[-1]:
        raise ValueError("pred_cls_raw must have shape [N, graph feature dimension]")
    if pred_raw.shape[0] != len(records):
        raise ValueError("prediction row count does not match records")

    batch_size, graph_len, feature_dim = features.shape
    vp_ids = nav_inputs["gmap_vp_ids"]
    if len(vp_ids) != batch_size:
        raise ValueError("gmap_vp_ids length must match graph batch size")
    graph_lookup = {}
    for env_index, ids in enumerate(vp_ids):
        if len(ids) > graph_len:
            raise ValueError("gmap_vp_ids length exceeds padded graph length")
        for col, vp in enumerate(ids):
            if vp is None or not str(vp).startswith("g"):
                continue
            key = (env_index, str(vp))
            if key in graph_lookup:
                raise ValueError("Duplicate ghost id in graph: %s" % (key,))
            graph_lookup[key] = env_index * graph_len + col

    graph_rows, prediction_rows, matched_keys, seen = [], [], [], set()
    for row, record in enumerate(records):
        key = (int(record.env_index), str(record.ghost_vp))
        if key in seen:
            raise ValueError("Duplicate RAE-NWM prediction record for %s" % (key,))
        seen.add(key)
        if key in graph_lookup:
            graph_rows.append(graph_lookup[key])
            prediction_rows.append(row)
            matched_keys.append(key)
    if not graph_rows:
        return nav_inputs, diagnostics

    masks = nav_inputs["gmap_masks"]
    visited = nav_inputs["gmap_visited_masks"]
    if masks.shape != features.shape[:2] or visited.shape != features.shape[:2]:
        raise ValueError("graph masks must have shape [B, L]")
    # Transfer index vectors once; never inspect CUDA masks from Python.
    graph_index = torch.tensor(graph_rows, device=features.device, dtype=torch.long)
    pred_index = torch.tensor(prediction_rows, device=pred_raw.device, dtype=torch.long)
    predicted = pred_raw.index_select(0, pred_index).to(device=features.device, dtype=features.dtype)
    flat_features = features.reshape(-1, feature_dim)
    observed = flat_features.index_select(0, graph_index)
    eligible = masks.reshape(-1).index_select(0, graph_index).bool()
    eligible = eligible & ~visited.reshape(-1).index_select(0, graph_index).bool()
    # Mask before the MLP as well as after it: invalid/padded rows must not
    # contribute non-finite activations or gradients to shared parameters.
    safe_observed = torch.where(eligible[:, None], observed, torch.zeros_like(observed))
    safe_predicted = torch.where(eligible[:, None], predicted, torch.zeros_like(predicted))
    fused, row_diagnostics = adapter(safe_observed, safe_predicted)
    valid = eligible & row_diagnostics["valid_mask"]
    # Scaling by observation count must also remain finite in the accumulator.
    counts = [persistent[e].ghost_embeds[g][1]
              if e in persistent and g in persistent[e].ghost_embeds else 1
              for e, g in matched_keys]
    if persistent:
        scaled = fused * fused.new_tensor(counts)[:, None]
        valid = valid & torch.isfinite(scaled).all(-1)
    replacements = torch.where(valid[:, None], fused, observed)
    output = dict(nav_inputs)
    output["gmap_img_fts"] = flat_features.index_copy(0, graph_index, replacements).reshape_as(features)
    applied = valid if adapter.alpha != 0.0 else torch.zeros_like(valid)
    if persistent and adapter.alpha != 0.0:
        # This is the only GPU-to-CPU synchronization needed for dictionary writes.
        write_flags = applied.detach().cpu().tolist()
        written, previously_retained, state_norms = 0, 0, []
        for row, ((env, vp), should_write) in enumerate(zip(matched_keys, write_flags)):
            graph = persistent.get(env)
            if not should_write or graph is None or vp not in graph.ghost_embeds:
                continue
            previously_retained += int(vp in graph.ghost_concat_state_vps)
            graph.write_ghost_concat_state(vp, fused[row], validated=True)
            state_norms.append(fused[row].detach().float().norm().double())
            written += 1
        diagnostics["persistent_writeback_count"] = features.new_tensor(written).detach()
        diagnostics["persistent_retained_state_count"] = features.new_tensor(
            retained - previously_retained).detach()
        diagnostics["persistent_state_norm_count"] = features.new_tensor(written).detach()
        if state_norms:
            diagnostics["persistent_state_norm_sum"] = torch.stack(state_norms).sum()
    norms = torch.where(applied, row_diagnostics["fusion_delta_norm"], 0).double()
    diagnostics.update({
        "eligible_candidate_count": eligible.sum().detach(),
        "fused_candidate_count": applied.sum().detach(),
        "invalid_prediction_count": (eligible & ~valid).sum().detach(),
        "fusion_delta_norm_sum": norms.sum(),
        "fusion_delta_norm_square_sum": norms.square().sum(),
        "fusion_delta_norm_count": applied.sum().detach(),
    })
    return output, diagnostics
