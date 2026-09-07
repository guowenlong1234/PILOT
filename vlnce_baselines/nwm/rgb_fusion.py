import torch
import torch.nn as nn
import torch.nn.functional as F


_GHOST_TARGET_KINDS = {"new_ghost", "existing_ghost"}
_FUSION_DIAGNOSTIC_NAMES = (
    "gate",
    "raw_wm_cosine",
    "fusion_delta_norm",
)


def clone_wp_outputs_candidate_rgb(wp_outputs):
    cloned = dict(wp_outputs)
    cloned["cand_rgb"] = [value.clone() for value in wp_outputs["cand_rgb"]]
    return cloned


class _ResidualMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, zero_init: bool):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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


class RaeNwmRgbFusionAdapter(nn.Module):
    """Fuse predicted and observed RGB CLS features with a gated residual."""

    def __init__(
        self,
        input_dim=768,
        hidden_dim=768,
        zero_init=True,
        alpha=1.0,
        gate_bias_init=-8.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.alpha = float(alpha)
        self.gate_bias_init = float(gate_bias_init)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")

        self.residual = _ResidualMlp(
            self.input_dim, self.hidden_dim, bool(zero_init)
        )
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim * 3 + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        if zero_init:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, self.gate_bias_init)

    def forward(self, raw_rgb, wm_rgb, agreement, distance):
        if not torch.is_tensor(raw_rgb) or not torch.is_tensor(wm_rgb):
            raise TypeError("raw_rgb and wm_rgb must be torch.Tensor values")
        if raw_rgb.ndim != 2 or wm_rgb.ndim != 2:
            raise ValueError("raw_rgb and wm_rgb must have shape [N, D]")
        if raw_rgb.shape != wm_rgb.shape:
            raise ValueError(
                "raw_rgb and wm_rgb must have the same shape, got "
                f"{tuple(raw_rgb.shape)} and {tuple(wm_rgb.shape)}"
            )
        if int(raw_rgb.shape[-1]) != self.input_dim:
            raise ValueError(
                "raw_rgb feature dim must match input_dim: "
                f"{raw_rgb.shape[-1]} vs {self.input_dim}"
            )

        agreement = _as_tensor_like(agreement, raw_rgb).reshape(-1, 1)
        distance = _as_tensor_like(distance, raw_rgb).reshape(-1, 1)
        if agreement.shape[0] != raw_rgb.shape[0]:
            raise ValueError("agreement batch size must match raw_rgb")
        if distance.shape[0] != raw_rgb.shape[0]:
            raise ValueError("distance batch size must match raw_rgb")

        delta = wm_rgb - raw_rgb
        gate_input = torch.cat(
            [raw_rgb, wm_rgb, delta, agreement, distance], dim=-1
        )
        gate = torch.sigmoid(self.gate(gate_input)) * self.alpha
        fused = raw_rgb + gate * self.residual(delta)
        diagnostics = {
            "gate": gate.detach(),
            "raw_wm_cosine": F.cosine_similarity(
                raw_rgb, wm_rgb, dim=-1
            ).detach(),
            "fusion_delta_norm": (fused - raw_rgb).norm(dim=-1).detach(),
        }
        return fused, diagnostics


def _as_tensor_like(value, reference):
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(
        value, dtype=reference.dtype, device=reference.device
    )


def _row_count(value):
    if torch.is_tensor(value):
        return 1 if value.ndim == 0 else int(value.shape[0])
    return len(value)


def _prediction_lookup(prediction):
    pred_cls_raw = getattr(prediction, "pred_cls_raw", None)
    native_cls = pred_cls_raw is not None
    pred_cls = pred_cls_raw if native_cls else getattr(prediction, "pred_cls", None)
    confidence = getattr(prediction, "confidence", None)
    if prediction is None or pred_cls is None:
        return {}
    if not native_cls and confidence is None:
        return {}

    records = list((getattr(prediction, "meta", None) or {}).get("records", []))
    if len(records) != _row_count(pred_cls):
        raise ValueError(
            "RAE-NWM prediction row count does not match records: "
            f"{_row_count(pred_cls)} vs {len(records)}"
        )
    if not native_cls and len(records) != _row_count(confidence):
        raise ValueError(
            "RAE-NWM confidence row count does not match records: "
            f"{_row_count(confidence)} vs {len(records)}"
        )

    lookup = {}
    for row_index, record in enumerate(records):
        key = (int(record.env_index), str(record.ghost_vp))
        if key in lookup:
            raise ValueError(f"Duplicate RAE-NWM prediction record for {key}")
        lookup[key] = (
            pred_cls[row_index],
            None if native_cls else confidence[row_index],
            float(getattr(record, "distance_m", 0.0)),
            native_cls,
        )
    return lookup


def apply_rgb_fusion_to_current_candidates(
    wp_outputs, candidate_previews, prediction, fusion_adapter,
    prediction_transform=None,
):
    cand_rgbs = wp_outputs["cand_rgb"]
    if len(candidate_previews) != len(cand_rgbs):
        raise ValueError(
            "candidate_previews length must match cand_rgb env count: "
            f"{len(candidate_previews)} vs {len(cand_rgbs)}"
        )

    lookup = _prediction_lookup(prediction)
    diagnostics = []
    for env_index, previews in enumerate(candidate_previews):
        cand_rgb = cand_rgbs[env_index]
        if not torch.is_tensor(cand_rgb):
            raise TypeError(
                "wp_outputs['cand_rgb'] entries must be torch.Tensor values"
            )
        if len(previews) != int(cand_rgb.shape[0]):
            raise ValueError(
                "candidate preview count must match cand_rgb rows for env "
                f"{env_index}: {len(previews)} vs {cand_rgb.shape[0]}"
            )

        fused_cand_rgb = cand_rgb.clone()
        zero = torch.zeros((), device=cand_rgb.device, dtype=torch.float64)
        env_diagnostics = {
            "eligible_candidate_count": 0,
            "fused_candidate_count": 0,
        }
        for name in _FUSION_DIAGNOSTIC_NAMES:
            env_diagnostics[f"{name}_sum"] = zero.clone()
            env_diagnostics[f"{name}_square_sum"] = zero.clone()
            env_diagnostics[f"{name}_count"] = 0
        fused_count = 0
        for cand_index, preview in enumerate(previews):
            if preview.target_kind not in _GHOST_TARGET_KINDS:
                continue
            env_diagnostics["eligible_candidate_count"] += 1
            key = (env_index, str(preview.target_vp))
            if key not in lookup:
                continue
            wm_rgb, signal, distance_m, native_cls = lookup[key]
            raw_rgb = cand_rgb[cand_index : cand_index + 1]
            wm_rgb = _as_tensor_like(wm_rgb, raw_rgb).reshape_as(raw_rgb)
            if prediction_transform is not None:
                if not native_cls:
                    raise ValueError("Navigation CLS alignment requires raw native CLS predictions")
                wm_rgb = prediction_transform(wm_rgb)
            if native_cls:
                signal = (
                    F.cosine_similarity(raw_rgb, wm_rgb, dim=-1) + 1.0
                ) * 0.5
            fused_rgb, fusion_diagnostics = fusion_adapter(
                raw_rgb,
                wm_rgb,
                _as_tensor_like(signal, raw_rgb).reshape(1),
                _as_tensor_like([distance_m], raw_rgb),
            )
            fused_cand_rgb[cand_index] = fused_rgb[0]
            fused_count += 1
            for name in _FUSION_DIAGNOSTIC_NAMES:
                value = (fusion_diagnostics or {}).get(name)
                if value is None:
                    continue
                value = _as_tensor_like(value, cand_rgb).detach().to(
                    dtype=torch.float64
                ).reshape(-1)
                env_diagnostics[f"{name}_sum"].add_(value.sum())
                env_diagnostics[f"{name}_square_sum"].add_(
                    value.square().sum()
                )
                env_diagnostics[f"{name}_count"] += int(value.numel())
        wp_outputs["cand_rgb"][env_index] = fused_cand_rgb
        env_diagnostics["fused_candidate_count"] = fused_count
        diagnostics.append(env_diagnostics)
    return diagnostics
