"""Frozen DINO-CWP top1 inference and NWM-predicted E24 future tokens."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from vlnce_baselines.nwm.etp_adapter import RaeLatentTargetRequest

from .offline_checkpoint import sha256_file
from .online_e24 import quantize_like_offline_cache


DEFAULT_NONE_THRESHOLD = 0.3
SOURCE_CODE_COMMIT = "1045bbbee7f957511b00aa057cb79844f2748009"
PREDICTED_FUTURE_DIAGNOSTIC_NAMES = (
    "topk_slots", "oracle_q1_requested", "q0_record_present", "q0_context_present",
    "q0_cache_present", "q0_cache_invalid",
    "q0_cache_samples", "q0_cache_live_records_sum", "q0_cache_bytes_sum",
    "q0_first_stage_requested", "q0_first_stage_success",
    "q0_requested", "q0_nwm_success", "q0_batch_failures", "q0_row_failures",
    "cwp_requested", "cwp_invalid", "cwp_none", "cwp_top1",
    "cwp_batch_failures", "cwp_row_failures", "q1_requested",
    "q1_nwm_success", "q1_batch_failures", "q1_row_failures", "future_valid",
    "q0_first_stage_nwm_seconds", "q0_nwm_seconds", "cwp_seconds", "q1_nwm_seconds",
    "q0_latent_rows", "q0_latent_mean_sum", "q0_latent_std_sum", "q0_latent_norm_sum",
    "q1_latent_rows", "q1_latent_mean_sum", "q1_latent_std_sum", "q1_latent_norm_sum",
    "action_rows", "action_flips",
)


def summarize_predicted_future_diagnostics(totals: Mapping[str, float]) -> dict[str, float]:
    """Convert cross-rank predicted-future counters into reportable metrics."""

    values = {
        name: float(totals.get(name, 0.0))
        for name in PREDICTED_FUTURE_DIAGNOSTIC_NAMES
    }
    summary = {
        name: values[name]
        for name in (
            "topk_slots", "oracle_q1_requested", "q0_record_present",
            "q0_context_present", "q0_cache_present", "q0_cache_invalid",
            "q0_first_stage_requested", "q0_first_stage_success",
            "q0_requested", "q0_nwm_success",
            "cwp_requested", "cwp_invalid", "cwp_none", "cwp_top1",
            "cwp_batch_failures", "cwp_row_failures",
            "q1_requested", "q1_nwm_success", "future_valid",
            "q0_batch_failures", "q0_row_failures", "q1_batch_failures",
            "q1_row_failures",
        )
    }
    for name, numerator, denominator in (
        ("q0_record_coverage", "q0_record_present", "topk_slots"),
        ("q0_context_coverage", "q0_context_present", "topk_slots"),
        ("q0_success_rate", "q0_cache_present", "q0_context_present"),
        ("cwp_none_rate", "cwp_none", "cwp_requested"),
        ("q1_success_rate", "q1_nwm_success", "q1_requested"),
        ("valid_rate", "future_valid", "topk_slots"),
        ("action_flip_rate", "action_flips", "action_rows"),
    ):
        summary[name] = values[numerator] / max(1.0, values[denominator])
    for stage in ("q0", "q1"):
        rows = max(1.0, values[f"{stage}_latent_rows"])
        for statistic in ("mean", "std", "norm"):
            summary[f"{stage}_latent_{statistic}"] = (
                values[f"{stage}_latent_{statistic}_sum"] / rows
            )
    for name in (
        "q0_first_stage_nwm_seconds", "q0_nwm_seconds",
        "cwp_seconds", "q1_nwm_seconds",
    ):
        summary[name] = values[name]
    cache_samples = max(1.0, values["q0_cache_samples"])
    summary["q0_cache_live_records_mean"] = (
        values["q0_cache_live_records_sum"] / cache_samples
    )
    summary["q0_cache_mebibytes_mean"] = (
        values["q0_cache_bytes_sum"] / cache_samples / (1024.0 ** 2)
    )
    return summary


class SingleViewDinoCwpPredictor(nn.Module):
    """Inference-compatible copy of dino-cwp's current single-view model."""

    def __init__(
        self,
        *,
        patch_dim: int = 768,
        patch_grid: tuple[int, int] = (16, 16),
        hidden_dim: int = 512,
        num_heads: int = 8,
        encoder_layers: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        angle_bins: int = 30,
        distance_bins: int = 12,
    ) -> None:
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.patch_grid = tuple(int(item) for item in patch_grid)
        self.hidden_dim = int(hidden_dim)
        self.angle_bins = int(angle_bins)
        self.distance_bins = int(distance_bins)
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.input_norm = nn.LayerNorm(self.patch_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.patch_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.image_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(encoder_layers),
            norm=nn.LayerNorm(self.hidden_dim),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.angle_bins + 1, self.hidden_dim)
        )
        self.heatmap_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.distance_bins),
        )
        self.none_head = nn.Sequential(
            nn.Linear(self.hidden_dim, max(1, self.hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(max(1, self.hidden_dim // 2), 1),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def _validate_patch_tokens(self, patch_tokens: torch.Tensor) -> None:
        expected = (self.num_patches, self.patch_dim)
        if patch_tokens.ndim != 3 or tuple(patch_tokens.shape[1:]) != expected:
            raise ValueError(
                f"expected patch_tokens shaped [B,{expected[0]},{expected[1]}], "
                f"got {tuple(patch_tokens.shape)}"
            )

    def forward(self, patch_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        self._validate_patch_tokens(patch_tokens)
        batch = int(patch_tokens.shape[0])
        x = self.input_norm(patch_tokens.float())
        x = self.input_proj(x)
        x = x + self.position_embedding
        encoded = self.image_encoder(x)
        queries = self.query_tokens.expand(batch, -1, -1)
        decoded = self.query_decoder(tgt=queries, memory=encoded)
        angle_features = decoded[:, : self.angle_bins]
        none_feature = decoded[:, self.angle_bins]
        heatmap_logits = self.heatmap_head(angle_features)
        none_logit = self.none_head(none_feature).squeeze(-1)
        return {
            "heatmap_logits": heatmap_logits,
            "none_logit": none_logit,
        }


@dataclass(frozen=True)
class DinoCwpTop1Prediction:
    valid: bool
    pred_none: bool
    none_prob: float
    score: float
    local_angle_idx: int
    distance_idx: int
    local_angle_deg: float
    distance_m: float


def load_dino_cwp_predictor(
    checkpoint_path: str | Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> tuple[SingleViewDinoCwpPredictor, dict[str, Any]]:
    path = Path(checkpoint_path)
    actual_sha = sha256_file(path)
    if actual_sha != str(expected_sha256).strip().lower():
        raise ValueError(
            "DINO-CWP checkpoint SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("DINO-CWP checkpoint must contain model_state")
    model = SingleViewDinoCwpPredictor().to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": actual_sha,
        "source_code_commit": SOURCE_CODE_COMMIT,
        "checkpoint_step": int(payload.get("step", -1)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, metadata


def decode_dino_cwp_top1(
    output: Mapping[str, torch.Tensor],
    *,
    none_threshold: float = DEFAULT_NONE_THRESHOLD,
) -> list[DinoCwpTop1Prediction]:
    heatmap = output["heatmap_logits"].detach().float()
    none_logit = output["none_logit"].detach().float().reshape(-1)
    if heatmap.ndim != 3 or tuple(heatmap.shape[1:]) != (30, 12):
        raise ValueError(f"DINO-CWP heatmap must have shape [B,30,12], got {tuple(heatmap.shape)}")
    if int(heatmap.shape[0]) != int(none_logit.numel()):
        raise ValueError("DINO-CWP heatmap and none batch sizes differ")
    probs = torch.softmax(heatmap.reshape(heatmap.shape[0], -1), dim=1)
    values, indices = probs.max(dim=1)
    none_probs = torch.sigmoid(none_logit)
    result = []
    for row in range(int(heatmap.shape[0])):
        row_finite = bool(torch.isfinite(heatmap[row]).all()) and bool(
            torch.isfinite(none_logit[row])
        )
        if not row_finite:
            result.append(
                DinoCwpTop1Prediction(
                    valid=False,
                    pred_none=False,
                    none_prob=float("nan"),
                    score=float("nan"),
                    local_angle_idx=-1,
                    distance_idx=-1,
                    local_angle_deg=float("nan"),
                    distance_m=float("nan"),
                )
            )
            continue
        flat = int(indices[row].item())
        angle_idx, distance_idx = divmod(flat, 12)
        none_prob = float(none_probs[row].item())
        result.append(
            DinoCwpTop1Prediction(
                valid=True,
                pred_none=bool(none_prob >= float(none_threshold)),
                none_prob=none_prob,
                score=float(values[row].item()),
                local_angle_idx=angle_idx,
                distance_idx=distance_idx,
                local_angle_deg=-45.0 + (angle_idx + 0.5) * 3.0,
                distance_m=(distance_idx + 1) * 0.25,
            )
        )
    return result


def latent_to_patch_tokens(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4 or tuple(latent.shape[1:]) != (768, 16, 16):
        raise ValueError(
            "predicted q0 latent must have shape [B,768,16,16], "
            f"got {tuple(latent.shape)}"
        )
    return latent.permute(0, 2, 3, 1).reshape(latent.shape[0], 256, 768).contiguous()


def _wrap_to_pi(angle: float) -> float:
    return ((float(angle) + math.pi) % (2.0 * math.pi)) - math.pi


def _face_motion_heading_deg(source: Sequence[float], target: Sequence[float], fallback_yaw: float) -> float:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    dx = float(target[0] - source[0])
    dz = float(target[2] - source[2])
    if math.hypot(dx, dz) < 1.0e-6:
        return float(math.degrees(_wrap_to_pi(float(fallback_yaw) - math.pi)))
    return float(math.degrees(math.atan2(dx, dz)))


def waypoint_to_world_position(
    source_position: Sequence[float],
    *,
    heading_deg: float,
    local_angle_deg: float,
    distance_m: float,
) -> np.ndarray:
    source = np.asarray(source_position, dtype=np.float32)
    if source.shape != (3,):
        raise ValueError("waypoint source_position must have shape [3]")
    heading = math.radians(float(heading_deg) + float(local_angle_deg))
    return np.asarray(
        [
            float(source[0] + float(distance_m) * math.sin(heading)),
            float(source[1]),
            float(source[2] + float(distance_m) * math.cos(heading)),
        ],
        dtype=np.float32,
    )


def _add_latent_stats(diagnostics: dict[str, float], prefix: str, latent: torch.Tensor) -> None:
    value = latent.detach().float()
    rows = int(value.shape[0])
    flattened = value.reshape(rows, -1)
    diagnostics[f"{prefix}_latent_rows"] += float(rows)
    diagnostics[f"{prefix}_latent_mean_sum"] += float(flattened.mean(dim=1).sum().cpu())
    diagnostics[f"{prefix}_latent_std_sum"] += float(
        flattened.std(dim=1, unbiased=False).sum().cpu()
    )
    diagnostics[f"{prefix}_latent_norm_sum"] += float(flattened.norm(dim=1).sum().cpu())


def _validated_prediction_rows(
    prediction: Any,
    expected: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    latent = getattr(prediction, "pred_latent", None)
    pred_cls = getattr(prediction, "pred_cls", None)
    pred_tokens = getattr(prediction, "pred_tokens", None)
    if not torch.is_tensor(latent) or not torch.is_tensor(pred_cls):
        return {}
    if tuple(latent.shape) != (expected, 768, 16, 16):
        raise ValueError(
            "NWM predicted latent shape violates active-lookahead contract: "
            f"{tuple(latent.shape)} vs {(expected, 768, 16, 16)}"
        )
    if tuple(pred_cls.shape) != (expected, 768):
        raise ValueError(
            "NWM predicted CLS shape violates active-lookahead contract: "
            f"{tuple(pred_cls.shape)} vs {(expected, 768)}"
        )
    if pred_tokens is not None and (
        not torch.is_tensor(pred_tokens)
        or tuple(pred_tokens.shape) != (expected, 257, 768)
    ):
        shape = tuple(pred_tokens.shape) if torch.is_tensor(pred_tokens) else None
        raise ValueError(
            "NWM predicted token shape violates active-lookahead contract: "
            f"{shape} vs {(expected, 257, 768)}"
        )
    rows = {}
    for row in range(expected):
        token_row = None if pred_tokens is None else pred_tokens[row]
        finite = (
            bool(torch.isfinite(latent[row]).all())
            and bool(torch.isfinite(pred_cls[row]).all())
            and (
                token_row is None
                or bool(torch.isfinite(token_row).all())
            )
        )
        if finite:
            rows[row] = (latent[row], pred_cls[row], token_row)
    return rows


def _predict_nwm_rows(
    trainer: Any,
    requests: Sequence[RaeLatentTargetRequest],
    *,
    stage: str,
    diagnostics: dict[str, float],
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]:
    if not requests:
        return {}
    started = time.perf_counter()
    try:
        prediction = trainer.raenwm_runtime.predict_latent_targets(requests)
        rows = _validated_prediction_rows(prediction, len(requests))
    except (RuntimeError, FloatingPointError, ValueError):
        diagnostics[f"{stage}_batch_failures"] += 1.0
        rows = {}
        for index, request in enumerate(requests):
            try:
                prediction = trainer.raenwm_runtime.predict_latent_targets([request])
                single = _validated_prediction_rows(prediction, 1)
                if 0 in single:
                    rows[index] = single[0]
            except (RuntimeError, FloatingPointError, ValueError):
                diagnostics[f"{stage}_row_failures"] += 1.0
    diagnostics[f"{stage}_nwm_seconds"] += time.perf_counter() - started
    diagnostics[f"{stage}_nwm_success"] += float(len(rows))
    if rows:
        _add_latent_stats(
            diagnostics,
            stage,
            torch.stack([value[0] for _, value in sorted(rows.items())], dim=0),
        )
    return rows


def build_dino_cwp_nwm_future_tokens(
    trainer: Any,
    *,
    active_envs: Sequence[int],
    records_by_env: Sequence[Sequence[Any | None]],
    topk: int,
    feature_dim: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build q1 future tokens without querying a q1 simulator observation."""

    diagnostics: dict[str, float] = {
        key: 0.0 for key in PREDICTED_FUTURE_DIAGNOSTIC_NAMES
    }
    first_stage = getattr(
        trainer, "last_candidate_q0_prediction_diagnostics", None
    ) or {}
    for name in (
        "q0_first_stage_requested",
        "q0_first_stage_success",
        "q0_first_stage_nwm_seconds",
    ):
        diagnostics[name] = float(first_stage.get(name, 0.0))
    live_records = []
    for graph in getattr(trainer, "gmaps", ()):
        live_records.extend(
            getattr(graph, "ghost_candidate_q0", {}).values()
        )
    cache_bytes = sum(
        int(record.predicted_patch_cpu_fp16.numel())
        * int(record.predicted_patch_cpu_fp16.element_size())
        for record in live_records
    )
    unique_contexts = {}
    for record in live_records:
        snapshot = getattr(record, "source_context", None)
        context = getattr(snapshot, "context_latents", None)
        if torch.is_tensor(context):
            unique_contexts.setdefault(id(snapshot), context)
    cache_bytes += sum(
        int(context.numel()) * int(context.element_size())
        for context in unique_contexts.values()
    )
    diagnostics["q0_cache_samples"] = 1.0
    diagnostics["q0_cache_live_records_sum"] = float(len(live_records))
    diagnostics["q0_cache_bytes_sum"] = float(cache_bytes)
    future = reference.new_zeros((len(active_envs), int(topk), 257, int(feature_dim)))
    valid = torch.zeros(
        (len(active_envs), int(topk)), dtype=torch.bool, device=reference.device
    )
    q1_conditions = reference.new_zeros((len(active_envs), int(topk), 4))
    prepared = []
    for row, (env_index, records) in enumerate(zip(active_envs, records_by_env)):
        diagnostics["topk_slots"] += float(len(records))
        for slot, record in enumerate(records):
            if record is None:
                continue
            diagnostics["q0_record_present"] += 1.0
            if getattr(record, "contract_version", None) != (
                "r1_post_update_ghost_mean_cached_v1"
            ):
                diagnostics["q0_cache_invalid"] += 1.0
                continue
            snapshot = getattr(record, "source_context", None)
            patch = getattr(record, "predicted_patch_cpu_fp16", None)
            if snapshot is None:
                diagnostics["q0_cache_invalid"] += 1.0
                continue
            diagnostics["q0_context_present"] += 1.0
            if (
                not torch.is_tensor(patch)
                or patch.device.type != "cpu"
                or patch.dtype != torch.float16
                or tuple(patch.shape) != (768, 16, 16)
                or not bool(torch.isfinite(patch).all())
            ):
                diagnostics["q0_cache_invalid"] += 1.0
                continue
            prepared.append((row, slot, env_index, record, snapshot, patch))
            diagnostics["q0_cache_present"] += 1.0

    # Q0 is deliberately never recomputed here. Missing/invalid cache entries
    # stay invalid so the saved NWM forward is an invariant, not a best effort.
    diagnostics["q0_requested"] = 0.0
    diagnostics["q0_nwm_success"] = 0.0
    if not prepared:
        return future, q1_conditions, valid, diagnostics

    ordered_q0 = list(range(len(prepared)))
    cached_latents = torch.stack(
        [prepared[index][5].float() for index in ordered_q0], dim=0
    )
    _add_latent_stats(diagnostics, "q0", cached_latents)
    patches = cached_latents
    patches = latent_to_patch_tokens(patches)
    diagnostics["cwp_requested"] = float(len(ordered_q0))
    cwp_started = time.perf_counter()
    cwp_predictions = {}
    try:
        with torch.inference_mode():
            cwp_output = trainer.dino_cwp_future_predictor(patches.to(trainer.device))
            decoded = decode_dino_cwp_top1(
                cwp_output,
                none_threshold=float(
                    trainer._active_lookahead_config().dino_cwp_none_threshold
                ),
            )
        cwp_predictions = dict(enumerate(decoded))
    except (RuntimeError, FloatingPointError, ValueError):
        diagnostics["cwp_batch_failures"] += 1.0
        for row in range(int(patches.shape[0])):
            try:
                with torch.inference_mode():
                    output = trainer.dino_cwp_future_predictor(
                        patches[row : row + 1].to(trainer.device)
                    )
                    cwp_predictions[row] = decode_dino_cwp_top1(
                        output,
                        none_threshold=float(
                            trainer._active_lookahead_config().dino_cwp_none_threshold
                        ),
                    )[0]
            except (RuntimeError, FloatingPointError, ValueError):
                diagnostics["cwp_row_failures"] += 1.0
    diagnostics["cwp_seconds"] = time.perf_counter() - cwp_started

    q1_requests = []
    q1_destinations = []
    adapter = trainer.raenwm_runtime.adapter
    spacing = float(adapter.config.metric_waypoint_spacing)
    for cwp_row, q0_index in enumerate(ordered_q0):
        prediction = cwp_predictions.get(cwp_row)
        if prediction is None:
            diagnostics["cwp_invalid"] += 1.0
            continue
        if not prediction.valid:
            diagnostics["cwp_invalid"] += 1.0
            continue
        if prediction.pred_none:
            diagnostics["cwp_none"] += 1.0
            continue
        row, slot, env_index, record, snapshot, _cached_patch = prepared[q0_index]
        q0_position = np.asarray(record.canonical_q0_position, dtype=np.float32)
        q0_heading_deg = _face_motion_heading_deg(
            snapshot.source_position,
            q0_position,
            snapshot.source_yaw,
        )
        q1_position = waypoint_to_world_position(
            q0_position,
            heading_deg=q0_heading_deg,
            local_angle_deg=prediction.local_angle_deg,
            distance_m=prediction.distance_m,
        )
        q1_heading_deg = q0_heading_deg + prediction.local_angle_deg
        q1_yaw = _wrap_to_pi(math.radians(q1_heading_deg) + math.pi)
        q0_record = adapter.build_target_record(
            env_index=env_index,
            ghost_vp=str(record.ghost_vp),
            source_position=snapshot.source_position,
            source_yaw=snapshot.source_yaw,
            target_position=q0_position,
        )
        cumulative_horizon = q0_record.horizon + prediction.distance_m / spacing
        q1_record = adapter.build_target_record(
            env_index=int(env_index),
            ghost_vp=str(record.ghost_vp),
            source_position=snapshot.source_position,
            source_yaw=snapshot.source_yaw,
            target_position=q1_position,
            target_yaw=q1_yaw,
            horizon_override=cumulative_horizon,
        )
        q1_requests.append(
            RaeLatentTargetRequest(
                env_index=int(env_index),
                ghost_vp=str(record.ghost_vp),
                snapshot=snapshot,
                target_position=q1_position,
                target_yaw=q1_yaw,
                horizon_override=cumulative_horizon,
            )
        )
        q1_destinations.append(
            (
                row,
                slot,
                (
                    q1_record.condition.dx,
                    q1_record.condition.dy,
                    q1_record.condition.dtheta,
                    q1_record.condition.rel_t,
                ),
            )
        )
        diagnostics["cwp_top1"] += 1.0

    diagnostics["q1_requested"] = float(len(q1_requests))
    q1_rows = _predict_nwm_rows(
        trainer, q1_requests, stage="q1", diagnostics=diagnostics
    )
    diagnostics["q1_nwm_success"] = float(len(q1_rows))
    if not q1_rows:
        return future, q1_conditions, valid, diagnostics

    encoded_rows = []
    encoded_destinations = []
    for request_index, (pred_latent, pred_cls, pred_tokens) in sorted(q1_rows.items()):
        patch_tokens = latent_to_patch_tokens(pred_latent.unsqueeze(0))[0]
        if pred_tokens is None:
            encoded = torch.cat((pred_cls.reshape(1, 768), patch_tokens), dim=0)
        else:
            if not torch.equal(pred_tokens[1:], patch_tokens):
                raise ValueError(
                    "native NWM token patch and pred_latent patch differ"
                )
            encoded = torch.cat((pred_cls.reshape(1, 768), pred_tokens[1:]), dim=0)
        encoded_rows.append(encoded)
        encoded_destinations.append(q1_destinations[request_index])
    encoded = quantize_like_offline_cache(torch.stack(encoded_rows, dim=0)).to(
        device=reference.device, dtype=reference.dtype
    )
    if tuple(encoded.shape[1:]) != (257, feature_dim):
        raise ValueError(
            "predicted q1 tokens violate E24 contract: "
            f"{tuple(encoded.shape[1:])} vs {(257, feature_dim)}"
        )
    for encoded_row, (row, slot, condition) in enumerate(encoded_destinations):
        future[row, slot] = encoded[encoded_row]
        q1_conditions[row, slot] = reference.new_tensor(condition)
        valid[row, slot] = True
    diagnostics["future_valid"] = float(len(encoded_destinations))
    return future, q1_conditions, valid, diagnostics
