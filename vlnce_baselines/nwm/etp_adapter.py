import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from vlnce_baselines.nwm.types import NwmCondition

_RAE_IMAGE_TRANSFORM = None
_OPTIONAL_RAE_TRANSFORM_DEPS = {"torchvision", "matplotlib"}


def _wrap_to_pi(angle: float) -> float:
    return ((float(angle) + math.pi) % (2.0 * math.pi)) - math.pi


def _as_position3(position) -> np.ndarray:
    arr = np.asarray(position, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"position must have 3 values [x, y, z], got shape {arr.shape}")
    return arr.copy()


def _as_env_index(env_index) -> int:
    if type(env_index) is int or isinstance(env_index, np.integer):
        return int(env_index)
    raise TypeError(f"env_index must be an int, got {type(env_index).__name__}")


def _as_rgb_uint8(rgb) -> np.ndarray:
    if torch.is_tensor(rgb):
        value = rgb.detach().cpu()
        if value.ndim == 3 and value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        arr = value.numpy()
    else:
        arr = np.asarray(rgb)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"rgb must be HWC or CHW image, got shape {arr.shape}")
    if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"rgb must have 3 channels, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if arr.max(initial=0.0) <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _rgb_to_rae_tensor(rgb: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(_as_rgb_uint8(rgb))
    return _get_rae_image_transform()(image)


def _as_context_latent(latent) -> torch.Tensor:
    if not torch.is_tensor(latent):
        raise TypeError(f"latent must be a torch.Tensor, got {type(latent).__name__}")
    if latent.ndim == 3:
        if tuple(latent.shape) != (768, 16, 16):
            raise ValueError(
                "patch latent must have shape [768,16,16], got "
                f"{tuple(latent.shape)}"
            )
    elif latent.ndim == 2:
        if tuple(latent.shape) != (257, 768):
            raise ValueError(
                "native latent must have shape [257,768], got "
                f"{tuple(latent.shape)}"
            )
    else:
        raise ValueError(
            "latent must have shape [768,16,16] or [257,768], got "
            f"{tuple(latent.shape)}"
        )
    return latent.detach().clone().to(dtype=torch.float32)


def _get_rae_image_transform():
    global _RAE_IMAGE_TRANSFORM
    if _RAE_IMAGE_TRANSFORM is not None:
        return _RAE_IMAGE_TRANSFORM

    try:
        from vlnce_baselines.nwm.raenwm_core.misc import transform as rae_image_transform
    except ModuleNotFoundError as exc:
        if not _should_fallback_rae_transform(exc):
            raise
        _RAE_IMAGE_TRANSFORM = _fallback_rae_image_transform
    else:
        _RAE_IMAGE_TRANSFORM = rae_image_transform
    return _RAE_IMAGE_TRANSFORM


def _should_fallback_rae_transform(exc: ModuleNotFoundError) -> bool:
    missing_name = getattr(exc, "name", None)
    if missing_name == "vlnce_baselines.nwm.raenwm_core.misc":
        return True
    if missing_name in _OPTIONAL_RAE_TRANSFORM_DEPS:
        return True
    return False


def _fallback_rae_image_transform(image: Image.Image) -> torch.Tensor:
    from vlnce_baselines.common.rae_visual_encoder import (
        prepare_rae_rgb_tensor_normalized,
    )

    return prepare_rae_rgb_tensor_normalized(
        image,
        device=torch.device("cpu"),
        size=224,
    )[0]


@dataclass(frozen=True)
class RaeEtpAdapterConfig:
    context_size: int = 4
    max_buffer_size: int = 64
    image_size: int = 224
    metric_waypoint_spacing: float = 0.24975892673356762
    action_min: Sequence[float] = (-64.0, -64.0)
    action_max: Sequence[float] = (64.0, 64.0)
    pos_eps: float = 1.0e-3
    yaw_eps: float = 1.0e-3
    static_run_k: int = 3
    min_horizon: float = 1.0
    max_horizon: float = 64.0

    def __post_init__(self):
        for name in ("context_size", "max_buffer_size", "image_size", "static_run_k"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an int")
        for name in ("metric_waypoint_spacing", "min_horizon", "max_horizon", "pos_eps", "yaw_eps"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.context_size <= 0:
            raise ValueError("context_size must be > 0")
        if self.max_buffer_size < self.context_size:
            raise ValueError("max_buffer_size must be >= context_size")
        if self.image_size != 224:
            raise ValueError("image_size must be 224 to match the RAE training transform")
        if self.metric_waypoint_spacing <= 0:
            raise ValueError("metric_waypoint_spacing must be > 0")
        if len(self.action_min) != 2 or len(self.action_max) != 2:
            raise ValueError("action_min and action_max must each contain 2 values")
        for idx, (min_value, max_value) in enumerate(zip(self.action_min, self.action_max)):
            if not math.isfinite(float(min_value)) or not math.isfinite(float(max_value)):
                raise ValueError("action_min and action_max values must be finite")
            if float(max_value) <= float(min_value):
                raise ValueError(f"action_max[{idx}] must be > action_min[{idx}]")
        if self.pos_eps < 0:
            raise ValueError("pos_eps must be >= 0")
        if self.yaw_eps < 0:
            raise ValueError("yaw_eps must be >= 0")
        if self.static_run_k <= 0:
            raise ValueError("static_run_k must be > 0")
        if self.min_horizon <= 0:
            raise ValueError("min_horizon must be > 0")
        if self.max_horizon < self.min_horizon:
            raise ValueError("max_horizon must be >= min_horizon")


@dataclass
class RaeContextFrame:
    rgb: Optional[np.ndarray]
    position: np.ndarray
    yaw: float
    latent: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class RaeSourceContextSnapshot:
    """Episode-local four-frame latent context for one persistent q0 source."""

    source_front_vp: str
    source_high_level_step: int
    context_latents: torch.Tensor
    source_position: np.ndarray
    source_yaw: float


@dataclass(frozen=True)
class RaeLatentTargetRequest:
    """One latent-context NWM target used by predicted active lookahead."""

    env_index: int
    ghost_vp: str
    snapshot: RaeSourceContextSnapshot
    target_position: np.ndarray
    target_yaw: Optional[float] = None
    horizon_override: Optional[float] = None


@dataclass
class RaeContextBufferStats:
    pushed_frames: int = 0
    kept_frames: int = 0
    dropped_static_frames: int = 0
    dropped_static_runs: int = 0


class RaeContextBuffer:
    def __init__(
        self,
        context_size: int = 4,
        max_size: int = 64,
        pos_eps: float = 1.0e-3,
        yaw_eps: float = 1.0e-3,
        static_run_k: int = 3,
    ):
        self.context_size = int(context_size)
        self.max_size = int(max_size)
        self.pos_eps = float(pos_eps)
        self.yaw_eps = float(yaw_eps)
        self.static_run_k = int(static_run_k)
        self.frames: List[RaeContextFrame] = []
        self._static_candidates: List[RaeContextFrame] = []
        self._last_raw_frame: Optional[RaeContextFrame] = None
        self._dropping_static_run = False
        self.stats = RaeContextBufferStats()

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    def clear(self) -> None:
        self.frames.clear()
        self._static_candidates.clear()
        self._last_raw_frame = None
        self._dropping_static_run = False
        self.stats = RaeContextBufferStats()

    def _trim(self) -> None:
        if len(self.frames) > self.max_size:
            del self.frames[: len(self.frames) - self.max_size]

    def _is_static_against_previous_raw_frame(self, frame: RaeContextFrame) -> bool:
        if self._last_raw_frame is None:
            return False
        prev = self._last_raw_frame
        delta_pos = float(np.linalg.norm(frame.position[[0, 2]] - prev.position[[0, 2]]))
        delta_yaw = abs(_wrap_to_pi(frame.yaw - prev.yaw))
        return delta_pos < self.pos_eps and delta_yaw < self.yaw_eps

    def push(self, rgb, position, yaw: float, latent=None) -> bool:
        if rgb is None and latent is None:
            raise ValueError("Either rgb or latent must be provided for a context frame")
        frame = RaeContextFrame(
            rgb=None if rgb is None else _as_rgb_uint8(rgb),
            position=_as_position3(position),
            yaw=_wrap_to_pi(float(yaw)),
            latent=None if latent is None else _as_context_latent(latent),
        )
        self.stats.pushed_frames += 1

        if self._is_static_against_previous_raw_frame(frame):
            self._last_raw_frame = frame
            if self._dropping_static_run:
                self.stats.dropped_static_frames += 1
                return False
            self._static_candidates.append(frame)
            if len(self._static_candidates) >= self.static_run_k:
                self.stats.dropped_static_frames += len(self._static_candidates)
                self.stats.dropped_static_runs += 1
                self._static_candidates.clear()
                self._dropping_static_run = True
            return False

        self._last_raw_frame = frame
        self._dropping_static_run = False
        if self._static_candidates:
            self.frames.extend(self._static_candidates)
            self._static_candidates.clear()
        self.frames.append(frame)
        self._trim()
        self.stats.kept_frames = len(self.frames)
        return True

    def is_ready(self) -> bool:
        return len(self.frames) >= self.context_size

    def get_context(self) -> List[RaeContextFrame]:
        if not self.is_ready():
            return []
        return list(self.frames[-self.context_size :])


@dataclass(frozen=True)
class RaeGhostInputRequest:
    env_index: int
    ghost_vp: str
    current_position: np.ndarray
    current_yaw: float
    ghost_position: np.ndarray

    @property
    def query_id(self) -> str:
        return self.ghost_vp


@dataclass(frozen=True)
class RaeGhostInputRecord:
    env_index: int
    ghost_vp: str
    local_dx_m: float
    local_dy_m: float
    distance_m: float
    horizon: float
    condition: NwmCondition

    @property
    def query_id(self) -> str:
        return self.ghost_vp


@dataclass
class RaeNwmInputBatch:
    context: torch.Tensor
    curr_delta: torch.Tensor
    rel_t: torch.Tensor
    condition_tensor: torch.Tensor
    context_latent: Optional[torch.Tensor] = None
    records: List[RaeGhostInputRecord] = field(default_factory=list)
    skipped: Dict[str, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.records) == 0


class NwmEtpAdapter:
    def __init__(self, config: Optional[RaeEtpAdapterConfig] = None):
        self.config = config or RaeEtpAdapterConfig()
        self.buffers: List[RaeContextBuffer] = []
        self.last_batch: Optional[RaeNwmInputBatch] = None

    def reset(self, num_envs: int) -> None:
        self.buffers = [
            RaeContextBuffer(
                context_size=self.config.context_size,
                max_size=self.config.max_buffer_size,
                pos_eps=self.config.pos_eps,
                yaw_eps=self.config.yaw_eps,
                static_run_k=self.config.static_run_k,
            )
            for _ in range(int(num_envs))
        ]
        self.last_batch = None

    def ensure_num_envs(self, num_envs: int) -> None:
        if len(self.buffers) != int(num_envs):
            self.reset(int(num_envs))

    def _require_env_index(self, env_index: int) -> int:
        index = _as_env_index(env_index)
        if index < 0 or index >= len(self.buffers):
            raise IndexError(f"env_index {index} is out of range for {len(self.buffers)} buffers")
        return index

    def pause_at(self, env_index: int) -> None:
        del self.buffers[self._require_env_index(env_index)]

    def update_context(self, env_index: int, rgb, position, yaw: float, latent=None) -> bool:
        index = self._require_env_index(env_index)
        return self.buffers[index].push(
            rgb=rgb,
            position=position,
            yaw=yaw,
            latent=latent,
        )

    def source_context_snapshot(
        self,
        env_index: int,
        *,
        source_front_vp: str,
        source_high_level_step: int,
    ) -> Optional[RaeSourceContextSnapshot]:
        """Copy four normalized latent frames to immutable CPU storage."""

        index = self._require_env_index(env_index)
        frames = self.buffers[index].get_context()
        if len(frames) != self.config.context_size:
            return None
        if any(frame.latent is None for frame in frames):
            return None
        context = torch.stack(
            [frame.latent.detach().to(device="cpu", dtype=torch.float32) for frame in frames],
            dim=0,
        ).contiguous()
        if context.ndim not in (3, 4) or int(context.shape[0]) != self.config.context_size:
            raise ValueError(
                "source latent context must have shape [T,C,H,W] or [T,L,C], "
                f"got {tuple(context.shape)}"
            )
        if not bool(torch.isfinite(context).all()):
            return None
        latest = frames[-1]
        return RaeSourceContextSnapshot(
            source_front_vp=str(source_front_vp),
            source_high_level_step=int(source_high_level_step),
            context_latents=context,
            source_position=_as_position3(latest.position),
            source_yaw=_wrap_to_pi(float(latest.yaw)),
        )

    def _local_displacement_m(self, current_position, current_yaw: float, ghost_position):
        cur = _as_position3(current_position)
        ghost = _as_position3(ghost_position)
        delta_x = float(ghost[0] - cur[0])
        delta_z = float(ghost[2] - cur[2])
        yaw = float(current_yaw)

        forward_world = -delta_z
        left_world = -delta_x
        local_forward = math.cos(yaw) * forward_world + math.sin(yaw) * left_world
        local_left = -math.sin(yaw) * forward_world + math.cos(yaw) * left_world
        distance = float(math.hypot(delta_x, delta_z))
        heading = math.atan2(left_world, forward_world)
        dtheta = _wrap_to_pi(heading - yaw)
        return local_forward, local_left, distance, dtheta

    def _normalize_xy(self, dx_m: float, dy_m: float):
        spacing = float(self.config.metric_waypoint_spacing)
        scaled = np.asarray([dx_m / spacing, dy_m / spacing], dtype=np.float32)
        mins = np.asarray(self.config.action_min, dtype=np.float32)
        maxs = np.asarray(self.config.action_max, dtype=np.float32)
        normalized = ((scaled - mins) / (maxs - mins)) * 2.0 - 1.0
        return float(normalized[0]), float(normalized[1])

    def _horizon_from_distance(self, distance_m: float) -> float:
        raw = float(distance_m) / float(self.config.metric_waypoint_spacing)
        return float(np.clip(raw, self.config.min_horizon, self.config.max_horizon))

    def build_ghost_condition(
        self,
        env_index: int,
        ghost_vp: str,
        current_position,
        current_yaw: float,
        ghost_position,
    ) -> RaeGhostInputRecord:
        dx_m, dy_m, distance_m, dtheta = self._local_displacement_m(
            current_position=current_position,
            current_yaw=current_yaw,
            ghost_position=ghost_position,
        )
        norm_dx, norm_dy = self._normalize_xy(dx_m, dy_m)
        horizon = self._horizon_from_distance(distance_m)
        condition = NwmCondition(
            dx=norm_dx,
            dy=norm_dy,
            dtheta=dtheta,
            rel_t=horizon / 128.0,
        )
        return RaeGhostInputRecord(
            env_index=_as_env_index(env_index),
            ghost_vp=str(ghost_vp),
            local_dx_m=float(dx_m),
            local_dy_m=float(dy_m),
            distance_m=float(distance_m),
            horizon=float(horizon),
            condition=condition,
        )

    def build_target_record(
        self,
        *,
        env_index: int,
        ghost_vp: str,
        source_position,
        source_yaw: float,
        target_position,
        target_yaw: Optional[float] = None,
        horizon_override: Optional[float] = None,
    ) -> RaeGhostInputRecord:
        """Build a condition for a historical context and explicit target."""

        dx_m, dy_m, distance_m, bearing_delta = self._local_displacement_m(
            current_position=source_position,
            current_yaw=source_yaw,
            ghost_position=target_position,
        )
        norm_dx, norm_dy = self._normalize_xy(dx_m, dy_m)
        horizon = (
            self._horizon_from_distance(distance_m)
            if horizon_override is None
            else float(
                np.clip(
                    float(horizon_override),
                    self.config.min_horizon,
                    self.config.max_horizon,
                )
            )
        )
        dtheta = (
            bearing_delta
            if target_yaw is None
            else _wrap_to_pi(float(target_yaw) - float(source_yaw))
        )
        condition = NwmCondition(
            dx=norm_dx,
            dy=norm_dy,
            dtheta=dtheta,
            rel_t=horizon / 128.0,
        )
        return RaeGhostInputRecord(
            env_index=_as_env_index(env_index),
            ghost_vp=str(ghost_vp),
            local_dx_m=float(dx_m),
            local_dy_m=float(dy_m),
            distance_m=float(distance_m),
            horizon=float(horizon),
            condition=condition,
        )

    def build_raenwm_latent_batch(
        self,
        requests: Sequence[RaeLatentTargetRequest],
        device="cpu",
    ) -> RaeNwmInputBatch:
        """Batch heterogeneous historical latent contexts for frozen inference."""

        device = torch.device(device)
        if not requests:
            return self._empty_batch(device=device)
        contexts = []
        records = []
        for request in requests:
            snapshot = request.snapshot
            context = snapshot.context_latents
            if not torch.is_tensor(context) or context.ndim not in (3, 4):
                raise ValueError(
                    "latent target context must have shape [T,C,H,W] or [T,L,C]"
                )
            if int(context.shape[0]) != self.config.context_size:
                raise ValueError(
                    "latent target context length differs from NWM context size: "
                    f"{context.shape[0]} vs {self.config.context_size}"
                )
            if not bool(torch.isfinite(context).all()):
                raise ValueError("latent target context contains non-finite values")
            contexts.append(context.to(dtype=torch.float32))
            records.append(
                self.build_target_record(
                    env_index=request.env_index,
                    ghost_vp=request.ghost_vp,
                    source_position=snapshot.source_position,
                    source_yaw=snapshot.source_yaw,
                    target_position=request.target_position,
                    target_yaw=request.target_yaw,
                    horizon_override=request.horizon_override,
                )
            )

        condition_values = [
            [
                record.condition.dx,
                record.condition.dy,
                record.condition.dtheta,
                record.condition.rel_t,
            ]
            for record in records
        ]
        return RaeNwmInputBatch(
            context=torch.empty(
                (0, self.config.context_size, 3, self.config.image_size, self.config.image_size),
                dtype=torch.float32,
                device=device,
            ),
            curr_delta=torch.as_tensor(
                [values[:3] for values in condition_values],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(1),
            rel_t=torch.as_tensor(
                [values[3] for values in condition_values],
                dtype=torch.float32,
                device=device,
            ),
            condition_tensor=torch.as_tensor(
                condition_values, dtype=torch.float32, device=device
            ),
            context_latent=torch.stack(contexts, dim=0).to(device=device),
            records=records,
            skipped={},
        )

    def _empty_batch(self, device, skipped: Optional[Dict[str, int]] = None) -> RaeNwmInputBatch:
        context_shape = (0, self.config.context_size, 3, self.config.image_size, self.config.image_size)
        return RaeNwmInputBatch(
            context=torch.empty(context_shape, dtype=torch.float32, device=device),
            curr_delta=torch.empty((0, 1, 3), dtype=torch.float32, device=device),
            rel_t=torch.empty((0,), dtype=torch.float32, device=device),
            condition_tensor=torch.empty((0, 4), dtype=torch.float32, device=device),
            context_latent=None,
            records=[],
            skipped=skipped or {},
        )

    def build_raenwm_batch(self, requests: Sequence[RaeGhostInputRequest], device="cpu") -> RaeNwmInputBatch:
        device = torch.device(device)
        curr_delta_values = []
        rel_t_values = []
        condition_values = []
        prepared_entries = []
        records: List[RaeGhostInputRecord] = []
        skipped: Dict[str, int] = {}

        for request in requests:
            env_index = _as_env_index(request.env_index)
            if env_index < 0 or env_index >= len(self.buffers):
                skipped["missing_buffer"] = skipped.get("missing_buffer", 0) + 1
                continue

            context_frames = self.buffers[env_index].get_context()
            if len(context_frames) != self.config.context_size:
                skipped["context_not_ready"] = skipped.get("context_not_ready", 0) + 1
                continue

            record = self.build_ghost_condition(
                env_index=env_index,
                ghost_vp=request.ghost_vp,
                current_position=request.current_position,
                current_yaw=request.current_yaw,
                ghost_position=request.ghost_position,
            )
            prepared_entries.append((env_index, context_frames, record))
            curr_delta_values.append([record.condition.dx, record.condition.dy, record.condition.dtheta])
            rel_t_values.append(record.condition.rel_t)
            condition_values.append(
                [
                    record.condition.dx,
                    record.condition.dy,
                    record.condition.dtheta,
                    record.condition.rel_t,
                ]
            )
            records.append(record)

        if not records:
            batch = self._empty_batch(device=device, skipped=skipped)
            self.last_batch = batch
            return batch

        use_latent_context = all(
            all(frame.latent is not None for frame in context_frames)
            for _env_index, context_frames, _record in prepared_entries
        )
        context_cache = {}
        context_values = []
        for env_index, context_frames, _record in prepared_entries:
            if env_index not in context_cache:
                if use_latent_context:
                    context_cache[env_index] = torch.stack(
                        [frame.latent for frame in context_frames],
                        dim=0,
                    )
                else:
                    if any(frame.rgb is None for frame in context_frames):
                        skipped["missing_rgb_context"] = skipped.get(
                            "missing_rgb_context",
                            0,
                        ) + 1
                        continue
                    context_cache[env_index] = torch.stack(
                        [_rgb_to_rae_tensor(frame.rgb) for frame in context_frames],
                        dim=0,
                    )
            context_values.append(context_cache[env_index])

        if len(context_values) != len(records):
            batch = self._empty_batch(device=device, skipped=skipped)
            self.last_batch = batch
            return batch

        empty_context_shape = (
            0,
            self.config.context_size,
            3,
            self.config.image_size,
            self.config.image_size,
        )
        context = torch.empty(empty_context_shape, dtype=torch.float32, device=device)
        context_latent = None
        if use_latent_context:
            context_latent = torch.stack(context_values, dim=0).to(device=device)
        else:
            context = torch.stack(context_values, dim=0).to(device=device)

        batch = RaeNwmInputBatch(
            context=context,
            curr_delta=torch.as_tensor(curr_delta_values, dtype=torch.float32, device=device).unsqueeze(1),
            rel_t=torch.as_tensor(rel_t_values, dtype=torch.float32, device=device),
            condition_tensor=torch.as_tensor(condition_values, dtype=torch.float32, device=device),
            context_latent=context_latent,
            records=records,
            skipped=skipped,
        )
        self.last_batch = batch
        return batch
