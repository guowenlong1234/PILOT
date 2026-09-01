"""Low-level movement RGB context contract shared by SFT, eval, and GRPO."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch


HIGH_LEVEL_CONTEXT_SOURCE = "high_level_nav_latent"
LOW_LEVEL_CONTEXT_SOURCE = "low_level_move_rgb_anchor"
HIGH_LEVEL_CONTEXT_CONTRACT = "r1_high_level_nav_latent_v1"
LOW_LEVEL_CONTEXT_CONTRACT = "r1_low_level_move_rgb_anchor_v1"
LOW_LEVEL_EVENT_PAYLOAD_FORMAT = "r1_low_level_context_events_v1"
LOW_LEVEL_SAMPLING_ACTION = "MOVE_FORWARD"
LOW_LEVEL_TELEPORT_ANCHOR_POLICY = "clear_then_record_landing_rgb"
LOW_LEVEL_RGB_SENSOR_UUID = "rgb"
CONTEXT_METADATA_FORMAT = "r1_nwm_context_metadata_v1"
LOW_LEVEL_CONTEXT_DIAGNOSTIC_FORMAT = (
    "r1_low_level_context_diagnostics_v1"
)
LOW_LEVEL_CONTEXT_DIAGNOSTIC_NAMES = (
    "drain_count",
    "environment_count",
    "reset_events",
    "frame_events",
    "encoded_frames",
    "encode_batches",
    "encode_seconds",
    "context_ready",
    "context_ready_ratio",
    "dropped_static_frames",
)


def _diagnostic_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def summarize_low_level_context_diagnostics(
    totals: Mapping[str, Any],
) -> Dict[str, float]:
    """Turn cross-drain counter totals into evaluation-friendly metrics."""

    values = {
        name: float(totals.get(name, 0.0))
        for name in LOW_LEVEL_CONTEXT_DIAGNOSTIC_NAMES
    }
    drains = values["drain_count"]
    environments = values["environment_count"]
    frames = values["encoded_frames"]
    batches = values["encode_batches"]
    ready = values["context_ready"]
    return {
        "drain_count": drains,
        "environment_count": environments,
        "mean_active_environments": _diagnostic_ratio(
            environments, drains
        ),
        "reset_events": values["reset_events"],
        "frame_events": values["frame_events"],
        "encoded_frames": frames,
        "encode_batches": batches,
        "encode_seconds": values["encode_seconds"],
        "context_ready": ready,
        "context_ready_ratio": _diagnostic_ratio(ready, environments),
        "mean_drain_context_ready_ratio": _diagnostic_ratio(
            values["context_ready_ratio"], drains
        ),
        "dropped_static_frames": values["dropped_static_frames"],
        "frames_per_drain": _diagnostic_ratio(frames, drains),
        "frames_per_environment": _diagnostic_ratio(
            frames, environments
        ),
        "encode_seconds_per_frame": _diagnostic_ratio(
            values["encode_seconds"], frames
        ),
        "encode_seconds_per_batch": _diagnostic_ratio(
            values["encode_seconds"], batches
        ),
    }


def normalize_context_source(value: Any) -> str:
    source = str(value or HIGH_LEVEL_CONTEXT_SOURCE).strip().lower()
    supported = {HIGH_LEVEL_CONTEXT_SOURCE, LOW_LEVEL_CONTEXT_SOURCE}
    if source not in supported:
        raise ValueError(
            f"Unsupported MODEL.RAENWM.context_source={source!r}; "
            f"expected one of {sorted(supported)}"
        )
    return source


def context_contract_for_source(source: Any) -> str:
    source = normalize_context_source(source)
    if source == LOW_LEVEL_CONTEXT_SOURCE:
        return LOW_LEVEL_CONTEXT_CONTRACT
    return HIGH_LEVEL_CONTEXT_CONTRACT


def context_metadata_from_config(config: Any) -> Dict[str, Any]:
    source = normalize_context_source(getattr(config, "context_source", None))
    batch_size = int(getattr(config, "low_level_encode_batch_size", 64))
    if batch_size <= 0:
        raise ValueError("MODEL.RAENWM.low_level_encode_batch_size must be positive")
    metadata = {
        "format": CONTEXT_METADATA_FORMAT,
        "context_source": source,
        "context_contract": context_contract_for_source(source),
        "context_size": int(getattr(config, "context_size", 4)),
    }
    if source == LOW_LEVEL_CONTEXT_SOURCE:
        metadata.update(
            {
                "sampling_action": LOW_LEVEL_SAMPLING_ACTION,
                "teleport_anchor_policy": LOW_LEVEL_TELEPORT_ANCHOR_POLICY,
                "rgb_sensor_uuid": LOW_LEVEL_RGB_SENSOR_UUID,
                "encode_batch_size": batch_size,
            }
        )
    else:
        metadata.update(
            {
                "sampling_action": "HIGH_LEVEL_DECISION",
                "teleport_anchor_policy": "not_applicable",
                "rgb_sensor_uuid": LOW_LEVEL_RGB_SENSOR_UUID,
                "encode_batch_size": batch_size,
            }
        )
    return metadata


def _wrap_to_pi(angle: float) -> float:
    return ((float(angle) + math.pi) % (2.0 * math.pi)) - math.pi


def _as_position3(position: Any) -> np.ndarray:
    array = np.asarray(position, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(
            f"low-level context position must have shape [3], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("low-level context position must be finite")
    return array.copy()


def _as_rgb_uint8(rgb: Any) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(
            f"low-level context RGB must be HWC or CHW, got {array.shape}"
        )
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(
            f"low-level context RGB must have three channels, got {array.shape}"
        )
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        if array.size and float(array.max()) <= 1.0:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


@dataclass(frozen=True)
class LowLevelContextFrame:
    rgb: np.ndarray
    position: np.ndarray
    yaw: float


class LowLevelContextEventBuffer:
    """Episode-local raw RGB event buffer owned by one Habitat worker."""

    def __init__(
        self,
        *,
        pos_eps: float = 1.0e-3,
        yaw_eps: float = 1.0e-3,
        rgb_key: str = LOW_LEVEL_RGB_SENSOR_UUID,
    ):
        self.pos_eps = float(pos_eps)
        self.yaw_eps = float(yaw_eps)
        self.rgb_key = str(rgb_key)
        self._last_raw_frame: Optional[LowLevelContextFrame] = None
        self._pending_events: List[Dict[str, Any]] = []
        self._pending_stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> Dict[str, int]:
        return {
            "reset_events": 0,
            "frame_events": 0,
            "dropped_static_frames": 0,
        }

    def reset_trace(self) -> None:
        self._last_raw_frame = None
        self._pending_events.append({"type": "reset"})
        self._pending_stats["reset_events"] += 1

    def append_observation(
        self,
        observations: Mapping[str, Any],
        position: Any,
        yaw: float,
        *,
        movement_frame: bool = False,
    ) -> bool:
        if not isinstance(observations, Mapping):
            raise ValueError("low-level context observation must be a mapping")
        if self.rgb_key not in observations:
            raise ValueError(
                f"low-level context observation is missing {self.rgb_key!r}"
            )
        frame = LowLevelContextFrame(
            rgb=_as_rgb_uint8(observations[self.rgb_key]),
            position=_as_position3(position),
            yaw=_wrap_to_pi(float(yaw)),
        )
        previous = self._last_raw_frame
        is_static = False
        if previous is not None:
            delta_pos = float(
                np.linalg.norm(frame.position[[0, 2]] - previous.position[[0, 2]])
            )
            delta_yaw = abs(_wrap_to_pi(frame.yaw - previous.yaw))
            # A MOVE_FORWARD event represents translation.  A collision is
            # therefore static even if turns changed the camera yaw first.
            is_static = delta_pos < self.pos_eps and (
                movement_frame or delta_yaw < self.yaw_eps
            )
        self._last_raw_frame = frame
        if is_static:
            self._pending_stats["dropped_static_frames"] += 1
            return False
        self._pending_events.append(
            {
                "type": "frame",
                "rgb": frame.rgb,
                "position": frame.position,
                "yaw": float(frame.yaw),
            }
        )
        self._pending_stats["frame_events"] += 1
        return True

    def pop_payload(self) -> Dict[str, Any]:
        payload = {
            "format": LOW_LEVEL_EVENT_PAYLOAD_FORMAT,
            "contract": LOW_LEVEL_CONTEXT_CONTRACT,
            "events": list(self._pending_events),
            "stats": dict(self._pending_stats),
        }
        self._pending_events.clear()
        self._pending_stats = self._empty_stats()
        return payload


class LowLevelContextSynchronizer:
    """Drain raw worker events, batch-encode RGB, and update one NWM runtime."""

    def __init__(self, *, runtime: Any, encoder: Any, device: Any, batch_size: int):
        self.runtime = runtime
        self.encoder = encoder
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("low-level context encode batch size must be positive")
        if normalize_context_source(getattr(runtime, "context_source", None)) != (
            LOW_LEVEL_CONTEXT_SOURCE
        ):
            raise ValueError("low-level context synchronizer requires low-level runtime")
        if not hasattr(encoder, "forward_with_raw_cls_and_patch_latents"):
            raise ValueError(
                "low-level context requires a RAE/DINOv2 encoder with raw CLS and patch output"
            )

    @staticmethod
    def _validate_payload(payload: Any, env_index: int):
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"environment {env_index} returned an invalid low-level event payload"
            )
        if payload.get("format") != LOW_LEVEL_EVENT_PAYLOAD_FORMAT:
            raise ValueError(
                f"environment {env_index} returned the wrong low-level event format"
            )
        if payload.get("contract") != LOW_LEVEL_CONTEXT_CONTRACT:
            raise ValueError(
                f"environment {env_index} returned the wrong low-level context contract"
            )
        events = payload.get("events")
        stats = payload.get("stats")
        if not isinstance(events, list) or not isinstance(stats, Mapping):
            raise ValueError(
                f"environment {env_index} low-level payload lacks events or stats"
            )
        return events, stats

    def _encode_frames(self, frames: List[np.ndarray]):
        if not frames:
            return (
                torch.empty((0, 768), device=self.device),
                torch.empty((0, 768, 16, 16), device=self.device),
                0,
            )
        expected_shape = frames[0].shape
        if any(frame.shape != expected_shape for frame in frames):
            raise ValueError("low-level RGB frames in one drain must have equal shapes")
        raw_cls_chunks = []
        patch_chunks = []
        encode_batches = 0
        for start in range(0, len(frames), self.batch_size):
            rgb_batch = np.stack(frames[start : start + self.batch_size], axis=0)
            with torch.no_grad():
                raw_cls, _navigation_cls, patch = (
                    self.encoder.forward_with_raw_cls_and_patch_latents(
                        {"rgb": rgb_batch}
                    )
                )
            if tuple(raw_cls.shape[1:]) != (768,):
                raise ValueError(
                    f"low-level raw CLS must have shape [N,768], got {tuple(raw_cls.shape)}"
                )
            if tuple(patch.shape[1:]) != (768, 16, 16):
                raise ValueError(
                    "low-level patch latent must have shape [N,768,16,16], "
                    f"got {tuple(patch.shape)}"
                )
            raw_cls_chunks.append(raw_cls.detach())
            patch_chunks.append(patch.detach())
            encode_batches += 1
        return (
            torch.cat(raw_cls_chunks, dim=0),
            torch.cat(patch_chunks, dim=0),
            encode_batches,
        )

    def drain(self, envs: Any) -> Dict[str, float]:
        num_envs = int(getattr(envs, "num_envs", 0))
        if num_envs <= 0:
            return {
                "drain_count": 0.0,
                "environment_count": 0.0,
                "reset_events": 0.0,
                "frame_events": 0.0,
                "encoded_frames": 0.0,
                "encode_batches": 0.0,
                "encode_seconds": 0.0,
                "context_ready": 0.0,
                "context_ready_ratio": 0.0,
                "dropped_static_frames": 0.0,
            }
        if len(self.runtime.adapter.buffers) != num_envs:
            raise RuntimeError(
                "low-level NWM buffer count does not match active environments"
            )
        payloads = envs.call(["pop_raenwm_context_events"] * num_envs)
        if not isinstance(payloads, (list, tuple)) or len(payloads) != num_envs:
            raise ValueError("low-level event read returned the wrong environment count")

        event_sequence: List[Dict[str, Any]] = []
        frames: List[np.ndarray] = []
        reset_events = 0
        dropped_static_frames = 0
        for env_index, payload in enumerate(payloads):
            events, stats = self._validate_payload(payload, env_index)
            payload_reset_count = sum(
                isinstance(event, Mapping) and event.get("type") == "reset"
                for event in events
            )
            payload_frame_count = sum(
                isinstance(event, Mapping) and event.get("type") == "frame"
                for event in events
            )
            if int(stats.get("reset_events", -1)) != payload_reset_count or int(
                stats.get("frame_events", -1)
            ) != payload_frame_count:
                raise ValueError(
                    f"environment {env_index} low-level event stats do not match events"
                )
            dropped_static_frames += int(stats.get("dropped_static_frames", 0))
            for event in events:
                if not isinstance(event, Mapping):
                    raise ValueError("low-level context event must be a mapping")
                event_type = event.get("type")
                if event_type == "reset":
                    reset_events += 1
                    event_sequence.append(
                        {"type": "reset", "env_index": env_index}
                    )
                elif event_type == "frame":
                    frame_index = len(frames)
                    frames.append(_as_rgb_uint8(event.get("rgb")))
                    event_sequence.append(
                        {
                            "type": "frame",
                            "env_index": env_index,
                            "frame_index": frame_index,
                            "position": _as_position3(event.get("position")),
                            "yaw": _wrap_to_pi(float(event.get("yaw"))),
                        }
                    )
                else:
                    raise ValueError(
                        f"Unknown low-level context event type: {event_type!r}"
                    )

        if frames and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        raw_cls, raw_patch, encode_batches = self._encode_frames(frames)
        if frames and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        encode_seconds = time.perf_counter() - started
        self.runtime.apply_low_level_context_events(
            event_sequence,
            raw_patch_latents=raw_patch,
            raw_cls=raw_cls,
        )
        ready = sum(buffer.is_ready() for buffer in self.runtime.adapter.buffers)
        runtime_static = sum(
            int(buffer.stats.dropped_static_frames)
            for buffer in self.runtime.adapter.buffers
        )
        return {
            "drain_count": 1.0,
            "environment_count": float(num_envs),
            "reset_events": float(reset_events),
            "frame_events": float(len(frames)),
            "encoded_frames": float(len(frames)),
            "encode_batches": float(encode_batches),
            "encode_seconds": float(encode_seconds),
            "context_ready": float(ready),
            "context_ready_ratio": float(ready) / float(num_envs),
            "dropped_static_frames": float(
                dropped_static_frames + runtime_static
            ),
        }
