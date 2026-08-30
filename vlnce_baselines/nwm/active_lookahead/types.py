from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np
import torch


@dataclass(frozen=True)
class PersistentGhostQ0:
    """One goal-independent, trajectory-validated q0 observation of a ghost."""

    ghost_vp: str
    source_front_vp: str
    source_high_level_step: int
    estimated_position: np.ndarray
    raw_real_position: np.ndarray
    canonical_q0_position: np.ndarray
    navmesh_island_id: int
    candidate_forward_m: float
    current_view_index: int
    trajectory_valid: bool


@dataclass(frozen=True)
class CandidateQ0:
    """One reusable first-stage prediction for the current state of a ghost."""

    contract_version: str
    position_source: str
    ghost_vp: str
    source_front_vp: str
    source_high_level_step: int
    source_position: np.ndarray
    source_yaw: float
    target_position: np.ndarray
    target_yaw: float
    condition: Tuple[float, float, float, float]
    horizon: float
    source_context: Any
    predicted_patch_cpu_fp16: torch.Tensor
    candidate_forward_m: float
    current_view_index: int

    @property
    def canonical_q0_position(self) -> np.ndarray:
        """Compatibility name used by the E24 geometry/future path."""

        return self.target_position


@dataclass(frozen=True)
class BoundedInvariantResult:
    """Worst-case bounded-residual proof for a frozen base ghost winner."""

    invariant: bool
    winner_index: int
    winner_lower_bound: float
    max_competitor_index: Optional[int]
    max_competitor_upper_bound: float

    def __bool__(self) -> bool:
        return self.invariant


@dataclass(frozen=True)
class TopKQueryPlan:
    """Deterministic query selection before any teacher or future is read."""

    base_action_index: int
    base_stop: bool
    executable_ghost_indices: Tuple[int, ...]
    ranked_ghost_indices: Tuple[int, ...]
    ranked_ghost_ids: Tuple[str, ...]
    query_indices: Tuple[int, ...]
    query_ghost_ids: Tuple[str, ...]
    skip_reason: Optional[str]
    bounded_result: Optional[BoundedInvariantResult]

    @property
    def query_count(self) -> int:
        return len(self.query_indices)

    @property
    def should_query(self) -> bool:
        return self.query_count > 0


@dataclass(frozen=True)
class FutureHeadOutput:
    raw_delta: torch.Tensor
    delta: torch.Tensor
    future_summary: torch.Tensor
    base_ghost_log_prob: torch.Tensor
