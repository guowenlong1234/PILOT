"""Direct signed supervision for Stage-0 V2 offline training.

This module intentionally has no dependency on planner logits.  The offline
objective only teaches the bounded residual of each queried candidate to be
positive (teacher) or negative (non-teacher).  Pairwise, final-decision and
regularization losses belong to the later closed-loop joint stage and are
rejected here.
"""

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OfflineSignedLossConfig:
    training_stage: str = "offline"
    margin: float = 0.25
    temperature: float = 0.25
    eta_absent: float = 0.5
    positive_group_weight: float = 0.5
    negative_group_weight: float = 0.5
    pair_weight: float = 0.0
    final_weight: float = 0.0
    regularization_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.training_stage != "offline":
            raise ValueError("direct offline loss requires training_stage=offline")
        scalars = {
            "margin": self.margin,
            "temperature": self.temperature,
            "eta_absent": self.eta_absent,
            "positive_group_weight": self.positive_group_weight,
            "negative_group_weight": self.negative_group_weight,
            "pair_weight": self.pair_weight,
            "final_weight": self.final_weight,
            "regularization_weight": self.regularization_weight,
        }
        if any(not math.isfinite(float(value)) for value in scalars.values()):
            raise ValueError("offline loss weights and scales must be finite")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.eta_absent < 0:
            raise ValueError("eta_absent must be non-negative")
        if self.positive_group_weight < 0 or self.negative_group_weight < 0:
            raise ValueError("positive/negative group weights must be non-negative")
        total = self.positive_group_weight + self.negative_group_weight
        if abs(total - 1.0) > 1e-8:
            raise ValueError("positive/negative group weights must sum to 1")
        forbidden = {
            "pair_weight": self.pair_weight,
            "final_weight": self.final_weight,
            "regularization_weight": self.regularization_weight,
        }
        enabled = {name: value for name, value in forbidden.items() if float(value) != 0.0}
        if enabled:
            raise ValueError(
                "offline stage forbids pair/final/regularization losses; "
                f"non-zero weights: {enabled}"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OfflineSignedLossConfig":
        return cls(**dict(values))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OfflineSignedLossResult:
    loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    absent_loss: torch.Tensor
    valid_sample_count: int
    teacher_present_count: int
    teacher_absent_count: int
    skipped_stop_count: int
    skipped_no_teacher_count: int
    skipped_invalid_positive_count: int
    skipped_no_valid_candidate_count: int


def offline_signed_delta_loss(
    deltas: torch.Tensor,
    teacher_rank_in_topk: torch.Tensor,
    *,
    topk_valid_mask: torch.Tensor,
    teacher_valid: torch.Tensor,
    teacher_stop: torch.Tensor,
    no_vp_left: torch.Tensor,
    base_stop: torch.Tensor,
    config: OfflineSignedLossConfig,
) -> OfflineSignedLossResult:
    """Compute the signed-only Stage-0 V2 offline objective.

    Args:
        deltas: Bounded residuals with shape ``[B,K]``.
        teacher_rank_in_topk: Teacher slot in ``[0,K)`` or ``-1`` when the
            MOVE teacher is outside topK.
        topk_valid_mask: Candidates with a valid cached future/q0.
        teacher_valid: False when collection could not construct a trustworthy
            teacher label; these rows are diagnostics-only.
        teacher_stop: True for STOP teachers; these rows are skipped.
        no_vp_left: True for ``teacher=-100`` / no executable teacher.
        base_stop: True when the frozen planner selected STOP.  Strict STOP
            isolation excludes these rows from MOVE residual training.
        config: Validated offline-only loss contract.
    """
    if not isinstance(config, OfflineSignedLossConfig):
        raise TypeError("config must be OfflineSignedLossConfig")
    if deltas.ndim != 2:
        raise ValueError("deltas must have shape [B,K]")
    if not deltas.is_floating_point():
        raise ValueError("deltas must be floating point")
    batch, topk = deltas.shape
    if topk_valid_mask.shape != deltas.shape:
        raise ValueError("topk_valid_mask must match deltas")
    for name, value in {
        "teacher_rank_in_topk": teacher_rank_in_topk,
        "teacher_valid": teacher_valid,
        "teacher_stop": teacher_stop,
        "no_vp_left": no_vp_left,
        "base_stop": base_stop,
    }.items():
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape [B]")

    device = deltas.device
    ranks = teacher_rank_in_topk.detach().to(device=device, dtype=torch.long)
    valid = topk_valid_mask.detach().to(device=device, dtype=torch.bool)
    teacher_valid = teacher_valid.detach().to(device=device, dtype=torch.bool)
    teacher_stop = teacher_stop.detach().to(device=device, dtype=torch.bool)
    no_vp_left = no_vp_left.detach().to(device=device, dtype=torch.bool)
    base_stop = base_stop.detach().to(device=device, dtype=torch.bool)
    invalid_rank = (ranks < -1) | (ranks >= topk)
    finite = torch.isfinite(deltas)
    row_has_valid = valid.any(dim=1)
    stopped = teacher_valid & (base_stop | teacher_stop)
    skipped_no_teacher_mask = ~teacher_valid | (
        teacher_valid & ~stopped & no_vp_left
    )
    eligible = teacher_valid & ~stopped & ~no_vp_left
    teacher_present = ranks >= 0

    if topk:
        safe_ranks = ranks.clamp(min=0, max=topk - 1)
        positive_valid = valid.gather(1, safe_ranks[:, None]).squeeze(1)
        positive_mask = F.one_hot(safe_ranks, num_classes=topk).to(torch.bool)
        positive_mask &= teacher_present[:, None]
        positive_values = F.softplus(
            (
                config.margin
                - deltas.gather(1, safe_ranks[:, None]).squeeze(1)
            )
            / config.temperature
        )
    else:
        positive_valid = torch.zeros(batch, dtype=torch.bool, device=device)
        positive_mask = torch.zeros_like(valid)
        positive_values = deltas.sum(dim=1) * 0.0

    usable_present = eligible & row_has_valid & teacher_present & positive_valid
    usable_absent = eligible & row_has_valid & ~teacher_present
    invalid_positive = eligible & row_has_valid & teacher_present & ~positive_valid
    no_valid_candidate = eligible & ~row_has_valid

    negative_mask = valid & ~positive_mask
    negative_counts = negative_mask.sum(dim=1)
    negative_terms = F.softplus(
        (config.margin + deltas) / config.temperature
    )
    negative_values = torch.where(
        negative_mask, negative_terms, torch.zeros_like(negative_terms)
    ).sum(dim=1) / negative_counts.clamp_min(1)

    present_row_loss = torch.where(
        negative_counts > 0,
        config.positive_group_weight * positive_values
        + config.negative_group_weight * negative_values,
        # K can be one near episode termination.  Do not halve the only
        # available supervision term merely because no negative exists.
        positive_values,
    )
    sample_mask = usable_present | usable_absent
    sample_values = torch.where(
        usable_present,
        present_row_loss,
        torch.where(
            usable_absent,
            config.eta_absent * negative_values,
            torch.zeros_like(negative_values),
        ),
    )
    negative_contributor = (
        usable_present & (negative_counts > 0)
    ) | usable_absent
    zero = deltas.sum() * 0.0

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        total = torch.where(mask, values, torch.zeros_like(values)).sum()
        return total / mask.sum().clamp_min(1)

    # Convert every diagnostic counter in one transfer.  The previous row-wise
    # implementation called bool()/int() on CUDA tensors hundreds of times per
    # batch, forcing the CPU to synchronize with the GPU for every candidate.
    stats = torch.stack(
        (
            (~finite).sum(),
            invalid_rank.sum(),
            sample_mask.sum(),
            usable_present.sum(),
            usable_absent.sum(),
            stopped.sum(),
            skipped_no_teacher_mask.sum(),
            invalid_positive.sum(),
            no_valid_candidate.sum(),
        )
    ).detach().cpu().tolist()
    (
        non_finite_count,
        invalid_rank_count,
        valid_sample_count,
        present_count,
        absent_count,
        skipped_stop,
        skipped_no_teacher,
        skipped_invalid_positive,
        skipped_no_valid,
    ) = (int(value) for value in stats)
    if non_finite_count:
        raise ValueError("deltas must be finite")
    if invalid_rank_count:
        bad = ranks[invalid_rank].detach().cpu().tolist()
        raise ValueError(f"teacher_rank_in_topk must be -1 or in [0,K): {bad}")

    return OfflineSignedLossResult(
        loss=_masked_mean(sample_values, sample_mask) if batch else zero,
        positive_loss=_masked_mean(positive_values, usable_present)
        if batch
        else zero,
        negative_loss=_masked_mean(negative_values, negative_contributor)
        if batch
        else zero,
        absent_loss=_masked_mean(negative_values, usable_absent)
        if batch
        else zero,
        valid_sample_count=valid_sample_count,
        teacher_present_count=present_count,
        teacher_absent_count=absent_count,
        skipped_stop_count=skipped_stop,
        skipped_no_teacher_count=skipped_no_teacher,
        skipped_invalid_positive_count=skipped_invalid_positive,
        skipped_no_valid_candidate_count=skipped_no_valid,
    )
