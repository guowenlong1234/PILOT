"""Decision-aware losses retained by the jointly trained E24 head."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .direct_loss import (
    OfflineSignedLossConfig,
    OfflineSignedLossResult,
    offline_signed_delta_loss,
)


DECISION_OBJECTIVE = "decision_aware_v1"
CHAMPION_ANCHORED_SCALAR_OBJECTIVE = "champion_anchored_scalar_v1"
ALL_DECISION_ROWS = "all_teacher_topk"

CHAMPION_FIX = 0
CHAMPION_HARM = 1
BASE_AND_CHAMPION_CORRECT = 2
BASE_AND_CHAMPION_WRONG = 3


@dataclass(frozen=True)
class OfflineDecisionLossConfig:
    training_stage: str = "offline"
    objective: str = DECISION_OBJECTIVE
    margin: float = 0.25
    temperature: float = 0.25
    eta_absent: float = 0.5
    positive_group_weight: float = 0.5
    negative_group_weight: float = 0.5
    signed_weight: float = 0.25
    present_signed_weight: float = 1.0
    absent_signed_weight: float = 1.0
    final_weight: float = 1.0
    pair_weight: float = 0.5
    regularization_weight: float = 0.01
    absent_noop_weight: float = 0.0
    pair_margin: float = 0.25
    pair_temperature: float = 0.25
    correct_row_weight: float = 1.0
    wrong_row_weight: float = 4.0
    decision_row_policy: str = ALL_DECISION_ROWS
    residual_bound: float = 1.0

    def __post_init__(self) -> None:
        if self.training_stage != "offline":
            raise ValueError("decision-aware loss requires training_stage=offline")
        if self.objective != DECISION_OBJECTIVE:
            raise ValueError(f"decision-aware objective must be {DECISION_OBJECTIVE}")
        scalars = {
            name: float(value)
            for name, value in asdict(self).items()
            if name not in {"training_stage", "objective", "decision_row_policy"}
        }
        if any(not math.isfinite(value) for value in scalars.values()):
            raise ValueError("decision-aware loss values must be finite")
        if self.margin < 0 or self.pair_margin < 0:
            raise ValueError("decision-aware margins must be non-negative")
        if self.temperature <= 0 or self.pair_temperature <= 0:
            raise ValueError("decision-aware temperatures must be positive")
        if self.residual_bound <= 0:
            raise ValueError("decision-aware residual_bound must be positive")
        if self.decision_row_policy != ALL_DECISION_ROWS:
            raise ValueError(
                "retained E24 loss requires "
                f"decision_row_policy={ALL_DECISION_ROWS!r}"
            )
        weights = (
            self.signed_weight,
            self.present_signed_weight,
            self.absent_signed_weight,
            self.final_weight,
            self.pair_weight,
            self.regularization_weight,
            self.absent_noop_weight,
            self.correct_row_weight,
            self.wrong_row_weight,
            self.eta_absent,
            self.positive_group_weight,
            self.negative_group_weight,
        )
        if any(value < 0 for value in weights):
            raise ValueError("decision-aware weights must be non-negative")
        if self.final_weight + self.pair_weight <= 0:
            raise ValueError("decision-aware loss requires final or pair supervision")
        if abs(self.positive_group_weight + self.negative_group_weight - 1.0) > 1e-8:
            raise ValueError("positive/negative group weights must sum to 1")
        if self.correct_row_weight + self.wrong_row_weight <= 0:
            raise ValueError("decision-aware row weights cannot both be zero")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OfflineDecisionLossConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signed_config(self) -> OfflineSignedLossConfig:
        return OfflineSignedLossConfig(
            training_stage="offline",
            margin=self.margin,
            temperature=self.temperature,
            eta_absent=self.eta_absent,
            positive_group_weight=self.positive_group_weight,
            negative_group_weight=self.negative_group_weight,
        )


@dataclass(frozen=True)
class OfflineChampionAnchoredScalarLossConfig:
    """Champion-default objective for a single non-negative row scale."""

    training_stage: str = "offline"
    objective: str = CHAMPION_ANCHORED_SCALAR_OBJECTIVE
    g_ref: float = 1.206
    g_min: float = 0.0
    g_max: float = 1.5
    delta_max: float = 1.0
    target_grid_step: float = 0.001
    interval_epsilon: float = 0.01
    fix_keep_weight: float = 4.0
    harm_undo_weight: float = 1.0
    stable_keep_weight: float = 0.25
    anchor_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.training_stage != "offline":
            raise ValueError("champion-anchored loss requires training_stage=offline")
        if self.objective != CHAMPION_ANCHORED_SCALAR_OBJECTIVE:
            raise ValueError(
                "champion-anchored objective must be "
                f"{CHAMPION_ANCHORED_SCALAR_OBJECTIVE}"
            )
        scalars = {
            name: float(value)
            for name, value in asdict(self).items()
            if name not in {"training_stage", "objective"}
        }
        if any(not math.isfinite(value) for value in scalars.values()):
            raise ValueError("champion-anchored loss values must be finite")
        if not self.g_min <= self.g_ref <= self.g_max:
            raise ValueError("g_ref must lie inside [g_min,g_max]")
        if self.g_min != 0 or self.g_max <= self.g_min:
            raise ValueError("champion-anchored g range is invalid")
        if self.delta_max <= 0 or self.target_grid_step <= 0:
            raise ValueError("delta_max and target_grid_step must be positive")
        if self.interval_epsilon < 0:
            raise ValueError("interval_epsilon must be non-negative")
        weights = (
            self.fix_keep_weight,
            self.harm_undo_weight,
            self.stable_keep_weight,
            self.anchor_weight,
        )
        if any(value < 0 for value in weights) or not any(weights):
            raise ValueError("champion-anchored weights must be non-negative and non-zero")
        grid_steps = round((self.g_max - self.g_min) / self.target_grid_step)
        reference_steps = round((self.g_ref - self.g_min) / self.target_grid_step)
        if (
            abs(self.g_min + grid_steps * self.target_grid_step - self.g_max) > 1e-8
            or abs(
                self.g_min + reference_steps * self.target_grid_step - self.g_ref
            )
            > 1e-8
        ):
            raise ValueError("g_max and g_ref must lie exactly on the target grid")

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any]
    ) -> "OfflineChampionAnchoredScalarLossConfig":
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OfflineDecisionLossResult:
    loss: torch.Tensor
    signed_loss: torch.Tensor
    positive_loss: torch.Tensor
    negative_loss: torch.Tensor
    absent_loss: torch.Tensor
    final_loss: torch.Tensor
    pair_loss: torch.Tensor
    regularization_loss: torch.Tensor
    absent_noop_loss: torch.Tensor
    decision_sample_count: int
    decision_weight_sum: float
    regularization_candidate_count: int
    absent_noop_candidate_count: int
    decision_base_correct_count: int
    decision_base_wrong_count: int
    decision_uncorrectable_wrong_count: int
    valid_sample_count: int
    teacher_present_count: int
    teacher_absent_count: int
    skipped_stop_count: int
    skipped_no_teacher_count: int
    skipped_invalid_positive_count: int
    skipped_no_valid_candidate_count: int


@dataclass
class ChampionAnchorTargets:
    group: torch.Tensor
    safe_lower: torch.Tensor
    safe_upper: torch.Tensor


@dataclass
class OfflineChampionAnchoredScalarLossResult:
    loss: torch.Tensor
    fix_keep_loss: torch.Tensor
    harm_undo_loss: torch.Tensor
    stable_keep_loss: torch.Tensor
    anchor_loss: torch.Tensor
    fix_count: int
    harm_count: int
    stable_count: int
    unresolved_count: int


def _correct_interval_at(
    correct: torch.Tensor, center_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the connected true interval containing one shared grid index."""

    if correct.ndim != 2 or not 0 <= center_index < correct.shape[1]:
        raise ValueError("correct grid or center index is invalid")
    batch, grid_size = correct.shape
    device = correct.device
    left_indices = torch.arange(center_index + 1, device=device)
    left_wrong = torch.where(
        ~correct[:, : center_index + 1],
        left_indices[None],
        torch.full((batch, center_index + 1), -1, device=device),
    )
    lower = left_wrong.max(dim=1).values + 1
    right_size = grid_size - center_index
    right_indices = torch.arange(right_size, device=device)
    right_wrong = torch.where(
        ~correct[:, center_index:],
        right_indices[None],
        torch.full((batch, right_size), right_size, device=device),
    )
    upper = center_index + right_wrong.min(dim=1).values - 1
    return lower, upper


def build_champion_anchor_targets(
    *,
    base_logits: torch.Tensor,
    ghost_valid_mask: torch.Tensor,
    topk_base_indices: torch.Tensor,
    topk_valid_mask: torch.Tensor,
    frozen_delta: torch.Tensor,
    teacher_base_index: torch.Tensor,
    config: OfflineChampionAnchoredScalarLossConfig,
) -> ChampionAnchorTargets:
    """Build exact-on-grid teacher-safe intervals for the scalar deployment ray."""

    if base_logits.ndim != 2 or ghost_valid_mask.shape != base_logits.shape:
        raise ValueError("champion target base logits/mask must have shape [B,G]")
    if (
        frozen_delta.ndim != 2
        or topk_base_indices.shape != frozen_delta.shape
        or topk_valid_mask.shape != frozen_delta.shape
    ):
        raise ValueError("champion target topK tensors must have shape [B,K]")
    batch, ghost_count = base_logits.shape
    if teacher_base_index.shape != (batch,):
        raise ValueError("champion target teacher index must have shape [B]")
    if not frozen_delta.is_floating_point() or not torch.isfinite(frozen_delta).all():
        raise ValueError("champion target frozen_delta must be finite floating point")
    valid = topk_valid_mask.to(device=base_logits.device, dtype=torch.bool)
    ghost_valid = ghost_valid_mask.to(device=base_logits.device, dtype=torch.bool)
    topk_indices = topk_base_indices.to(device=base_logits.device, dtype=torch.long)
    teachers = teacher_base_index.to(device=base_logits.device, dtype=torch.long)
    if bool((~ghost_valid.any(dim=1)).any()):
        raise ValueError("champion target row has no executable ghost")
    if bool(((teachers < 0) | (teachers >= ghost_count)).any()):
        raise ValueError("champion target teacher index is outside base logits")
    batch_rows, slots = valid.nonzero(as_tuple=True)
    selected = topk_indices[batch_rows, slots]
    if bool(((selected < 0) | (selected >= ghost_count)).any()):
        raise ValueError("champion target valid topK index is outside base logits")
    if bool((~ghost_valid[batch_rows, selected]).any()):
        raise ValueError("champion target valid topK index selects padding")

    grid_steps = round((config.g_max - config.g_min) / config.target_grid_step)
    grid = (
        torch.arange(
            grid_steps + 1,
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        * config.target_grid_step
        + config.g_min
    )
    reference_index = round(
        (config.g_ref - config.g_min) / config.target_grid_step
    )
    masked_base = base_logits.masked_fill(~ghost_valid, -torch.inf)
    scores = masked_base.unsqueeze(0).expand(grid.shape[0], -1, -1).clone()
    scaled = torch.clamp(
        grid[:, None, None] * frozen_delta[None],
        -config.delta_max,
        config.delta_max,
    )
    if batch_rows.numel():
        scores[:, batch_rows, selected] += scaled[:, batch_rows, slots]
    actions = scores.argmax(dim=2)
    correct = (actions == teachers[None]).transpose(0, 1)
    base_correct = correct[:, 0]
    reference_correct = correct[:, reference_index]

    group = torch.full(
        (batch,),
        BASE_AND_CHAMPION_WRONG,
        dtype=torch.long,
        device=base_logits.device,
    )
    group[~base_correct & reference_correct] = CHAMPION_FIX
    group[base_correct & ~reference_correct] = CHAMPION_HARM
    group[base_correct & reference_correct] = BASE_AND_CHAMPION_CORRECT

    ref_lower, ref_upper = _correct_interval_at(correct, reference_index)
    base_lower, base_upper = _correct_interval_at(correct, 0)
    lower_index = torch.zeros(batch, dtype=torch.long, device=base_logits.device)
    upper_index = torch.full(
        (batch,), grid_steps, dtype=torch.long, device=base_logits.device
    )
    use_reference = reference_correct
    use_base = base_correct & ~reference_correct
    lower_index = torch.where(use_reference, ref_lower, lower_index)
    upper_index = torch.where(use_reference, ref_upper, upper_index)
    lower_index = torch.where(use_base, base_lower, lower_index)
    upper_index = torch.where(use_base, base_upper, upper_index)
    return ChampionAnchorTargets(
        group=group,
        safe_lower=grid[lower_index],
        safe_upper=grid[upper_index],
    )


def offline_champion_anchored_scalar_loss(
    g: torch.Tensor,
    *,
    anchor_group: torch.Tensor,
    safe_lower: torch.Tensor,
    safe_upper: torch.Tensor,
    config: OfflineChampionAnchoredScalarLossConfig,
) -> OfflineChampionAnchoredScalarLossResult:
    """Protect champion corrections and only learn supported scalar rollbacks."""

    if g.ndim != 1:
        raise ValueError("champion-anchored g must have shape [B]")
    batch = g.shape[0]
    if (
        anchor_group.shape != (batch,)
        or safe_lower.shape != (batch,)
        or safe_upper.shape != (batch,)
    ):
        raise ValueError("champion-anchored targets must have shape [B]")
    group = anchor_group.to(device=g.device, dtype=torch.long)
    lower = safe_lower.to(device=g.device, dtype=g.dtype)
    upper = safe_upper.to(device=g.device, dtype=g.dtype)
    if bool(((group < CHAMPION_FIX) | (group > BASE_AND_CHAMPION_WRONG)).any()):
        raise ValueError("champion-anchored group is invalid")
    if not torch.isfinite(g).all() or not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        raise ValueError("champion-anchored values must be finite")
    if bool((lower > upper).any()):
        raise ValueError("champion-anchored safe interval is inverted")

    span = float(config.g_max - config.g_min)
    width = (upper - lower).clamp_min(0.0)
    inset = torch.minimum(
        torch.full_like(width, config.interval_epsilon),
        width * 0.25,
    )
    target_lower = lower + inset
    target_upper = upper - inset
    interval_rows = (
        F.relu(target_lower - g).square() + F.relu(g - target_upper).square()
    ) / (span * span)
    zero = g.sum() * 0.0

    def group_mean(group_id: int) -> torch.Tensor:
        mask = group == group_id
        return interval_rows[mask].mean() if bool(mask.any()) else zero

    fix_keep_loss = group_mean(CHAMPION_FIX)
    harm_undo_loss = group_mean(CHAMPION_HARM)
    stable_keep_loss = group_mean(BASE_AND_CHAMPION_CORRECT)
    anchor_loss = ((g - config.g_ref) / span).square().mean()
    total = (
        config.fix_keep_weight * fix_keep_loss
        + config.harm_undo_weight * harm_undo_loss
        + config.stable_keep_weight * stable_keep_loss
        + config.anchor_weight * anchor_loss
    )
    counts = torch.stack(
        [(group == value).sum() for value in range(BASE_AND_CHAMPION_WRONG + 1)]
    ).detach().cpu().tolist()
    return OfflineChampionAnchoredScalarLossResult(
        loss=total,
        fix_keep_loss=fix_keep_loss,
        harm_undo_loss=harm_undo_loss,
        stable_keep_loss=stable_keep_loss,
        anchor_loss=anchor_loss,
        fix_count=int(counts[CHAMPION_FIX]),
        harm_count=int(counts[CHAMPION_HARM]),
        stable_count=int(counts[BASE_AND_CHAMPION_CORRECT]),
        unresolved_count=int(counts[BASE_AND_CHAMPION_WRONG]),
    )


def offline_loss_config_from_mapping(values: Mapping[str, Any]):
    payload = dict(values)
    objective = str(payload.get("objective", ""))
    if objective == DECISION_OBJECTIVE:
        return OfflineDecisionLossConfig.from_mapping(payload)
    if objective == CHAMPION_ANCHORED_SCALAR_OBJECTIVE:
        return OfflineChampionAnchoredScalarLossConfig.from_mapping(payload)
    raise ValueError(f"unsupported offline objective: {objective}")


def offline_training_loss(
    deltas: torch.Tensor,
    teacher_rank_in_topk: torch.Tensor,
    *,
    topk_valid_mask: torch.Tensor,
    teacher_valid: torch.Tensor,
    teacher_stop: torch.Tensor,
    no_vp_left: torch.Tensor,
    base_stop: torch.Tensor,
    base_logits: torch.Tensor,
    ghost_valid_mask: torch.Tensor,
    topk_base_indices: torch.Tensor,
    teacher_base_index: torch.Tensor,
    config,
    residual_bound: float | None = None,
):
    if not isinstance(config, OfflineDecisionLossConfig):
        raise TypeError("only the retained decision-aware head loss is supported")
    return offline_decision_aware_loss(
        deltas,
        teacher_rank_in_topk,
        topk_valid_mask=topk_valid_mask,
        teacher_valid=teacher_valid,
        teacher_stop=teacher_stop,
        no_vp_left=no_vp_left,
        base_stop=base_stop,
        base_logits=base_logits,
        ghost_valid_mask=ghost_valid_mask,
        topk_base_indices=topk_base_indices,
        teacher_base_index=teacher_base_index,
        config=config,
    )


def offline_decision_aware_loss(
    deltas: torch.Tensor,
    teacher_rank_in_topk: torch.Tensor,
    *,
    topk_valid_mask: torch.Tensor,
    teacher_valid: torch.Tensor,
    teacher_stop: torch.Tensor,
    no_vp_left: torch.Tensor,
    base_stop: torch.Tensor,
    base_logits: torch.Tensor,
    ghost_valid_mask: torch.Tensor,
    topk_base_indices: torch.Tensor,
    teacher_base_index: torch.Tensor,
    config: OfflineDecisionLossConfig,
) -> OfflineDecisionLossResult:
    """Optimize the adjusted ghost decision while preserving signed guidance."""

    signed = offline_signed_delta_loss(
        deltas,
        teacher_rank_in_topk,
        topk_valid_mask=topk_valid_mask,
        teacher_valid=teacher_valid,
        teacher_stop=teacher_stop,
        no_vp_left=no_vp_left,
        base_stop=base_stop,
        config=config.signed_config(),
    )
    if base_logits.ndim != 2 or ghost_valid_mask.shape != base_logits.shape:
        raise ValueError("decision-aware base logits/mask must have shape [B,G]")
    if deltas.ndim != 2 or topk_base_indices.shape != deltas.shape:
        raise ValueError("decision-aware topK tensors must have shape [B,K]")
    batch = deltas.shape[0]
    if base_logits.shape[0] != batch or teacher_base_index.shape != (batch,):
        raise ValueError("decision-aware batch dimensions differ")

    device = deltas.device
    valid = topk_valid_mask.detach().to(device=device, dtype=torch.bool)
    ranks = teacher_rank_in_topk.detach().to(device=device, dtype=torch.long)
    ghost_valid = ghost_valid_mask.detach().to(device=device, dtype=torch.bool)
    topk_indices = topk_base_indices.detach().to(device=device, dtype=torch.long)
    teachers = teacher_base_index.detach().to(device=device, dtype=torch.long)
    if deltas.shape[1]:
        safe_ranks = ranks.clamp(min=0, max=deltas.shape[1] - 1)
        positive_valid = (ranks >= 0) & valid.gather(
            1, safe_ranks[:, None]
        ).squeeze(1)
    else:
        positive_valid = torch.zeros(batch, dtype=torch.bool, device=device)
    decision_candidate_eligible = (
        teacher_valid.detach().to(device=device, dtype=torch.bool)
        & ~teacher_stop.detach().to(device=device, dtype=torch.bool)
        & ~no_vp_left.detach().to(device=device, dtype=torch.bool)
        & ~base_stop.detach().to(device=device, dtype=torch.bool)
        & valid.any(dim=1)
        & (teachers >= 0)
    )
    decision_eligible = decision_candidate_eligible & positive_valid
    absent_noop_rows = decision_candidate_eligible & ~positive_valid
    if bool((decision_eligible & (teachers >= base_logits.shape[1])).any()):
        raise ValueError("teacher_base_index is outside base logits")
    rows, slots = valid.nonzero(as_tuple=True)
    selected = topk_indices[rows, slots]
    if bool(((selected < 0) | (selected >= base_logits.shape[1])).any()):
        raise ValueError("valid topK index is outside base logits")
    if bool((~ghost_valid[rows, selected]).any()):
        raise ValueError("valid topK index selects a padded ghost")
    if deltas.shape[1]:
        teacher_topk_indices = topk_indices.gather(1, safe_ranks[:, None]).squeeze(1)
        inconsistent = decision_eligible & (teacher_topk_indices != teachers)
        if bool(inconsistent.any()):
            raise ValueError(
                "teacher rank/topK base index differs from teacher_base_index"
            )

    masked_base = base_logits.detach().masked_fill(~ghost_valid, -torch.inf)
    base_actions_all = masked_base.argmax(dim=1)
    safe_teachers = teachers.clamp(min=0, max=base_logits.shape[1] - 1)
    teacher_base_scores = masked_base.gather(
        1, safe_teachers[:, None]
    ).squeeze(1)
    valid_teacher_rows = decision_eligible.nonzero(as_tuple=False).flatten()
    base_correct_all = base_actions_all == teachers
    adjustable = torch.zeros_like(ghost_valid)
    adjustable[rows, selected] = True
    reachable_teacher = teacher_base_scores + config.residual_bound * adjustable[
        torch.arange(batch, device=device), safe_teachers
    ].to(deltas.dtype)
    reachable_competitors = masked_base.clone()
    reachable_competitors -= config.residual_bound * adjustable.to(deltas.dtype)
    if valid_teacher_rows.numel():
        reachable_competitors[
            valid_teacher_rows, teachers[valid_teacher_rows]
        ] = -torch.inf
    reachable_margin = (
        reachable_teacher - reachable_competitors.max(dim=1).values
    )
    uncorrectable_wrong = (
        decision_eligible
        & ~base_correct_all
        & (reachable_margin < 0)
    )
    active = decision_eligible

    adjusted = masked_base.clone()
    adjusted[rows, selected] = adjusted[rows, selected] + deltas[rows, slots]
    active_rows = active.nonzero(as_tuple=False).flatten()
    zero = deltas.sum() * 0.0
    if active_rows.numel():
        active_adjusted = adjusted[active_rows]
        active_teachers = teachers[active_rows]
        base_actions = base_actions_all[active_rows]
        row_weights = torch.where(
            base_actions == active_teachers,
            torch.full_like(active_teachers, config.correct_row_weight, dtype=deltas.dtype),
            torch.full_like(active_teachers, config.wrong_row_weight, dtype=deltas.dtype),
        )
        weight_sum = row_weights.sum().clamp_min(torch.finfo(deltas.dtype).eps)
        final_rows = F.cross_entropy(
            active_adjusted, active_teachers, reduction="none"
        )
        final_loss = (final_rows * row_weights).sum() / weight_sum
        teacher_scores = active_adjusted.gather(1, active_teachers[:, None]).squeeze(1)
        competitors = active_adjusted.clone()
        competitors.scatter_(1, active_teachers[:, None], -torch.inf)
        hard_negative = competitors.max(dim=1).values
        pair_rows = F.softplus(
            (config.pair_margin - teacher_scores + hard_negative)
            / config.pair_temperature
        )
        pair_loss = (pair_rows * row_weights).sum() / weight_sum
        active_candidates = valid & active[:, None]
        regularization_loss = deltas[active_candidates].square().mean()
        decision_base_correct_count = int(
            (base_actions == active_teachers).sum().detach().cpu()
        )
        decision_base_wrong_count = int(active_rows.numel()) - decision_base_correct_count
    else:
        final_loss = pair_loss = regularization_loss = zero
        decision_base_correct_count = decision_base_wrong_count = 0

    absent_noop_candidates = valid & absent_noop_rows[:, None]
    absent_noop_loss = (
        deltas[absent_noop_candidates].square().mean()
        if bool(absent_noop_candidates.any())
        else zero
    )

    # Teacher-present rows already receive direct action-ranking supervision.
    # Keep their signed proxy independently tunable without weakening the only
    # available all-negative signal when the teacher is outside cached TopK.
    signed_count = signed.teacher_present_count + signed.teacher_absent_count
    if signed_count:
        absent_sum = (
            config.eta_absent
            * signed.absent_loss
            * signed.teacher_absent_count
        )
        present_sum = signed.loss * signed_count - absent_sum
        weighted_signed_loss = (
            config.present_signed_weight * present_sum
            + config.absent_signed_weight * absent_sum
        ) / signed_count
    else:
        weighted_signed_loss = zero

    total = (
        config.signed_weight * weighted_signed_loss
        + config.final_weight * final_loss
        + config.pair_weight * pair_loss
        + config.regularization_weight * regularization_loss
        + config.absent_noop_weight * absent_noop_loss
    )
    return OfflineDecisionLossResult(
        loss=total,
        signed_loss=weighted_signed_loss,
        positive_loss=signed.positive_loss,
        negative_loss=signed.negative_loss,
        absent_loss=signed.absent_loss,
        final_loss=final_loss,
        pair_loss=pair_loss,
        regularization_loss=regularization_loss,
        absent_noop_loss=absent_noop_loss,
        decision_sample_count=int(active_rows.numel()),
        decision_weight_sum=(
            float(weight_sum.detach().cpu()) if active_rows.numel() else 0.0
        ),
        regularization_candidate_count=(
            int(active_candidates.sum().detach().cpu()) if active_rows.numel() else 0
        ),
        absent_noop_candidate_count=int(
            absent_noop_candidates.sum().detach().cpu()
        ),
        decision_base_correct_count=decision_base_correct_count,
        decision_base_wrong_count=decision_base_wrong_count,
        decision_uncorrectable_wrong_count=int(
            uncorrectable_wrong.sum().detach().cpu()
        ),
        valid_sample_count=signed.valid_sample_count,
        teacher_present_count=signed.teacher_present_count,
        teacher_absent_count=signed.teacher_absent_count,
        skipped_stop_count=signed.skipped_stop_count,
        skipped_no_teacher_count=signed.skipped_no_teacher_count,
        skipped_invalid_positive_count=signed.skipped_invalid_positive_count,
        skipped_no_valid_candidate_count=signed.skipped_no_valid_candidate_count,
    )
