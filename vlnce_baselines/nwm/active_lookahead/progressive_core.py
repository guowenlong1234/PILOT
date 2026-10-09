"""Pure geometry, budget and conservative argmax certificates (no simulator)."""
from dataclasses import dataclass
import math
from typing import Sequence
import torch


def validate_budget(max_future_depth=2, total_residual_bound=1.0,
                    budget_fractions=(0.5, 0.5), *, deployment_gain=1.0,
                    residual_clip=None, score_scale='logit'):
    if type(max_future_depth) is not int or max_future_depth < 1:
        raise ValueError('max_future_depth must be a positive integer')
    if len(budget_fractions) != max_future_depth:
        raise ValueError('budget length must equal max_future_depth')
    total = float(total_residual_bound)
    fractions = tuple(float(x) for x in budget_fractions)
    if not math.isfinite(total) or total < 0 or any(not math.isfinite(x) or x < 0 for x in fractions):
        raise ValueError('budget must be finite and nonnegative')
    if math.fsum(fractions) > 1.0:
        raise ValueError('budget fractions must sum to at most one')
    if deployment_gain != 1.0 or residual_clip is not None or score_scale != 'logit':
        raise ValueError('progressive requires gain=1, no residual clip and logit scale')
    return tuple(total * x for x in fractions)


@dataclass(frozen=True)
class DecisionCertificate:
    certified: bool
    winner_id: int | None
    winner_lower_bound: float
    strongest_competitor_upper_bound: float
    certificate_margin: float
    numerical_guard: float


def certify_decision(scores, remaining_budget, candidate_present_mask=None, *, numerical_guard=1e-6):
    """Certify one complete action row, including unqueried non-TopK actions.

    Guard includes FP32 accumulation/comparison roundoff. Strict ties never
    certify. Callers may only zero a budget after structural branch termination.
    """
    s = torch.as_tensor(scores).detach().to(dtype=torch.float64)
    r = torch.as_tensor(remaining_budget, device=s.device).detach().to(dtype=torch.float64)
    present = torch.ones_like(s, dtype=torch.bool) if candidate_present_mask is None else torch.as_tensor(candidate_present_mask, device=s.device, dtype=torch.bool)
    if s.ndim != 1 or r.shape != s.shape or present.shape != s.shape:
        raise ValueError('certificate requires equally shaped one dimensional rows')
    if numerical_guard < 0 or not math.isfinite(numerical_guard):
        raise ValueError('numerical_guard must be finite and nonnegative')
    if not torch.isfinite(s[present]).all() or not torch.isfinite(r[present]).all() or (r[present] < 0).any():
        raise ValueError('present scores and nonnegative budgets must be finite')
    if not present.any():
        return DecisionCertificate(False, None, -math.inf, math.inf, -math.inf, numerical_guard)
    scale = max(1., float((s[present].abs() + r[present]).max()))
    guard = max(float(numerical_guard), 16 * torch.finfo(torch.float32).eps * scale)
    winner = int(s.masked_fill(~present, -math.inf).argmax())
    lower = float(s[winner] - r[winner])
    other = present.clone()
    other[winner] = False
    upper = float((s + r)[other].max()) if other.any() else -math.inf
    margin = lower - upper
    return DecisionCertificate(margin > guard, winner, lower, upper, margin, guard)


def successor_geometry(q0_position: Sequence[float], q0_yaw: float,
                       target_position: Sequence[float], target_yaw: float,
                       cumulative_path_length: float, distance_scale: float = 1.0):
    """Habitat yaw -> horizontal angle/forward axes anchored at candidate q0.

    Existing waypoint conversion uses motion heading = Habitat yaw - pi,
    world displacement = (sin(heading), 0, cos(heading)). Geometry axis 0 is
    increasing local waypoint angle; axis 1 is forward. Length starts q0.
    """
    values = (*q0_position, q0_yaw, *target_position, target_yaw, cumulative_path_length, distance_scale)
    if len(q0_position) != 3 or len(target_position) != 3 or not all(math.isfinite(float(x)) for x in values):
        raise ValueError('geometry requires finite 3D positions and yaw')
    if distance_scale <= 0 or cumulative_path_length < 0:
        raise ValueError('distance scale positive; cumulative length nonnegative')
    h = q0_yaw - math.pi
    dx, dz = target_position[0] - q0_position[0], target_position[2] - q0_position[2]
    relative = target_yaw - q0_yaw
    return ((dx * math.cos(h) - dz * math.sin(h)) / distance_scale,
            (dx * math.sin(h) + dz * math.cos(h)) / distance_scale,
            math.sin(relative), math.cos(relative), cumulative_path_length / distance_scale)
