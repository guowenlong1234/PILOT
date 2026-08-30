"""Goal-independent geometry features shared by E24 rollout and replay."""

import math
from typing import Any, Mapping, Sequence

import torch


def candidate_q0_geometry(
    records: Sequence[Mapping[str, Any] | None],
) -> torch.Tensor:
    if not records:
        return torch.empty((0, 3), dtype=torch.float32)
    values = []
    for index, record in enumerate(records):
        if record is None:
            values.append((0.0, 0.0, 0.0))
            continue
        if isinstance(record, Mapping):
            forward = float(record["candidate_forward_m"])
            view = int(record["current_view_index"])
        else:
            forward = float(getattr(record, "candidate_forward_m"))
            view = int(getattr(record, "current_view_index"))
        if not math.isfinite(forward) or forward < 0:
            raise ValueError(
                f"persistent_q0[{index}].candidate_forward_m must be finite and non-negative"
            )
        if view < 0:
            raise ValueError(
                f"persistent_q0[{index}].current_view_index must be non-negative"
            )
        angle = 2.0 * math.pi * (view % 12) / 12.0
        values.append((forward / (1.0 + forward), math.sin(angle), math.cos(angle)))
    return torch.tensor(values, dtype=torch.float32)
