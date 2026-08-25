"""Goal-independent persistent q0 records for current and historical ghosts."""

from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .types import PersistentGhostQ0


def _position(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite xyz position")
    return result.copy()


def _ghost_id(mapping_entry: Any) -> Optional[str]:
    value = mapping_entry[1] if isinstance(mapping_entry, (tuple, list)) else mapping_entry
    return None if value is None else str(value)


def build_persistent_q0_records(
    candidate_to_ghost: Sequence[Any],
    candidate_estimated_positions: Sequence[Any],
    candidate_real_positions: Sequence[Any],
    candidate_q0_records: Sequence[Mapping[str, Any]],
    candidate_view_indices: Sequence[int],
    candidate_forward_distances: Sequence[float],
    *,
    source_front_vp: str,
    source_high_level_step: int,
    skipped_candidates: Optional[list] = None,
) -> Tuple[PersistentGhostQ0, ...]:
    """Convert aligned current candidates into legal persistent q0 records.

    Invalid trajectory records and candidates that localized to an existing
    graph node are not persisted.  This function intentionally accepts no
    teacher, goal position or goal distance.
    """

    count = len(candidate_to_ghost)
    aligned = {
        "candidate_estimated_positions": candidate_estimated_positions,
        "candidate_real_positions": candidate_real_positions,
        "candidate_q0_records": candidate_q0_records,
        "candidate_view_indices": candidate_view_indices,
        "candidate_forward_distances": candidate_forward_distances,
    }
    for name, values in aligned.items():
        if len(values) != count:
            raise ValueError(f"{name} must align with candidate_to_ghost")

    front = str(source_front_vp)
    step = int(source_high_level_step)
    if not front:
        raise ValueError("source_front_vp must be non-empty")
    if step < 0:
        raise ValueError("source_high_level_step must be non-negative")

    result = []
    for index, mapping_entry in enumerate(candidate_to_ghost):
        ghost = _ghost_id(mapping_entry)
        if ghost is None:
            continue
        q0_record = candidate_q0_records[index]
        if not bool(q0_record.get("valid", False)):
            if skipped_candidates is not None:
                skipped_candidates.append(
                    {
                        "candidate_index": int(index),
                        "ghost_vp": ghost,
                        "error_type": q0_record.get("error_type"),
                        "error": q0_record.get("error"),
                    }
                )
            continue
        try:
            estimated = _position(
                candidate_estimated_positions[index],
                name=f"candidate_estimated_positions[{index}]",
            )
            raw_real = _position(
                q0_record.get("raw_position", candidate_real_positions[index]),
                name=f"candidate_real_positions[{index}]",
            )
            canonical = _position(
                q0_record.get("position"),
                name=f"candidate_q0_records[{index}].position",
            )
            island = int(q0_record["navmesh_island"])
            forward = float(candidate_forward_distances[index])
            view_index = int(candidate_view_indices[index])
            if not np.isfinite(forward) or forward < 0:
                raise ValueError("candidate forward distance must be finite and non-negative")
            if view_index < 0:
                raise ValueError("candidate view index must be non-negative")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            if skipped_candidates is not None:
                skipped_candidates.append(
                    {
                        "candidate_index": int(index),
                        "ghost_vp": ghost,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            continue
        result.append(
            PersistentGhostQ0(
                ghost_vp=ghost,
                source_front_vp=front,
                source_high_level_step=step,
                estimated_position=estimated,
                raw_real_position=raw_real,
                canonical_q0_position=canonical,
                navmesh_island_id=island,
                candidate_forward_m=forward,
                current_view_index=view_index,
                trajectory_valid=True,
            )
        )
    return tuple(result)


def append_persistent_q0_records(
    records_by_ghost: MutableMapping[str, list],
    records: Sequence[PersistentGhostQ0],
) -> None:
    """Append in observation order; list order is the final tie-breaker."""

    for record in records:
        if not isinstance(record, PersistentGhostQ0):
            raise TypeError("records must contain PersistentGhostQ0 values")
        if not record.trajectory_valid:
            raise ValueError("only trajectory-valid q0 records may be persisted")
        records_by_ghost.setdefault(record.ghost_vp, []).append(record)


def persistent_q0_to_dict(record: PersistentGhostQ0) -> Mapping[str, Any]:
    """Return JSON/shard-friendly fields matching the V2 offline schema."""

    if not isinstance(record, PersistentGhostQ0):
        raise TypeError("record must be a PersistentGhostQ0")
    return {
        "ghost_vp": str(record.ghost_vp),
        "source_front_vp": str(record.source_front_vp),
        "source_high_level_step": int(record.source_high_level_step),
        "estimated_position": _position(
            record.estimated_position, name="estimated_position"
        ).tolist(),
        "raw_real_position": _position(
            record.raw_real_position, name="raw_real_position"
        ).tolist(),
        "canonical_q0_position": _position(
            record.canonical_q0_position, name="canonical_q0_position"
        ).tolist(),
        "navmesh_island_id": int(record.navmesh_island_id),
        "candidate_forward_m": float(record.candidate_forward_m),
        "current_view_index": int(record.current_view_index),
        "trajectory_valid": bool(record.trajectory_valid),
    }


def persistent_q0_from_dict(payload: Mapping[str, Any]) -> PersistentGhostQ0:
    """Recreate one validated record from an offline sample."""

    forward = float(payload["candidate_forward_m"])
    view_index = int(payload["current_view_index"])
    step = int(payload["source_high_level_step"])
    island = int(payload["navmesh_island_id"])
    valid = bool(payload["trajectory_valid"])
    if not valid:
        raise ValueError("persistent q0 payload must be trajectory-valid")
    if not np.isfinite(forward) or forward < 0:
        raise ValueError("candidate_forward_m must be finite and non-negative")
    if min(view_index, step) < 0:
        raise ValueError("view index and source step must be non-negative")
    return PersistentGhostQ0(
        ghost_vp=str(payload["ghost_vp"]),
        source_front_vp=str(payload["source_front_vp"]),
        source_high_level_step=step,
        estimated_position=_position(payload["estimated_position"], name="estimated_position"),
        raw_real_position=_position(payload["raw_real_position"], name="raw_real_position"),
        canonical_q0_position=_position(
            payload["canonical_q0_position"], name="canonical_q0_position"
        ),
        navmesh_island_id=island,
        candidate_forward_m=forward,
        current_view_index=view_index,
        trajectory_valid=True,
    )


def select_canonical_q0(
    records: Sequence[PersistentGhostQ0],
    ghost_mean_position: Any,
) -> Optional[PersistentGhostQ0]:
    """Choose nearest estimated position, then newest step, then insertion order."""

    mean = _position(ghost_mean_position, name="ghost_mean_position")
    eligible = [record for record in records if record.trajectory_valid]
    if not eligible:
        return None
    # Python's min is stable, so exact ties after distance and step retain the
    # original insertion order without adding goal-dependent information.
    return min(
        eligible,
        key=lambda record: (
            float(np.linalg.norm(_position(record.estimated_position, name="estimated_position") - mean)),
            -int(record.source_high_level_step),
        ),
    )


def select_ghost_canonical_q0(
    records_by_ghost: Mapping[str, Sequence[PersistentGhostQ0]],
    ghost_vp: str,
    ghost_mean_position: Any,
) -> Optional[PersistentGhostQ0]:
    return select_canonical_q0(records_by_ghost.get(str(ghost_vp), ()), ghost_mean_position)
