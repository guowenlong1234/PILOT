from dataclasses import replace
import json

import numpy as np

from vlnce_baselines.nwm.active_lookahead.persistent_q0 import (
    append_persistent_q0_records,
    build_persistent_q0_records,
    persistent_q0_from_dict,
    persistent_q0_to_dict,
    select_canonical_q0,
)


def _make_records(skipped=None):
    return build_persistent_q0_records(
        [("c0", "g0"), ("c1", None), ("c2", "g0")],
        [[1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        [[1.0, 0.1, 0.0], [9.0, 0.0, 0.0], [1.2, 0.1, 0.0]],
        [
            {
                "valid": True,
                "position": [1.0, 0.0, 0.0],
                "raw_position": [1.0, 0.1, 0.0],
                "navmesh_island": 7,
            },
            {"valid": False, "error": "node is ignored"},
            {"valid": False, "error_type": "RuntimeError", "error": "bad trajectory"},
        ],
        [3, 8, 5],
        [1.0, 2.0, 1.25],
        source_front_vp="front-4",
        source_high_level_step=4,
        skipped_candidates=skipped,
    )


def test_build_persistent_q0_keeps_only_legal_ghost_trajectory_records():
    skipped = []
    records = _make_records(skipped)

    assert len(records) == 1
    record = records[0]
    assert record.ghost_vp == "g0"
    assert record.source_front_vp == "front-4"
    assert record.source_high_level_step == 4
    assert record.navmesh_island_id == 7
    assert record.current_view_index == 3
    assert record.candidate_forward_m == 1.0
    assert record.trajectory_valid is True
    assert np.array_equal(record.estimated_position, [1.0, 0.0, 0.0])
    assert np.array_equal(record.raw_real_position, [1.0, 0.1, 0.0])
    assert np.array_equal(record.canonical_q0_position, [1.0, 0.0, 0.0])
    assert skipped == [
        {
            "candidate_index": 2,
            "ghost_vp": "g0",
            "error_type": "RuntimeError",
            "error": "bad trajectory",
        }
    ]


def test_canonical_q0_uses_nearest_then_newest_then_insertion_order():
    base = _make_records()[0]
    far_new = replace(
        base,
        source_high_level_step=9,
        estimated_position=np.asarray([1.4, 0.0, 0.0]),
        canonical_q0_position=np.asarray([14.0, 0.0, 0.0]),
    )
    near_old = replace(
        base,
        source_high_level_step=3,
        estimated_position=np.asarray([1.1, 0.0, 0.0]),
        canonical_q0_position=np.asarray([31.0, 0.0, 0.0]),
    )
    near_new_first = replace(
        near_old,
        source_high_level_step=5,
        canonical_q0_position=np.asarray([51.0, 0.0, 0.0]),
    )
    near_new_second = replace(
        near_new_first,
        canonical_q0_position=np.asarray([52.0, 0.0, 0.0]),
    )

    chosen = select_canonical_q0(
        [far_new, near_old, near_new_first, near_new_second],
        ghost_mean_position=[1.0, 0.0, 0.0],
    )

    assert np.array_equal(chosen.canonical_q0_position, [51.0, 0.0, 0.0])


def test_empty_or_invalid_history_has_no_queryable_q0():
    base = _make_records()[0]
    assert select_canonical_q0([], [0.0, 0.0, 0.0]) is None
    assert (
        select_canonical_q0(
            [replace(base, trajectory_valid=False)], [0.0, 0.0, 0.0]
        )
        is None
    )


def test_persistent_q0_payload_is_json_serializable_and_round_trips():
    original = _make_records()[0]
    payload = persistent_q0_to_dict(original)
    encoded = json.dumps(payload, sort_keys=True)
    restored = persistent_q0_from_dict(json.loads(encoded))

    assert persistent_q0_to_dict(restored) == payload


def test_append_preserves_observation_order_per_ghost():
    first = _make_records()[0]
    second = replace(first, source_high_level_step=5)
    store = {}
    append_persistent_q0_records(store, [first, second])

    assert store == {"g0": [first, second]}
