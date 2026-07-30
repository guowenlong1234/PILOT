import copy
import random
from collections import Counter

import numpy as np


_FORMAT_VERSION = 1
_ITERATOR_FIELDS = (
    "_rep_count",
    "_step_count",
    "_prev_scene_id",
    "_max_rep_episode",
    "_max_rep_step",
)
_ITERATOR_SETTINGS = (
    "cycle",
    "group_by_scene",
    "shuffle",
    "max_scene_repetition_episodes",
    "max_scene_repetition_steps",
    "step_repetition_range",
)


def _episode_identity(episode):
    instruction = getattr(episode, "instruction", None)
    instruction_id = getattr(instruction, "instruction_id", None)
    return (
        str(episode.scene_id),
        str(episode.episode_id),
        None if instruction_id is None else str(instruction_id),
    )


def _snapshot_remaining(iterator):
    try:
        remaining_iterator = copy.copy(iterator._iterator)
    except (AttributeError, TypeError) as error:
        raise TypeError(
            "Habitat episode iterator does not expose a copyable remaining "
            "queue; exact episode-order checkpointing is unavailable"
        ) from error
    return list(remaining_iterator)


def capture_episode_iterator_state(core_env):
    """Capture the exact queue used by one Habitat environment worker."""
    iterator = core_env.episode_iterator
    if iterator is None:
        raise ValueError("Habitat environment has no episode iterator")

    required_attributes = _ITERATOR_FIELDS + _ITERATOR_SETTINGS
    missing_fields = [
        field for field in required_attributes
        if not hasattr(iterator, field)
    ]
    if missing_fields:
        raise TypeError(
            "Unsupported Habitat episode iterator; missing internal fields "
            f"{missing_fields}"
        )

    episodes = list(iterator.episodes)
    remaining = _snapshot_remaining(iterator)
    return {
        "format_version": _FORMAT_VERSION,
        "episode_order": [
            _episode_identity(episode) for episode in episodes
        ],
        "remaining_episode_order": [
            _episode_identity(episode) for episode in remaining
        ],
        "current_episode": _episode_identity(core_env.current_episode),
        "iterator_fields": {
            field: getattr(iterator, field) for field in _ITERATOR_FIELDS
        },
        "iterator_settings": {
            field: getattr(iterator, field) for field in _ITERATOR_SETTINGS
        },
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
    }


def _episode_lookup(episodes):
    lookup = {}
    duplicates = []
    for episode in episodes:
        identity = _episode_identity(episode)
        if identity in lookup:
            duplicates.append(identity)
        else:
            lookup[identity] = episode
    if duplicates:
        raise ValueError(
            "Episode identities must be unique within each environment; "
            f"duplicates include {duplicates[:3]}"
        )
    return lookup


def _validate_episode_set(saved_order, current_order):
    saved_counts = Counter(tuple(identity) for identity in saved_order)
    current_counts = Counter(current_order)
    if saved_counts == current_counts:
        return

    missing = list((saved_counts - current_counts).elements())
    extra = list((current_counts - saved_counts).elements())
    raise ValueError(
        "Cannot restore episode order because the dataset or environment "
        "split changed; "
        f"missing={missing[:3]}, extra={extra[:3]}, "
        f"saved_count={sum(saved_counts.values())}, "
        f"current_count={sum(current_counts.values())}"
    )


def restore_episode_iterator_state(core_env, state):
    """Restore one worker so its next reset selects the saved next episode."""
    if not isinstance(state, dict):
        raise TypeError("Episode iterator state must be a dictionary")
    if state.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            "Unsupported episode iterator state format: "
            f"{state.get('format_version')!r}"
        )

    required = {
        "episode_order",
        "remaining_episode_order",
        "current_episode",
        "iterator_fields",
        "iterator_settings",
        "python_random_state",
        "numpy_random_state",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(
            f"Episode iterator state is incomplete; missing {missing}"
        )

    iterator = core_env.episode_iterator
    if iterator is None:
        raise ValueError("Habitat environment has no episode iterator")

    current_episodes = list(iterator.episodes)
    lookup = _episode_lookup(current_episodes)
    current_order = [_episode_identity(episode) for episode in current_episodes]
    saved_order = [tuple(identity) for identity in state["episode_order"]]
    _validate_episode_set(saved_order, current_order)

    remaining_order = [
        tuple(identity) for identity in state["remaining_episode_order"]
    ]
    unknown_remaining = [
        identity for identity in remaining_order if identity not in lookup
    ]
    if unknown_remaining:
        raise ValueError(
            "Saved remaining episode queue contains episodes outside the "
            f"current environment split: {unknown_remaining[:3]}"
        )

    current_identity = tuple(state["current_episode"])
    if current_identity not in lookup:
        raise ValueError(
            "Saved current episode is outside the current environment split: "
            f"{current_identity}"
        )

    iterator_fields = state["iterator_fields"]
    missing_fields = sorted(set(_ITERATOR_FIELDS).difference(iterator_fields))
    if missing_fields:
        raise ValueError(
            "Episode iterator counters are incomplete; missing "
            f"{missing_fields}"
        )

    iterator_settings = state["iterator_settings"]
    missing_settings = sorted(
        set(_ITERATOR_SETTINGS).difference(iterator_settings)
    )
    if missing_settings:
        raise ValueError(
            "Episode iterator settings are incomplete; missing "
            f"{missing_settings}"
        )
    changed_settings = {
        field: (iterator_settings[field], getattr(iterator, field, None))
        for field in _ITERATOR_SETTINGS
        if iterator_settings[field] != getattr(iterator, field, None)
    }
    if changed_settings:
        raise ValueError(
            "Cannot restore episode order because iterator settings changed; "
            f"checkpoint/current={changed_settings}"
        )

    iterator.episodes = [lookup[identity] for identity in saved_order]
    iterator._iterator = iter(
        [lookup[identity] for identity in remaining_order]
    )
    for field in _ITERATOR_FIELDS:
        setattr(iterator, field, iterator_fields[field])

    core_env._current_episode = lookup[current_identity]
    core_env._episode_from_iter_on_reset = True
    core_env._episode_force_changed = False

    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
