import random
from types import SimpleNamespace

import numpy as np
import pytest
from habitat.core.dataset import EpisodeIterator

from vlnce_baselines.common.episode_iterator_state import (
    capture_episode_iterator_state,
    restore_episode_iterator_state,
)


def _episode(scene_id, episode_id):
    return SimpleNamespace(
        scene_id=scene_id,
        episode_id=episode_id,
    )


def _iterator(episodes, remaining):
    return SimpleNamespace(
        episodes=list(episodes),
        _iterator=iter(list(remaining)),
        _rep_count=7,
        _step_count=19,
        _prev_scene_id=episodes[1].scene_id,
        _max_rep_episode=None,
        _max_rep_step=None,
        cycle=True,
        group_by_scene=True,
        shuffle=True,
        max_scene_repetition_episodes=-1,
        max_scene_repetition_steps=-1,
        step_repetition_range=0.2,
    )


class _CoreEnv:
    def __init__(self, iterator, current_episode):
        self.episode_iterator = iterator
        self._current_episode = current_episode
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

    @property
    def current_episode(self):
        return self._current_episode

    def select_next_episode_on_reset(self):
        if self._episode_from_iter_on_reset:
            self._current_episode = next(self.episode_iterator)
        self._episode_from_iter_on_reset = True
        return self._current_episode


def _core_env(episodes, remaining):
    return _CoreEnv(_iterator(episodes, remaining), episodes[1])


def test_episode_iterator_state_round_trip_preserves_exact_remaining_queue():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        episodes = [
            _episode("scene-a", "1"),
            _episode("scene-a", "2"),
            _episode("scene-b", "3"),
            _episode("scene-b", "4"),
        ]
        core_env = _core_env(episodes, episodes[2:])
        random.seed(123)
        np.random.seed(456)

        state = capture_episode_iterator_state(core_env)
        expected_python_value = random.random()
        expected_numpy_value = np.random.random()

        assert next(core_env.episode_iterator._iterator) is episodes[2]

        core_env.episode_iterator.episodes = list(reversed(episodes))
        core_env.episode_iterator._iterator = iter([episodes[0]])
        core_env.episode_iterator._rep_count = -1
        core_env.episode_iterator._step_count = 0
        core_env._current_episode = episodes[-1]
        random.seed(999)
        np.random.seed(999)

        restore_episode_iterator_state(core_env, state)

        assert core_env.episode_iterator.episodes == episodes
        assert list(core_env.episode_iterator._iterator) == episodes[2:]
        assert core_env._current_episode is episodes[1]
        assert core_env._episode_from_iter_on_reset is True
        assert core_env._episode_force_changed is False
        assert core_env.episode_iterator._rep_count == 7
        assert core_env.episode_iterator._step_count == 19
        assert random.random() == expected_python_value
        assert np.random.random() == expected_numpy_value
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def test_episode_iterator_restore_rejects_changed_environment_split():
    episodes = [
        _episode("scene-a", "1"),
        _episode("scene-a", "2"),
        _episode("scene-b", "3"),
    ]
    state = capture_episode_iterator_state(
        _core_env(episodes, episodes[2:])
    )
    changed_episodes = episodes[:-1] + [_episode("scene-c", "9")]
    changed_env = _core_env(changed_episodes, changed_episodes[2:])

    with pytest.raises(ValueError, match="dataset or environment split changed"):
        restore_episode_iterator_state(changed_env, state)


def test_episode_iterator_restore_rejects_changed_iterator_settings():
    episodes = [
        _episode("scene-a", "1"),
        _episode("scene-a", "2"),
        _episode("scene-b", "3"),
    ]
    state = capture_episode_iterator_state(
        _core_env(episodes, episodes[2:])
    )
    changed_env = _core_env(episodes, episodes[2:])
    changed_env.episode_iterator.shuffle = False

    with pytest.raises(ValueError, match="iterator settings changed"):
        restore_episode_iterator_state(changed_env, state)


def test_state_adapter_matches_the_runtime_habitat_episode_iterator():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        episodes = [
            _episode("scene-a", "1"),
            _episode("scene-a", "2"),
            _episode("scene-b", "3"),
            _episode("scene-b", "4"),
        ]
        iterator = EpisodeIterator(
            list(episodes),
            shuffle=True,
            group_by_scene=True,
            seed=17,
        )
        current_episode = next(iterator)
        core_env = _CoreEnv(iterator, current_episode)
        state = capture_episode_iterator_state(core_env)

        fresh_iterator = EpisodeIterator(
            list(episodes),
            shuffle=True,
            group_by_scene=True,
            seed=999,
        )
        fresh_current_episode = next(fresh_iterator)
        fresh_env = _CoreEnv(fresh_iterator, fresh_current_episode)
        fresh_env._episode_from_iter_on_reset = False
        fresh_env._episode_force_changed = True

        restore_episode_iterator_state(fresh_env, state)

        expected_next = state["remaining_episode_order"][0]
        restored_next = fresh_env.select_next_episode_on_reset()
        assert (
            restored_next.scene_id,
            str(restored_next.episode_id),
            None,
        ) == expected_next
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
