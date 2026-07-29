import os
import re
from pathlib import Path

import torch


_CHECKPOINT_PATTERN = re.compile(r"^ckpt\.iter([0-9]+)\.pth$")
_TRAIN_STATE_PATTERN = re.compile(r"^train_state\.iter([0-9]+)\.pth$")


def checkpoint_iteration(path):
    match = _CHECKPOINT_PATTERN.fullmatch(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def training_state_iteration(path):
    match = _TRAIN_STATE_PATTERN.fullmatch(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def _numbered_paths(checkpoint_dir, iteration_fn):
    directory = Path(checkpoint_dir)
    if not directory.is_dir():
        return []
    paths = []
    for path in directory.iterdir():
        iteration = iteration_fn(path)
        if path.is_file() and iteration is not None:
            paths.append((iteration, path))
    return [path for _, path in sorted(paths)]


def checkpoint_paths(checkpoint_dir):
    return _numbered_paths(checkpoint_dir, checkpoint_iteration)


def training_state_paths(checkpoint_dir):
    return _numbered_paths(checkpoint_dir, training_state_iteration)


def latest_checkpoint_path(checkpoint_dir):
    paths = checkpoint_paths(checkpoint_dir)
    if not paths:
        raise FileNotFoundError(
            f"No ckpt.iter*.pth checkpoint found in {checkpoint_dir}"
        )
    return str(paths[-1])


def latest_complete_checkpoint_pair(checkpoint_dir, training_state_dir=None):
    if training_state_dir is None:
        training_state_dir = Path(checkpoint_dir) / "train_states"
    models = {
        checkpoint_iteration(path): path
        for path in checkpoint_paths(checkpoint_dir)
    }
    states = {
        training_state_iteration(path): path
        for path in training_state_paths(training_state_dir)
    }
    complete_iterations = sorted(set(models).intersection(states))
    if not complete_iterations:
        raise FileNotFoundError(
            "No complete ckpt.iter*.pth and train_state.iter*.pth pair "
            f"found in {checkpoint_dir} and {training_state_dir}"
        )
    iteration = complete_iterations[-1]
    return str(models[iteration]), str(states[iteration])


def atomic_torch_save(obj, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}"
    )
    try:
        torch.save(obj=obj, f=str(temporary))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prune_training_states(checkpoint_dir, keep_last, keep_every_n_iters):
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    if keep_every_n_iters < 0:
        raise ValueError("keep_every_n_iters must be non-negative")

    paths = training_state_paths(checkpoint_dir)
    recent = set(paths[-keep_last:])
    retained = set(recent)
    if keep_every_n_iters:
        retained.update(
            path
            for path in paths
            if training_state_iteration(path) % keep_every_n_iters == 0
        )

    removed = []
    for path in paths:
        if path not in retained:
            path.unlink()
            removed.append(path)
    return removed
