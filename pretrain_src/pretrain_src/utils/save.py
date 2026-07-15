"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
import random
import re

import numpy as np
import torch


_STATE_PATTERN = re.compile(r"^train_state_(\d+)\.pt$")


def _atomic_torch_save(payload, output_path):
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _checkpoint_step(path, pattern):
    match = pattern.match(os.path.basename(path))
    return int(match.group(1)) if match else None


def resolve_resume_checkpoint(resume_checkpoint, checkpoint_dir):
    if resume_checkpoint in (None, ""):
        return None
    if resume_checkpoint == "latest":
        resume_checkpoint = checkpoint_dir
    resume_checkpoint = os.path.abspath(resume_checkpoint)
    if os.path.isdir(resume_checkpoint):
        candidates = []
        for name in os.listdir(resume_checkpoint):
            step = _checkpoint_step(name, _STATE_PATTERN)
            if step is None:
                continue
            state_path = os.path.join(resume_checkpoint, name)
            model_path = os.path.join(
                resume_checkpoint, f"model_step_{step}.pt"
            )
            if os.path.isfile(model_path):
                candidates.append((step, state_path))
        if not candidates:
            raise FileNotFoundError(
                f"No complete model/train-state checkpoint pair in {resume_checkpoint}"
            )
        return max(candidates)[1]
    if not os.path.isfile(resume_checkpoint):
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {resume_checkpoint}"
        )
    if _checkpoint_step(resume_checkpoint, _STATE_PATTERN) is None:
        raise ValueError(
            "resume_checkpoint must point to train_state_<step>.pt, not a "
            "model-only checkpoint"
        )
    return resume_checkpoint


def load_training_state(resume_checkpoint):
    state = torch.load(resume_checkpoint, map_location="cpu")
    required = {
        "format_version",
        "step",
        "model_checkpoint",
        "optimizer",
        "grad_scaler",
        "rng_state",
        "meta_loader_step",
        "training_config",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(
            f"Incomplete training state {resume_checkpoint}; missing {missing}"
        )
    if state["format_version"] != 1:
        raise ValueError(
            f"Unsupported training-state format: {state['format_version']}"
        )
    filename_step = _checkpoint_step(resume_checkpoint, _STATE_PATTERN)
    if filename_step != state["step"]:
        raise ValueError(
            f"Training-state step mismatch: filename={filename_step}, payload={state['step']}"
        )
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(resume_checkpoint)),
        state["model_checkpoint"],
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint referenced by training state is missing: {model_path}"
        )
    return state, model_path


def validate_resume_config(state, opts):
    current = {
        "gradient_accumulation_steps": opts.gradient_accumulation_steps,
        "train_batch_size": opts.train_batch_size,
        "world_size": opts.world_size,
        "model_config": os.path.abspath(opts.model_config),
    }
    mismatches = {
        key: (state["training_config"].get(key), value)
        for key, value in current.items()
        if state["training_config"].get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r}, current={current_value!r}"
            for key, (saved, current_value) in mismatches.items()
        )
        raise ValueError(f"Resume configuration mismatch: {details}")


def save_training_meta(args):
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True)

    with open(os.path.join(args.output_dir, 'logs', 'training_args.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    model_config = json.load(open(args.model_config))
    with open(os.path.join(args.output_dir, 'logs', 'model_config.json'), 'w') as writer:
        json.dump(model_config, writer, indent=4)


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(
        self,
        model,
        step,
        optimizer=None,
        grad_scaler=None,
        meta_loader_step=None,
        opts=None,
        keep_last_checkpoints=3,
        keep_every_n_steps=0,
    ):
        output_model_file = os.path.join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {}
        for k, v in model.state_dict().items():
            if k.startswith('module.'):
                k = k[7:]
            if isinstance(v, torch.Tensor):
                state_dict[k] = v.cpu()
            else:
                state_dict[k] = v
        _atomic_torch_save(state_dict, output_model_file)
        if optimizer is not None:
            if opts is None:
                raise ValueError("opts is required when saving resumable state")
            dump = {
                "format_version": 1,
                "step": step,
                "model_checkpoint": os.path.basename(output_model_file),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": (
                    grad_scaler.state_dict() if grad_scaler is not None else None
                ),
                "rng_state": capture_rng_state(),
                "meta_loader_step": (
                    meta_loader_step
                    if meta_loader_step is not None
                    else step * opts.gradient_accumulation_steps
                ),
                "training_config": {
                    "gradient_accumulation_steps": opts.gradient_accumulation_steps,
                    "train_batch_size": opts.train_batch_size,
                    "world_size": opts.world_size,
                    "model_config": os.path.abspath(opts.model_config),
                },
            }
            state_path = os.path.join(
                self.output_dir, f"train_state_{step}.pt"
            )
            _atomic_torch_save(dump, state_path)
            self.prune(
                keep_last_checkpoints=keep_last_checkpoints,
                keep_every_n_steps=keep_every_n_steps,
            )
            return output_model_file, state_path
        return output_model_file, None

    def prune(self, keep_last_checkpoints=3, keep_every_n_steps=0):
        if keep_last_checkpoints < 1:
            raise ValueError("keep_last_checkpoints must be at least 1")
        if keep_every_n_steps < 0:
            raise ValueError("keep_every_n_steps cannot be negative")

        complete_steps = []
        for name in os.listdir(self.output_dir):
            step = _checkpoint_step(name, _STATE_PATTERN)
            if step is None:
                continue
            model_path = os.path.join(
                self.output_dir, f"{self.prefix}_{step}.{self.suffix}"
            )
            if os.path.isfile(model_path):
                complete_steps.append(step)
        recent_steps = set(sorted(complete_steps)[-keep_last_checkpoints:])
        for step in complete_steps:
            if step in recent_steps:
                continue
            state_path = os.path.join(self.output_dir, f"train_state_{step}.pt")
            if os.path.exists(state_path):
                os.remove(state_path)
            is_milestone = (
                keep_every_n_steps > 0 and step % keep_every_n_steps == 0
            )
            if not is_milestone:
                model_path = os.path.join(
                    self.output_dir, f"{self.prefix}_{step}.{self.suffix}"
                )
                if os.path.exists(model_path):
                    os.remove(model_path)
