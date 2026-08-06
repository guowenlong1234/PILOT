#!/usr/bin/env python3

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch

from vlnce_baselines.models.checkpoint_utils import (
    is_rgb_backbone_key,
    validate_rgb_checkpoint_metadata,
)


IMG_LINEAR_MARKER = "img_embeddings.img_linear."
IMG_LINEAR_SUFFIXES = ("weight", "bias")
RESIDUAL_MLP_MARKER = "rgb_encoder.cls_residual_mlp.layers."
RESIDUAL_MLP_SUFFIXES = (
    "0.weight",
    "0.bias",
    "2.weight",
    "2.bias",
    "4.weight",
    "4.bias",
)


def _load(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return torch.load(path, map_location="cpu")


def _state_dict(checkpoint):
    for key in ("state_dict", "model"):
        state = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if isinstance(state, dict):
            return state
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("checkpoint must contain a state mapping")


def _parameters_by_suffix(checkpoint, marker, expected_suffixes):
    parameters = {}
    expected_suffixes = set(expected_suffixes)
    for key, value in _state_dict(checkpoint).items():
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        if suffix not in expected_suffixes:
            continue
        if suffix in parameters:
            raise ValueError(f"duplicate {marker} suffix: {suffix}")
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"visual parameter is invalid: {key}")
        parameters[suffix] = value.cpu()
    if set(parameters) != expected_suffixes:
        missing = sorted(expected_suffixes - set(parameters))
        extra = sorted(set(parameters) - expected_suffixes)
        raise ValueError(
            f"{marker} parameter mismatch; missing={missing}, extra={extra}"
        )
    return parameters


def _img_linear(checkpoint):
    return _parameters_by_suffix(
        checkpoint,
        IMG_LINEAR_MARKER,
        IMG_LINEAR_SUFFIXES,
    )


def _residual_mlp(checkpoint):
    return _parameters_by_suffix(
        checkpoint,
        RESIDUAL_MLP_MARKER,
        RESIDUAL_MLP_SUFFIXES,
    )


def _assert_online_checkpoint(
    checkpoint, expected_iteration, training_state=None
):
    state = _state_dict(checkpoint)
    backbone_keys = [key for key in state if is_rgb_backbone_key(key)]
    if backbone_keys:
        raise ValueError(
            "online checkpoint contains the frozen DINO backbone: "
            + ", ".join(sorted(backbone_keys))
        )
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("online checkpoint is missing config for metadata audit")
    validate_rgb_checkpoint_metadata(checkpoint, config)
    metadata = checkpoint.get("rgb_encoder")
    if checkpoint.get("iteration") != expected_iteration:
        raise ValueError(
            "checkpoint iteration mismatch: "
            f"expected {expected_iteration}, got {checkpoint.get('iteration')}"
        )
    if training_state is None:
        training_state = checkpoint
    if training_state.get("iteration", expected_iteration) != expected_iteration:
        raise ValueError(
            "training-state iteration mismatch: "
            f"expected {expected_iteration}, "
            f"got {training_state.get('iteration')}"
        )
    optimizer = training_state.get("optim_state")
    if not isinstance(optimizer, Mapping) or not optimizer.get("state"):
        raise ValueError("online checkpoint optimizer state is empty")
    scheduler = training_state.get("scheduler_state")
    if not isinstance(scheduler, Mapping) or not scheduler:
        raise ValueError("online checkpoint scheduler_state is missing or empty")
    last_epoch = scheduler.get("last_epoch")
    if (
        isinstance(last_epoch, bool)
        or not isinstance(last_epoch, int)
        or last_epoch < 0
    ):
        raise ValueError(
            "online checkpoint scheduler_state.last_epoch must be a "
            f"non-negative integer, got {last_epoch!r}"
        )
    return metadata


def audit_pretrain(path, initial_img_linear_path=None):
    checkpoint = _load(path)
    state = _state_dict(checkpoint)
    img_linear = _img_linear(checkpoint)
    required_prefixes = (
        "bert.embeddings.",
        "bert.global_encoder.",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(str(key).startswith(prefix) for key in state)
    ]
    if missing:
        raise ValueError(
            "pretrain checkpoint is not a complete formal model; "
            f"missing key prefixes: {missing}"
        )
    if initial_img_linear_path is not None:
        initial = _img_linear(_load(initial_img_linear_path))
        for suffix in IMG_LINEAR_SUFFIXES:
            if torch.equal(img_linear[suffix], initial[suffix]):
                raise ValueError(
                    f"formal pretrain did not update img_linear {suffix}"
                )
        print(
            "AUDIT_PRETRAIN_UPDATE_PASS "
            f"parameters={len(img_linear)} initial={initial_img_linear_path}"
        )
    print(f"AUDIT_PRETRAIN_PASS parameters={len(img_linear)} path={path}")


def audit_sft(
    pretrain_path, checkpoint_path, iteration, training_state_path=None
):
    pretrain = _img_linear(_load(pretrain_path))
    checkpoint = _load(checkpoint_path)
    online = _img_linear(checkpoint)
    residual = _residual_mlp(checkpoint)
    training_state = (
        _load(training_state_path) if training_state_path else None
    )
    _assert_online_checkpoint(checkpoint, iteration, training_state)
    for suffix in IMG_LINEAR_SUFFIXES:
        difference = (online[suffix] - pretrain[suffix]).abs()
        changed = int(torch.count_nonzero(difference))
        if changed == 0:
            raise ValueError(f"SFT did not update img_linear {suffix}")
        print(
            f"AUDIT_SFT_IMG_LINEAR suffix={suffix} changed={changed} "
            f"max_abs={float(difference.max())}"
        )
    if not any(torch.count_nonzero(residual[suffix]) for suffix in ("4.weight", "4.bias")):
        raise ValueError("SFT did not update the zero-initialized CLS residual MLP")
    print(f"AUDIT_SFT_PASS path={checkpoint_path}")


def audit_frozen(before_path, after_path, iteration):
    before_checkpoint = _load(before_path)
    after_checkpoint = _load(after_path)
    before_parameters = {
        **{f"img_linear.{k}": v for k, v in _img_linear(before_checkpoint).items()},
        **{f"residual_mlp.{k}": v for k, v in _residual_mlp(before_checkpoint).items()},
    }
    after_parameters = {
        **{f"img_linear.{k}": v for k, v in _img_linear(after_checkpoint).items()},
        **{f"residual_mlp.{k}": v for k, v in _residual_mlp(after_checkpoint).items()},
    }
    before_metadata = before_checkpoint.get("rgb_encoder")
    after_metadata = _assert_online_checkpoint(after_checkpoint, iteration)
    if after_metadata != before_metadata:
        raise ValueError("GRPO changed RGB encoder metadata")
    for name, before_value in before_parameters.items():
        torch.testing.assert_close(
            after_parameters[name],
            before_value,
            rtol=0,
            atol=0,
        )
    print(f"AUDIT_FROZEN_PASS path={after_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("checkpoint")
    pretrain.add_argument("--initial-img-linear")

    sft = subparsers.add_parser("sft")
    sft.add_argument("pretrain_checkpoint")
    sft.add_argument("online_checkpoint")
    sft.add_argument("--iteration", type=int, required=True)
    sft.add_argument("--train-state")

    frozen = subparsers.add_parser("frozen")
    frozen.add_argument("before_checkpoint")
    frozen.add_argument("after_checkpoint")
    frozen.add_argument("--iteration", type=int, required=True)

    args = parser.parse_args()
    if args.command == "pretrain":
        audit_pretrain(args.checkpoint, args.initial_img_linear)
    elif args.command == "sft":
        audit_sft(
            args.pretrain_checkpoint,
            args.online_checkpoint,
            args.iteration,
            args.train_state,
        )
    else:
        audit_frozen(
            args.before_checkpoint,
            args.after_checkpoint,
            args.iteration,
        )


if __name__ == "__main__":
    main()
