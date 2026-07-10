#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch


PROJECTION_MARKER = "rgb_projection."
EXPECTED_SUFFIXES = (
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


def _projection_by_suffix(checkpoint):
    projection = {}
    for key, value in _state_dict(checkpoint).items():
        if PROJECTION_MARKER not in key:
            continue
        suffix = key.split(PROJECTION_MARKER, 1)[1]
        if suffix in EXPECTED_SUFFIXES:
            if suffix in projection:
                raise ValueError(f"duplicate projection suffix: {suffix}")
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f"projection parameter is invalid: {key}")
            projection[suffix] = value.cpu()
    if set(projection) != set(EXPECTED_SUFFIXES):
        missing = sorted(set(EXPECTED_SUFFIXES) - set(projection))
        extra = sorted(set(projection) - set(EXPECTED_SUFFIXES))
        raise ValueError(
            f"projection parameter mismatch; missing={missing}, extra={extra}"
        )
    return projection


def _assert_online_checkpoint(checkpoint, expected_iteration):
    state = _state_dict(checkpoint)
    backbone_keys = [
        key for key in state if ".rgb_encoder.backbone." in key
    ]
    if backbone_keys:
        raise ValueError("online checkpoint contains the frozen DINO backbone")
    metadata = checkpoint.get("rgb_encoder")
    expected_metadata = {
        "type": "rae_dinov2",
        "raw_output_size": 768,
        "output_size": 512,
    }
    if not isinstance(metadata, dict):
        raise ValueError("online checkpoint is missing RGB encoder metadata")
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"RGB encoder metadata {key} mismatch: {metadata.get(key)!r}"
            )
    if checkpoint.get("iteration") != expected_iteration:
        raise ValueError(
            "checkpoint iteration mismatch: "
            f"expected {expected_iteration}, got {checkpoint.get('iteration')}"
        )
    optimizer = checkpoint.get("optim_state")
    if not isinstance(optimizer, dict) or not optimizer.get("state"):
        raise ValueError("online checkpoint optimizer state is empty")
    return metadata


def audit_pretrain(path, initial_projection_path=None):
    checkpoint = _load(path)
    state = _state_dict(checkpoint)
    projection = _projection_by_suffix(checkpoint)
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
    if initial_projection_path is not None:
        initial = _projection_by_suffix(_load(initial_projection_path))
        for suffix in EXPECTED_SUFFIXES:
            if torch.equal(projection[suffix], initial[suffix]):
                raise ValueError(
                    f"formal pretrain did not update projection parameter {suffix}"
                )
        print(
            "AUDIT_PRETRAIN_UPDATE_PASS "
            f"parameters={len(projection)} initial={initial_projection_path}"
        )
    print(f"AUDIT_PRETRAIN_PASS parameters={len(projection)} path={path}")


def audit_sft(pretrain_path, checkpoint_path, iteration):
    pretrain = _projection_by_suffix(_load(pretrain_path))
    checkpoint = _load(checkpoint_path)
    online = _projection_by_suffix(checkpoint)
    _assert_online_checkpoint(checkpoint, iteration)
    for suffix in EXPECTED_SUFFIXES:
        difference = (online[suffix] - pretrain[suffix]).abs()
        changed = int(torch.count_nonzero(difference))
        if changed == 0:
            raise ValueError(f"SFT did not update projection parameter {suffix}")
        print(
            f"AUDIT_SFT_PARAMETER suffix={suffix} changed={changed} "
            f"max_abs={float(difference.max())}"
        )
    print(f"AUDIT_SFT_PASS path={checkpoint_path}")


def audit_frozen(before_path, after_path, iteration):
    before_checkpoint = _load(before_path)
    after_checkpoint = _load(after_path)
    before = _projection_by_suffix(before_checkpoint)
    after = _projection_by_suffix(after_checkpoint)
    before_metadata = before_checkpoint.get("rgb_encoder")
    after_metadata = _assert_online_checkpoint(after_checkpoint, iteration)
    if after_metadata != before_metadata:
        raise ValueError("GRPO changed RGB encoder metadata")
    for suffix in EXPECTED_SUFFIXES:
        torch.testing.assert_close(
            after[suffix],
            before[suffix],
            rtol=0,
            atol=0,
        )
    print(f"AUDIT_FROZEN_PASS path={after_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrain = subparsers.add_parser("pretrain")
    pretrain.add_argument("checkpoint")
    pretrain.add_argument("--initial-projection")

    sft = subparsers.add_parser("sft")
    sft.add_argument("pretrain_checkpoint")
    sft.add_argument("online_checkpoint")
    sft.add_argument("--iteration", type=int, required=True)

    frozen = subparsers.add_parser("frozen")
    frozen.add_argument("before_checkpoint")
    frozen.add_argument("after_checkpoint")
    frozen.add_argument("--iteration", type=int, required=True)

    args = parser.parse_args()
    if args.command == "pretrain":
        audit_pretrain(args.checkpoint, args.initial_projection)
    elif args.command == "sft":
        audit_sft(
            args.pretrain_checkpoint,
            args.online_checkpoint,
            args.iteration,
        )
    else:
        audit_frozen(
            args.before_checkpoint,
            args.after_checkpoint,
            args.iteration,
        )


if __name__ == "__main__":
    main()
