#!/usr/bin/env python3
"""Generate and compare deterministic native-CLS RAE-NWM samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys

import torch


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def prepare_input(path: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(int(seed))
    payload = {
        "format_version": "etpr1-native-cls-parity-input-v1",
        "context": torch.randn(1, 4, 257, 768, generator=generator),
        "curr_delta": torch.tensor([[[0.125, -0.25, 0.375]]]),
        "rel_t": torch.tensor([8.0 / 128.0]),
        "initial_noise": torch.randn(1, 257, 768, generator=generator),
    }
    _save(payload, path)


def _ema_state(checkpoint: dict) -> dict:
    state = checkpoint.get("ema") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint must contain a non-empty ema mapping")
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
    }


def _implementation_modules(kind: str, upstream_root: Path | None):
    if kind == "etpr1":
        from vlnce_baselines.nwm.raenwm_core.models import CDiT_models
        from vlnce_baselines.nwm.raenwm_core.RAE.src.stage2.transport.transport import (
            ModelType,
            PathType,
            Sampler,
            Transport,
            WeightType,
        )

        return CDiT_models, ModelType, PathType, Sampler, Transport, WeightType
    if upstream_root is None:
        raise ValueError("--upstream-root is required for upstream sampling")
    sys.path.insert(0, str(upstream_root.resolve()))
    from models import CDiT_models
    from RAE.src.stage2.transport.transport import (
        ModelType,
        PathType,
        Sampler,
        Transport,
        WeightType,
    )

    return CDiT_models, ModelType, PathType, Sampler, Transport, WeightType


def sample(
    *,
    kind: str,
    upstream_root: Path | None,
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    device: str,
) -> None:
    modules = _implementation_modules(kind, upstream_root)
    CDiT_models, ModelType, PathType, Sampler, Transport, WeightType = modules
    torch_device = torch.device(device)
    model = CDiT_models["CDiT-B/2"](
        context_size=4,
        input_size=16,
        in_channels=768,
        learn_sigma=False,
        head_width=2048,
        head_depth=2,
        head_num_heads=16,
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    incompatible = model.load_state_dict(_ema_state(checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict EMA load failed: {incompatible}")
    model.requires_grad_(False).eval().to(torch_device)

    shift_dim = 768 * 257
    transport = Transport(
        model_type=ModelType.VELOCITY,
        path_type=PathType.LINEAR,
        loss_type=WeightType.VELOCITY,
        time_dist_type="uniform",
        time_dist_shift=math.sqrt(float(shift_dim) / 4096.0),
        train_eps=1.0e-3,
        sample_eps=1.0e-3,
    )
    sample_fn = Sampler(transport).sample_ode(
        sampling_method="euler",
        num_steps=10,
        atol=1.0e-6,
        rtol=1.0e-3,
        reverse=False,
    )
    inputs = _load(input_path)
    context = inputs["context"].to(torch_device)
    curr_delta = inputs["curr_delta"].to(torch_device)
    rel_t = inputs["rel_t"].to(torch_device)
    noise = inputs["initial_noise"].to(torch_device, dtype=context.dtype)
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        with torch.no_grad(), torch.autocast(
            device_type=torch_device.type,
            enabled=torch_device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            trajectory = sample_fn(
                noise,
                model,
                y=curr_delta.flatten(0, 1),
                x_cond=context,
                rel_t=rel_t,
            )
            tokens = torch.nan_to_num(trajectory[-1]).float().cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
    output = {
        "format_version": "etpr1-native-cls-parity-output-v1",
        "implementation": kind,
        "tokens": tokens,
        "cls_normalized": tokens[:, 0],
        "patch_tokens": tokens[:, 1:],
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        },
    }
    _save(output, output_path)
    print(json.dumps({
        "implementation": kind,
        "output": str(output_path),
        "shape": list(tokens.shape),
        "finite": bool(torch.isfinite(tokens).all()),
        "versions": output["versions"],
    }, sort_keys=True))


def compare(reference_path: Path, candidate_path: Path, report_path: Path) -> None:
    reference = _load(reference_path)
    candidate = _load(candidate_path)
    report = {
        "format_version": "etpr1-native-cls-parity-report-v1",
        "reference_versions": reference["versions"],
        "candidate_versions": candidate["versions"],
        "metrics": {},
    }
    for name in ("tokens", "cls_normalized", "patch_tokens"):
        expected = reference[name].float()
        actual = candidate[name].float()
        if actual.shape != expected.shape:
            raise ValueError(
                f"{name} shape differs: {tuple(actual.shape)} vs {tuple(expected.shape)}"
            )
        difference = (actual - expected).abs()
        cosine = torch.nn.functional.cosine_similarity(
            actual.reshape(actual.shape[0], -1),
            expected.reshape(expected.shape[0], -1),
            dim=1,
        ).mean()
        report["metrics"][name] = {
            "max_abs": float(difference.max()),
            "mean_abs": float(difference.mean()),
            "cosine": float(cosine),
            "finite": bool(torch.isfinite(actual).all()),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=240826)
    generate = subparsers.add_parser("sample")
    generate.add_argument("--implementation", choices=("etpr1", "upstream"), required=True)
    generate.add_argument("--upstream-root", type=Path)
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--input", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--device", default="cuda:0")
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--reference", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_input(args.output, args.seed)
    elif args.command == "sample":
        sample(
            kind=args.implementation,
            upstream_root=args.upstream_root,
            checkpoint_path=args.checkpoint,
            input_path=args.input,
            output_path=args.output,
            device=args.device,
        )
    else:
        compare(args.reference, args.candidate, args.report)


if __name__ == "__main__":
    main()
