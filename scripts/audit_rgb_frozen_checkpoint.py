#!/usr/bin/env python3
"""CPU audit of a trusted, locally produced B/C navigation checkpoint.

Example (run in the training machine's project environment)::

    python scripts/audit_rgb_frozen_checkpoint.py \
      --base pretrained/active_lookahead/base_iter14200.pth \
      --checkpoint data/logs/experiment/checkpoints/ckpt.iter2000.pth \
      --expected-align true --output data/logs/experiment/frozen_audit.json

Checks every serialized policy tensor, including buffers. Frozen pretrained
backbone tensors deliberately omitted by the project checkpoint format are
outside this file comparison; encoder provenance metadata is compared too.
Only load trusted project checkpoints: their config uses Python pickle.
"""
import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re

import torch


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(config, *keys):
    for key in keys:
        if isinstance(config, Mapping):
            config = config.get(key)
        else:
            config = getattr(config, key, None)
    return config


def _state(checkpoint):
    state = checkpoint.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("checkpoint state_dict must be a nonempty mapping")
    normalized = {}
    for name, value in state.items():
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise ValueError("policy state_dict must contain named tensors only")
        # Old dual-rank navigation uses net.module.*, frozen navigation net.*.
        key = "net." + name[len("net.module."):] if name.startswith("net.module.") else name
        if key in normalized:
            raise ValueError("policy prefix normalization collision: " + key)
        normalized[key] = value
    return normalized


def audit(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    base_path = Path(args.base).resolve(strict=True)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    # str paths are intentional: torch mmap loading requires filesystem paths.
    base = torch.load(str(base_path), map_location="cpu", mmap=True, weights_only=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
    report = {
        "ok": False,
        "base": str(base_path),
        "checkpoint": str(checkpoint_path),
        "base_sha256": _sha256(base_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "iteration": checkpoint.get("iteration"),
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "comparison_scope": "all serialized policy tensors and encoder provenance",
        "errors": [],
    }
    errors = report["errors"]
    original, candidate = _state(base), _state(checkpoint)
    missing = sorted(original.keys() - candidate.keys())
    extra = sorted(candidate.keys() - original.keys())
    differing = []
    for key in sorted(original.keys() & candidate.keys()):
        left, right = original[key], candidate[key]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            differing.append(key)
    report["policy"] = {
        "base_tensor_count": len(original),
        "checkpoint_tensor_count": len(candidate),
        "base_tensor_elements": sum(t.numel() for t in original.values()),
        "checkpoint_tensor_elements": sum(t.numel() for t in candidate.values()),
        "missing_keys": missing,
        "extra_keys": extra,
        "different_keys": differing,
        "all_equal": not (missing or extra or differing),
    }
    if not report["policy"]["all_equal"]:
        errors.append("navigation policy tensors changed or key sets differ")
    encoder_metadata_equal = base.get("rgb_encoder") == checkpoint.get("rgb_encoder")
    report["encoder_metadata_equal"] = encoder_metadata_equal
    report["encoder_metadata"] = checkpoint.get("rgb_encoder")
    if not encoder_metadata_equal or not checkpoint.get("rgb_encoder"):
        errors.append("encoder provenance metadata is missing or differs from base")

    saved = checkpoint.get("rgb_fusion_navigation_contract")
    config = checkpoint.get("config")
    config_contract = {
        "freeze_navigation_backbone": _get(config, "IL", "freeze_navigation_backbone"),
        "align_navigation_cls": _get(config, "MODEL", "RAENWM", "rgb_fusion_align_navigation_cls"),
        "condition_source_pose": _get(config, "MODEL", "RAENWM", "condition_source_pose"),
    }
    report["saved_contract"] = saved
    report["config_contract"] = config_contract
    report["context_metadata"] = checkpoint.get("raenwm_context_metadata")
    if not isinstance(saved, Mapping):
        errors.append("missing RGB fusion navigation contract")
        saved = {}
    if saved.get("freeze_navigation_backbone") is not True:
        errors.append("checkpoint does not declare a frozen navigation backbone")
    if not isinstance(saved.get("align_navigation_cls"), bool):
        errors.append("checkpoint alignment contract must be boolean")
    if args.expected_align is not None and saved.get("align_navigation_cls") != (args.expected_align == "true"):
        errors.append("checkpoint alignment differs from --expected-align")
    if saved.get("condition_source_pose") != args.expected_source:
        errors.append("checkpoint source pose differs from --expected-source")
    if any(saved.get(key) != value for key, value in config_contract.items()):
        errors.append("saved checkpoint contract and training config disagree")
    if _get(config, "MODEL", "ACTIVE_LOOKAHEAD", "enabled") is not False:
        errors.append("checkpoint must explicitly disable active lookahead E24")
    if _get(config, "MODEL", "RAENWM", "rgb_fusion_enabled") is not True:
        errors.append("checkpoint must explicitly enable RGB fusion")

    adapter = checkpoint.get("raenwm_rgb_fusion_adapter_state_dict")
    if not isinstance(adapter, Mapping) or not adapter:
        errors.append("missing RGB fusion adapter weights")
    else:
        tensor_items = [(key, value) for key, value in adapter.items() if torch.is_tensor(value)]
        invalid = [key for key, value in tensor_items if not torch.isfinite(value).all().item()]
        layer_indices = [int(match.group(1)) for key in adapter
                         for match in [re.fullmatch(r"residual\.layers\.(\d+)\.weight", key)] if match]
        final_layer = {}
        if not layer_indices:
            errors.append("missing RGB adapter residual linear layers")
        else:
            prefix = f"residual.layers.{max(layer_indices)}."
            for suffix in ("weight", "bias"):
                key = prefix + suffix
                value = adapter.get(key)
                if not torch.is_tensor(value):
                    errors.append("missing adapter final residual parameter: " + key)
                    continue
                final_layer[key] = {
                    "elements": value.numel(),
                    "nonzero_elements": torch.count_nonzero(value).item(),
                    "max_abs": value.abs().max().item() if key not in invalid else None,
                    "l2_norm": value.double().norm().item() if key not in invalid else None,
                }
            if not any(item["nonzero_elements"] for item in final_layer.values()):
                errors.append("adapter final residual layer remains zero")
        if invalid or len(tensor_items) != len(adapter):
            errors.append("adapter state contains nonfinite values or nontensor entries")
        report["adapter"] = {
            "tensor_count": len(tensor_items),
            "tensor_elements": sum(value.numel() for _, value in tensor_items),
            "nonfinite_keys": invalid,
            "final_residual_layer": final_layer,
        }
    report["ok"] = not errors
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-align", choices=("true", "false"))
    parser.add_argument("--expected-source", choices=("context_last", "query_current"), default="context_last")
    parser.add_argument("--output", help="Optional JSON report file; stdout always receives JSON")
    args = parser.parse_args()
    try:
        if args.output and Path(args.output).resolve() in {Path(args.base).resolve(), Path(args.checkpoint).resolve()}:
            raise ValueError("output cannot overwrite an input checkpoint")
        report = audit(args)
    except Exception as exc:
        report = {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    try:
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        report = {"ok": False, "errors": [f"report serialization failed: {exc}"]}
        payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        # Never write over checkpoints, even when validation raised above.
        if output.resolve() not in {Path(args.base).resolve(), Path(args.checkpoint).resolve()}:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
            temporary.write_text(payload + "\n", encoding="utf-8")
            os.replace(temporary, output)
    print(payload)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
