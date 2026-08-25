"""Exact state-dict hashing for the frozen Stage-0 base."""

import hashlib
import json
from pathlib import Path

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def capture_base_tensor_manifest(named_modules):
    tensors = {}
    for module_name, module in named_modules.items():
        if module is None:
            continue
        for tensor_name, tensor in module.state_dict().items():
            if not torch.is_tensor(tensor):
                continue
            full_name = f"{module_name}.{tensor_name}"
            if full_name in tensors:
                raise ValueError(f"duplicate frozen tensor name: {full_name}")
            tensors[full_name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
    return {
        "schema_version": 1,
        "tensor_count": len(tensors),
        "tensors": dict(sorted(tensors.items())),
    }


def compare_base_tensor_manifests(before, after):
    before_tensors = before.get("tensors", {})
    after_tensors = after.get("tensors", {})
    missing = sorted(set(before_tensors) - set(after_tensors))
    unexpected = sorted(set(after_tensors) - set(before_tensors))
    changed = sorted(
        name
        for name in set(before_tensors) & set(after_tensors)
        if before_tensors[name] != after_tensors[name]
    )
    return {
        "exact_match": not missing and not unexpected and not changed,
        "tensor_count_before": len(before_tensors),
        "tensor_count_after": len(after_tensors),
        "missing": missing,
        "unexpected": unexpected,
        "changed": changed,
    }


def write_base_freeze_report(path, *, before, after, comparison, iteration, rank):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "iteration": int(iteration),
        "rank": int(rank),
        "comparison": comparison,
        "before_manifest": before,
        "after_manifest": after,
    }
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
