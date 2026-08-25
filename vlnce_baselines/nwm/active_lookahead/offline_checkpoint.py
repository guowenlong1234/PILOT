"""Recoverable, provenance-checked checkpoints for Stage-0 V2 offline training."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .offline_objective import (
    OfflineDecisionLossConfig,
    offline_loss_config_from_mapping,
)


FORMAT_VERSION = "stage0-topk-offline-checkpoint-v2"
DECISION_SOURCE_CONTRACT = "2026-07-20-stage0-v2-offline-net-correction-optimization-log.md"
SOURCE_CONTRACT = DECISION_SOURCE_CONTRACT
REQUIRED_FIELDS = frozenset(
    {
        "format_version",
        "source_contract",
        "training_stage",
        "global_step",
        "epoch",
        "batch_in_epoch",
        "future_head_state_dict",
        "future_head_optimizer_state_dict",
        "scheduler_state_dict",
        "model_config",
        "optimizer_config",
        "train_config",
        "loss_config",
        "dataset_provenance",
        "git_commit",
        "training_seed",
        "rng_state",
    }
)


def sha256_file(path: Any, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_offline_checkpoint_payload(
    head,
    optimizer,
    *,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    model_config: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    loss_config,
    dataset_provenance: Mapping[str, Any],
    git_commit: str,
    training_seed: int,
    rng_state: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = {
        "format_version": FORMAT_VERSION,
        "source_contract": DECISION_SOURCE_CONTRACT,
        "training_stage": "offline",
        "global_step": int(global_step),
        "epoch": int(epoch),
        "batch_in_epoch": int(batch_in_epoch),
        "future_head_state_dict": head.state_dict(),
        "future_head_optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "model_config": dict(model_config),
        "optimizer_config": dict(optimizer_config),
        "train_config": dict(train_config),
        "loss_config": loss_config.to_dict(),
        "dataset_provenance": dict(dataset_provenance),
        "git_commit": str(git_commit),
        "training_seed": int(training_seed),
        "rng_state": dict(rng_state),
    }
    validate_offline_checkpoint_payload(payload)
    return payload


def validate_offline_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"Offline checkpoint is missing fields: {sorted(missing)}")
    forbidden = [key for key in payload if "policy_state" in str(key).lower()]
    if forbidden:
        raise ValueError(f"Offline checkpoint must not contain policy state: {forbidden}")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("Offline checkpoint format version mismatch")
    parsed_loss = offline_loss_config_from_mapping(payload["loss_config"])
    if not isinstance(parsed_loss, OfflineDecisionLossConfig):
        raise ValueError("only the retained E24 decision loss is supported")
    if payload["source_contract"] != DECISION_SOURCE_CONTRACT:
        raise ValueError("Offline checkpoint source contract mismatch")
    if payload["training_stage"] != "offline":
        raise ValueError("Offline checkpoint requires training_stage=offline")
    if payload["future_head_optimizer_state_dict"] is None:
        raise ValueError("Offline checkpoint must contain optimizer state")
    if not isinstance(payload["optimizer_config"], Mapping) or not payload["optimizer_config"]:
        raise ValueError("Offline checkpoint must contain optimizer config")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping) or set(rng_state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }:
        raise ValueError("Offline checkpoint has an invalid RNG state contract")
    if not torch.is_tensor(rng_state["torch_cpu"]):
        raise ValueError("Offline checkpoint torch CPU RNG state must be a tensor")
    if not isinstance(rng_state["torch_cuda"], list) or any(
        not torch.is_tensor(value) for value in rng_state["torch_cuda"]
    ):
        raise ValueError("Offline checkpoint torch CUDA RNG states must be tensors")
    provenance = payload["dataset_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("dataset_provenance must be a mapping")
    for field in ("schema_version", "manifest_sha256", "root", "split"):
        if field not in provenance or str(provenance[field]).strip() == "":
            raise ValueError(f"dataset_provenance requires non-empty {field}")
    for field in ("global_step", "epoch", "batch_in_epoch"):
        if int(payload[field]) < 0:
            raise ValueError(f"{field} must be non-negative")
    deployment = payload.get("deployment_only")
    if deployment is not None:
        if not isinstance(deployment, Mapping) or not deployment.get(
            "resume_forbidden", False
        ):
            raise ValueError("deployment_only must forbid training resume")


def save_offline_checkpoint(path: Any, payload: Mapping[str, Any]) -> None:
    validate_offline_checkpoint_payload(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(destination)


def load_offline_checkpoint(
    path: Any,
    head,
    optimizer,
    *,
    expected_dataset_manifest_sha256: Optional[str] = None,
    map_location: Any = "cpu",
) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, Mapping):
        raise ValueError("Offline checkpoint payload must be a mapping")
    validate_offline_checkpoint_payload(payload)
    deployment = payload.get("deployment_only", {})
    posthoc = payload.get("posthoc_weight_average", {})
    if deployment.get("resume_forbidden", False) or posthoc.get(
        "resume_forbidden", False
    ):
        raise ValueError("deployment-only checkpoint cannot resume training")
    if expected_dataset_manifest_sha256 is not None:
        actual = payload["dataset_provenance"]["manifest_sha256"]
        if actual != expected_dataset_manifest_sha256:
            raise ValueError("Offline dataset manifest SHA256 mismatch")
    head.load_state_dict(payload["future_head_state_dict"], strict=True)
    optimizer.load_state_dict(payload["future_head_optimizer_state_dict"])
    return dict(payload)
