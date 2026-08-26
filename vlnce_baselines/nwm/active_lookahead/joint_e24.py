"""Gradient-decoupled E24 SFT using predicted q1 future tokens only."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .dino_cwp_future import (
    PREDICTED_FUTURE_DIAGNOSTIC_NAMES,
    build_dino_cwp_nwm_future_tokens,
)
from .geometry import candidate_q0_geometry
from .offline_objective import (
    OfflineDecisionLossConfig,
    offline_decision_aware_loss,
)
from .offline_checkpoint import sha256_file
from .online_e24 import (
    E24_AVG3_SOURCE_STEPS,
    E24_FACTORY,
    quantize_like_offline_cache,
)
from .native_cls_adapter import Top5NativeClsAdapter
from .persistent_q0 import persistent_q0_to_dict
from .residual_head import InterleavedCrossModalTopKFutureLogitResidualHead
from .topk_query import executable_ghost_indices, stable_topk_ghost_indices


@dataclass(frozen=True)
class E24JointDecisionPack:
    """One rollout step of detached E24 inputs and optional SFT targets."""

    env_indices: torch.Tensor
    owner_embeddings: torch.Tensor
    text_tokens: torch.Tensor
    text_token_mask: torch.Tensor
    future_tokens: torch.Tensor
    future_valid_mask: torch.Tensor
    topk_slot_mask: torch.Tensor
    candidate_geometry: torch.Tensor
    base_logits: torch.Tensor
    ghost_valid_mask: torch.Tensor
    ghost_global_indices: torch.Tensor
    topk_base_indices: torch.Tensor
    topk_global_indices: torch.Tensor
    q1_conditions: torch.Tensor | None = None
    full_base_logits: torch.Tensor | None = None
    full_valid_mask: torch.Tensor | None = None
    teacher_actions: torch.Tensor | None = None
    teacher_rank_in_topk: torch.Tensor | None = None
    teacher_valid: torch.Tensor | None = None
    teacher_stop: torch.Tensor | None = None
    no_vp_left: torch.Tensor | None = None
    base_stop: torch.Tensor | None = None
    teacher_base_index: torch.Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.env_indices.shape[0])

    @property
    def topk_valid_mask(self) -> torch.Tensor:
        return self.topk_slot_mask & self.future_valid_mask

    def to_cpu_fp16(self) -> "E24JointDecisionPack":
        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is None:
                values[name] = None
            elif value.is_floating_point():
                values[name] = value.detach().to(device="cpu", dtype=torch.float16)
            else:
                values[name] = value.detach().to(device="cpu")
        return E24JointDecisionPack(**values)


class E24JointTrainModule(nn.Module):
    """Standard-forward wrapper so the custom E24 scorer can use DDP."""

    def __init__(
        self,
        head: nn.Module,
        cls_adapter: Top5NativeClsAdapter | None = None,
    ) -> None:
        super().__init__()
        self.head = head
        self.cls_adapter = cls_adapter

    @property
    def delta_max(self) -> float:
        return float(self.head.delta_max)

    def forward(
        self,
        owner_embeddings: torch.Tensor,
        text_tokens: torch.Tensor,
        future_tokens: torch.Tensor,
        selected_log_probs: torch.Tensor,
        candidate_mask: torch.Tensor,
        text_token_mask: torch.Tensor,
        candidate_geometry: torch.Tensor,
        q1_conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.cls_adapter is not None:
            if q1_conditions is None:
                raise ValueError("native CLS E24 forward requires q1 conditions")
            future_tokens, _diagnostics = self.cls_adapter.adapt_tokens(
                future_tokens,
                q1_conditions,
            )
        delta = self.head.forward_topk_from_log_probs(
            owner_embeddings,
            text_tokens,
            future_tokens,
            selected_log_probs,
            candidate_mask,
            text_token_mask=text_token_mask,
            candidate_geometry=candidate_geometry,
        ).delta
        # A sparse rank can reach the final synchronized DDP replay round with
        # only a dummy row.  Keep every head parameter in that zero-valued
        # graph so gradients accumulated under no_sync() are all reduced on
        # the final round instead of being classified as unused.
        parameter_zero = delta.new_zeros(())
        for parameter in self.parameters():
            parameter_zero = parameter_zero + parameter.reshape(-1)[0] * 0.0
        return delta + parameter_zero


def _load_e24_joint_weights_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("E24 joint init checkpoint must contain a mapping")
    return dict(payload)


def load_e24_joint_head(
    checkpoint_path: str,
    *,
    device: torch.device,
    expected_checkpoint_sha256: str,
    expected_source_base_manifest_sha256: str,
    topk: int = 5,
):
    """Load the retained avg3 artifact as trainable weights-only init."""

    if int(topk) != 5:
        raise ValueError("E24 joint weights-only initialization requires topk=5")
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"E24 joint init checkpoint does not exist: {path}")
    expected_sha = str(expected_checkpoint_sha256).strip().lower()
    if len(expected_sha) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError("E24 joint expected checkpoint SHA256 is invalid")
    sha_before = sha256_file(path)
    if sha_before != expected_sha:
        raise ValueError(
            "E24 joint checkpoint SHA256 mismatch: "
            f"expected={expected_sha} actual={sha_before}"
        )
    payload = _load_e24_joint_weights_mapping(path)
    if sha256_file(path) != sha_before:
        raise RuntimeError("E24 joint init checkpoint changed while loading")
    if payload.get("format_version") != "stage0-topk-offline-checkpoint-v2":
        raise ValueError("E24 joint init checkpoint has the wrong format")
    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping) or model_config.get("factory") != E24_FACTORY:
        raise ValueError("E24 joint init checkpoint uses the wrong scorer factory")
    kwargs = model_config.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise ValueError("E24 joint init checkpoint lacks scorer kwargs")
    provenance = payload.get("dataset_provenance", {})
    if str(provenance.get("base_manifest_sha256", "")) != str(
        expected_source_base_manifest_sha256
    ):
        raise ValueError("E24 joint init source base manifest SHA256 mismatch")
    posthoc = payload.get("posthoc_weight_average", {})
    source_steps = tuple(
        int(source.get("global_step", -1)) for source in posthoc.get("sources", ())
    )
    if source_steps != E24_AVG3_SOURCE_STEPS:
        raise ValueError(
            f"E24 joint init requires avg3 source steps {E24_AVG3_SOURCE_STEPS}"
        )
    state_dict = payload.get("future_head_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("E24 joint init checkpoint lacks future_head_state_dict")

    # Deliberately ignore the old optimizer/scheduler/RNG state.  An averaged
    # artifact may mark resume_forbidden because those states are not
    # meaningful after averaging; that does not forbid trainable weights-only
    # initialization in the new joint stage.
    head = InterleavedCrossModalTopKFutureLogitResidualHead(**dict(kwargs))
    head.load_state_dict(state_dict, strict=True)
    head.to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    head.train()
    metadata = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha_before,
        "source_steps": list(source_steps),
        "global_step": int(payload.get("global_step", -1)),
        "parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "weights_only_initialization": True,
        "resume_forbidden_ignored_for_weights_only": bool(
            posthoc.get("resume_forbidden", False)
        ),
        "delta_scale": 1.0,
        "model_kwargs": dict(kwargs),
        "source_base_manifest_sha256": str(
            expected_source_base_manifest_sha256
        ),
    }
    return head, metadata


def joint_action_scale(iteration: int, start_iteration: int, warmup_iterations: int) -> float:
    if warmup_iterations < 0:
        raise ValueError("E24 action warmup iterations must be non-negative")
    if warmup_iterations == 0:
        return 1.0
    progress = max(0, int(iteration) - int(start_iteration))
    return min(1.0, float(progress) / float(warmup_iterations))


def _forward_head(head, pack: E24JointDecisionPack) -> torch.Tensor:
    head_dtype = next(head.parameters()).dtype
    valid = pack.topk_valid_mask
    ghost_valid = pack.ghost_valid_mask.to(torch.bool)
    masked_base = pack.base_logits.detach().to(dtype=head_dtype).masked_fill(
        ~ghost_valid, -torch.inf
    )
    base_log_probs = torch.log_softmax(masked_base, dim=1)
    owner_embeddings = pack.owner_embeddings.to(dtype=head_dtype)
    selected_log_probs = owner_embeddings.new_zeros(
        pack.owner_embeddings.shape[:2]
    )
    rows, slots = valid.nonzero(as_tuple=True)
    if rows.numel():
        selected = pack.topk_base_indices[rows, slots]
        selected_log_probs[rows, slots] = base_log_probs[rows, selected].to(
            dtype=head_dtype
        )
    if hasattr(head, "forward_topk_from_log_probs"):
        output = head.forward_topk_from_log_probs(
            owner_embeddings,
            pack.text_tokens.to(dtype=head_dtype),
            pack.future_tokens.to(dtype=head_dtype),
            selected_log_probs,
            valid,
            text_token_mask=pack.text_token_mask,
            candidate_geometry=pack.candidate_geometry.to(dtype=head_dtype),
        ).delta
    else:
        output = head(
            owner_embeddings,
            pack.text_tokens.to(dtype=head_dtype),
            pack.future_tokens.to(dtype=head_dtype),
            selected_log_probs,
            valid,
            pack.text_token_mask,
            pack.candidate_geometry.to(dtype=head_dtype),
            None
            if pack.q1_conditions is None
            else pack.q1_conditions.to(dtype=head_dtype),
        )
    return output.masked_fill(~valid, 0.0)


def build_e24_joint_step(
    trainer: Any,
    *,
    nav_inputs: Mapping[str, Any],
    nav_outs: Mapping[str, torch.Tensor],
    txt_embeds: torch.Tensor,
    txt_masks: torch.Tensor,
) -> tuple[
    torch.Tensor,
    list[int],
    E24JointDecisionPack | None,
    dict[str, float],
]:
    """Build source-selected futures and return detached action/replay inputs."""

    cfg = trainer._active_lookahead_config()
    head = trainer._e24_joint_head_state_module()
    topk = int(cfg.offline_topk)
    source = str(getattr(cfg, "source", "dino_cwp_nwm")).strip().lower()
    if source != "dino_cwp_nwm":
        raise ValueError("E24 joint SFT only supports predicted q1 source dino_cwp_nwm")
    base_logits = nav_outs["global_logits"]
    native_cls = bool(
        getattr(trainer.raenwm_runtime, "predict_cls_token", False)
    )
    global_delta = torch.zeros_like(base_logits)
    active_envs: list[int] = []
    ranked_by_env: list[tuple[int, ...]] = []
    ghost_globals_by_env: list[tuple[int, ...]] = []
    records_by_env: list[list[Any | None]] = []

    for env_index, ids in enumerate(nav_inputs["gmap_vp_ids"]):
        row_logits = base_logits[env_index, : len(ids)]
        if not native_cls and int(row_logits.detach().argmax()) == 0:
            continue
        ranked = stable_topk_ghost_indices(ids, row_logits.detach(), k=topk)
        ghosts = executable_ghost_indices(ids)
        if not ranked or not ghosts:
            raise RuntimeError("E24 joint base MOVE row has no executable ghost")
        records = [
            trainer.gmaps[env_index].select_persistent_q0(ids[index])
            for index in ranked
        ]
        active_envs.append(env_index)
        ranked_by_env.append(ranked)
        ghost_globals_by_env.append(ghosts)
        records_by_env.append(records)

    if not active_envs:
        return (
            global_delta,
            [0] * trainer.envs.num_envs,
            None,
            {name: 0.0 for name in PREDICTED_FUTURE_DIAGNOSTIC_NAMES},
        )

    batch = len(active_envs)
    feature_dim = int(txt_embeds.shape[-1])
    future = base_logits.new_zeros((batch, topk, 257, feature_dim))
    future_valid = torch.zeros((batch, topk), dtype=torch.bool, device=trainer.device)
    future, q1_conditions, future_valid, source_diagnostics = (
        build_dino_cwp_nwm_future_tokens(
        trainer,
        active_envs=active_envs,
        records_by_env=records_by_env,
        topk=topk,
        feature_dim=feature_dim,
        reference=base_logits,
        )
    )

    owner = base_logits.new_zeros((batch, topk, feature_dim))
    slot_mask = torch.zeros((batch, topk), dtype=torch.bool, device=trainer.device)
    geometry = base_logits.new_zeros((batch, topk, 3))
    max_ghosts = max(len(indices) for indices in ghost_globals_by_env)
    ghost_logits = base_logits.new_full((batch, max_ghosts), -torch.inf)
    ghost_valid = torch.zeros((batch, max_ghosts), dtype=torch.bool, device=trainer.device)
    ghost_globals = torch.full(
        (batch, max_ghosts), -1, dtype=torch.long, device=trainer.device
    )
    topk_local = torch.full((batch, topk), -1, dtype=torch.long, device=trainer.device)
    topk_global = torch.full((batch, topk), -1, dtype=torch.long, device=trainer.device)
    full_base_logits = None
    full_valid_mask = None
    if native_cls:
        full_base_logits = base_logits.index_select(
            0, torch.tensor(active_envs, dtype=torch.long, device=trainer.device)
        ).detach()
        full_valid_mask = torch.zeros_like(full_base_logits, dtype=torch.bool)

    for row, (env_index, ranked, ghosts, records) in enumerate(
        zip(active_envs, ranked_by_env, ghost_globals_by_env, records_by_env)
    ):
        count = len(ranked)
        slot_mask[row, :count] = True
        owner[row, :count] = nav_outs["gmap_embeds"][env_index, list(ranked)].detach()
        ghost_logits[row, : len(ghosts)] = base_logits[env_index, list(ghosts)].detach()
        ghost_valid[row, : len(ghosts)] = True
        ghost_globals[row, : len(ghosts)] = torch.tensor(
            ghosts, dtype=torch.long, device=trainer.device
        )
        topk_global[row, :count] = torch.tensor(
            ranked, dtype=torch.long, device=trainer.device
        )
        if full_valid_mask is not None:
            full_valid_mask[row, : len(nav_inputs["gmap_vp_ids"][env_index])] = True
        local_lookup = {global_index: local for local, global_index in enumerate(ghosts)}
        topk_local[row, :count] = torch.tensor(
            [local_lookup[index] for index in ranked],
            dtype=torch.long,
            device=trainer.device,
        )
        geometry[row, :count] = candidate_q0_geometry(
            [
                None if record is None else persistent_q0_to_dict(record)
                for record in records
            ]
        ).to(device=trainer.device, dtype=base_logits.dtype)

    active_index = torch.tensor(active_envs, dtype=torch.long, device=trainer.device)
    pack = E24JointDecisionPack(
        env_indices=active_index,
        owner_embeddings=owner.detach(),
        text_tokens=txt_embeds.index_select(0, active_index).detach(),
        text_token_mask=txt_masks.index_select(0, active_index).detach(),
        future_tokens=future.detach(),
        future_valid_mask=future_valid,
        topk_slot_mask=slot_mask,
        candidate_geometry=geometry.detach(),
        base_logits=ghost_logits.detach(),
        ghost_valid_mask=ghost_valid,
        ghost_global_indices=ghost_globals,
        topk_base_indices=topk_local,
        topk_global_indices=topk_global,
        q1_conditions=q1_conditions.detach() if native_cls else None,
        full_base_logits=full_base_logits,
        full_valid_mask=full_valid_mask,
    )

    train_module_getter = getattr(trainer, "_e24_joint_train_module", None)
    rollout_module = train_module_getter() if train_module_getter else None
    if rollout_module is None:
        rollout_module = head
    was_training = rollout_module.training
    rollout_module.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=trainer.device.type, enabled=False
    ):
        deltas = _forward_head(rollout_module, pack)
    rollout_module.train(was_training)
    scale = float(getattr(cfg, "e24_train_delta_scale", 1.0))
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("E24 joint train delta scale must be finite and non-negative")
    applied = (deltas * scale).clamp(-float(head.delta_max), float(head.delta_max))
    for row, (env_index, ranked) in enumerate(zip(active_envs, ranked_by_env)):
        for slot, global_index in enumerate(ranked):
            global_delta[env_index, global_index] = applied[row, slot]
    query_counts = [0] * trainer.envs.num_envs
    for row, env_index in enumerate(active_envs):
        query_counts[env_index] = int(future_valid[row].sum())
    return global_delta, query_counts, pack, source_diagnostics


def attach_e24_joint_targets(
    pack: E24JointDecisionPack,
    teacher_actions: torch.Tensor,
    no_vp_left: Sequence[bool],
) -> E24JointDecisionPack:
    batch = pack.batch_size
    ranks = torch.full((batch,), -1, dtype=torch.long, device=pack.env_indices.device)
    teacher_valid = torch.zeros(batch, dtype=torch.bool, device=pack.env_indices.device)
    teacher_stop = torch.zeros(batch, dtype=torch.bool, device=pack.env_indices.device)
    no_vp = torch.zeros(batch, dtype=torch.bool, device=pack.env_indices.device)
    teacher_base = torch.full(
        (batch,), -1, dtype=torch.long, device=pack.env_indices.device
    )
    base_stop = torch.zeros(batch, dtype=torch.bool, device=pack.env_indices.device)
    full_teacher = torch.full(
        (batch,), -100, dtype=torch.long, device=pack.env_indices.device
    )
    for row, env_index_tensor in enumerate(pack.env_indices):
        env_index = int(env_index_tensor)
        teacher = int(teacher_actions[env_index])
        if teacher >= 0 and not bool(no_vp_left[env_index]):
            full_teacher[row] = teacher
        no_vp[row] = bool(no_vp_left[env_index]) or teacher == -100
        if teacher == 0:
            teacher_stop[row] = True
            teacher_valid[row] = True
            continue
        if teacher < 0 or no_vp[row]:
            continue
        matches = (pack.ghost_global_indices[row] == teacher).nonzero(as_tuple=False)
        if not matches.numel():
            continue
        teacher_valid[row] = True
        teacher_base[row] = int(matches[0, 0])
        topk_matches = (pack.topk_global_indices[row] == teacher).nonzero(
            as_tuple=False
        )
        if topk_matches.numel():
            ranks[row] = int(topk_matches[0, 0])
    return replace(
        pack,
        teacher_rank_in_topk=ranks,
        teacher_valid=teacher_valid,
        teacher_stop=teacher_stop,
        no_vp_left=no_vp,
        base_stop=base_stop,
        teacher_base_index=teacher_base,
        teacher_actions=full_teacher,
    )


def collate_e24_joint_packs(
    packs: Sequence[E24JointDecisionPack],
) -> dict[str, torch.Tensor] | None:
    packs = [pack for pack in packs if pack.batch_size and pack.teacher_valid is not None]
    if not packs:
        return None
    max_ghosts = max(int(pack.base_logits.shape[1]) for pack in packs)
    max_text = max(int(pack.text_tokens.shape[1]) for pack in packs)

    def pad_ghost(value: torch.Tensor, fill_value: float | int | bool):
        if value.shape[1] == max_ghosts:
            return value
        padding = value.new_full(
            (value.shape[0], max_ghosts - value.shape[1]), fill_value
        )
        return torch.cat((value, padding), dim=1)

    def pad_text(value: torch.Tensor, fill_value: float | bool):
        if value.shape[1] == max_text:
            return value
        shape = (value.shape[0], max_text - value.shape[1], *value.shape[2:])
        return torch.cat((value, value.new_full(shape, fill_value)), dim=1)

    batch = {
        "owner_embeddings": torch.cat([pack.owner_embeddings for pack in packs]),
        "text_tokens": torch.cat([pad_text(pack.text_tokens, 0.0) for pack in packs]),
        "text_token_mask": torch.cat(
            [pad_text(pack.text_token_mask, False) for pack in packs]
        ),
        "future_tokens": torch.cat([pack.future_tokens for pack in packs]),
        "topk_slot_mask": torch.cat([pack.topk_slot_mask for pack in packs]),
        "topk_valid_mask": torch.cat([pack.topk_valid_mask for pack in packs]),
        "candidate_q0_geometry": torch.cat(
            [pack.candidate_geometry for pack in packs]
        ),
        "base_logits": torch.cat(
            [pad_ghost(pack.base_logits, -torch.inf) for pack in packs]
        ),
        "ghost_valid_mask": torch.cat(
            [pad_ghost(pack.ghost_valid_mask, False) for pack in packs]
        ),
        "topk_base_indices": torch.cat([pack.topk_base_indices for pack in packs]),
        "teacher_rank_in_topk": torch.cat(
            [pack.teacher_rank_in_topk for pack in packs]
        ),
        "teacher_valid": torch.cat([pack.teacher_valid for pack in packs]),
        "teacher_stop": torch.cat([pack.teacher_stop for pack in packs]),
        "no_vp_left": torch.cat([pack.no_vp_left for pack in packs]),
        "base_stop": torch.cat([pack.base_stop for pack in packs]),
        "teacher_base_index": torch.cat(
            [pack.teacher_base_index for pack in packs]
        ),
    }
    if any(pack.q1_conditions is not None for pack in packs):
        if not all(pack.q1_conditions is not None for pack in packs):
            raise ValueError("cannot mix native and legacy E24 replay packs")
        batch["q1_conditions"] = torch.cat(
            [pack.q1_conditions for pack in packs]
        )
    if any(pack.full_base_logits is not None for pack in packs):
        if not all(
            pack.full_base_logits is not None
            and pack.full_valid_mask is not None
            and pack.teacher_actions is not None
            for pack in packs
        ):
            raise ValueError("native replay pack is missing full-logit fields")
        max_full = max(int(pack.full_base_logits.shape[1]) for pack in packs)

        def pad_full(value: torch.Tensor, fill_value):
            if int(value.shape[1]) == max_full:
                return value
            padding = value.new_full(
                (value.shape[0], max_full - value.shape[1]), fill_value
            )
            return torch.cat((value, padding), dim=1)

        batch["full_base_logits"] = torch.cat(
            [pad_full(pack.full_base_logits, -torch.inf) for pack in packs]
        )
        batch["full_valid_mask"] = torch.cat(
            [pad_full(pack.full_valid_mask, False) for pack in packs]
        )
        batch["topk_global_indices"] = torch.cat(
            [pack.topk_global_indices for pack in packs]
        )
        batch["teacher_actions"] = torch.cat(
            [pack.teacher_actions for pack in packs]
        )
    return batch


def slice_e24_joint_batch(
    batch: Mapping[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    return {name: value[start:stop] for name, value in batch.items()}


def e24_joint_batch_to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device, *, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in batch.items():
        target_dtype = dtype if value.is_floating_point() else value.dtype
        result[name] = value.to(device=device, dtype=target_dtype, non_blocking=True)
    return result


def forward_e24_joint_batch(
    train_module,
    batch: Mapping[str, torch.Tensor],
    *,
    loss_config: OfflineDecisionLossConfig,
):
    owner = batch["owner_embeddings"]
    text = batch["text_tokens"]
    future = batch["future_tokens"]
    valid = batch["topk_valid_mask"].to(torch.bool)
    teacher_valid = batch["teacher_valid"].to(torch.bool)
    teacher_stop = batch["teacher_stop"].to(torch.bool)
    no_vp = batch["no_vp_left"].to(torch.bool)
    base_stop = batch["base_stop"].to(torch.bool)
    candidate_mask = valid & (
        teacher_valid & ~teacher_stop & ~no_vp & ~base_stop
    )[:, None]
    masked_base = batch["base_logits"].detach().masked_fill(
        ~batch["ghost_valid_mask"].to(torch.bool), -torch.inf
    )
    selected_log_probs = owner.new_zeros(owner.shape[:2])
    rows, slots = candidate_mask.nonzero(as_tuple=True)
    if rows.numel():
        all_log_probs = torch.log_softmax(masked_base, dim=1)
        selected = batch["topk_base_indices"][rows, slots]
        selected_log_probs[rows, slots] = all_log_probs[rows, selected].to(owner.dtype)
        deltas = train_module(
            owner,
            text,
            future,
            selected_log_probs,
            candidate_mask,
            batch["text_token_mask"],
            batch["candidate_q0_geometry"],
            batch.get("q1_conditions"),
        )
    else:
        # Still call the DDP wrapper so ranks without eligible decisions join
        # the same reducer sequence as ranks with real replay rows.
        deltas = train_module(
            owner,
            text,
            future,
            selected_log_probs,
            candidate_mask,
            batch["text_token_mask"],
            batch["candidate_q0_geometry"],
            batch.get("q1_conditions"),
        )
    result = offline_decision_aware_loss(
        deltas,
        batch["teacher_rank_in_topk"],
        topk_valid_mask=valid,
        teacher_valid=teacher_valid,
        teacher_stop=teacher_stop,
        no_vp_left=no_vp,
        base_stop=base_stop,
        base_logits=batch["base_logits"],
        ghost_valid_mask=batch["ghost_valid_mask"],
        topk_base_indices=batch["topk_base_indices"],
        teacher_base_index=batch["teacher_base_index"],
        config=loss_config,
    )
    return result, deltas


@dataclass(frozen=True)
class NativeAdjustedLossResult:
    loss_sum: torch.Tensor
    row_count: int
    adjusted_logits: torch.Tensor
    deltas: torch.Tensor


def forward_native_adjusted_batch(
    train_module,
    batch: Mapping[str, torch.Tensor],
    *,
    delta_scale: float = 1.0,
) -> NativeAdjustedLossResult:
    """Recompute native adapter+E24 and score complete navigation logits."""

    required = (
        "full_base_logits",
        "full_valid_mask",
        "topk_global_indices",
        "teacher_actions",
        "q1_conditions",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"native adjusted replay is missing fields: {missing}")
    full_valid = batch["full_valid_mask"].to(torch.bool)
    topk_valid = batch["topk_valid_mask"].to(torch.bool)
    base = batch["full_base_logits"].detach().masked_fill(
        ~full_valid, -torch.inf
    )
    indices = batch["topk_global_indices"].to(torch.long)
    index_in_range = (indices >= 0) & (indices < base.shape[1])
    candidate_mask = topk_valid & index_in_range
    safe_indices = indices.clamp(min=0, max=max(0, base.shape[1] - 1))
    base_log_probs = torch.log_softmax(base, dim=1)
    owner = batch["owner_embeddings"].detach()
    selected_log_probs = owner.new_zeros(
        candidate_mask.shape
    )
    rows, slots = candidate_mask.nonzero(as_tuple=True)
    if rows.numel():
        selected_log_probs[rows, slots] = base_log_probs[
            rows, safe_indices[rows, slots]
        ].to(selected_log_probs.dtype)
    deltas = train_module(
        owner,
        batch["text_tokens"].detach(),
        batch["future_tokens"].detach(),
        selected_log_probs,
        candidate_mask,
        batch["text_token_mask"],
        batch["candidate_q0_geometry"].detach(),
        batch["q1_conditions"].detach(),
    )
    module = getattr(train_module, "module", train_module)
    delta_max = float(module.delta_max)
    scale = float(delta_scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("native adjusted delta scale must be finite and non-negative")
    applied = (deltas * scale).clamp(-delta_max, delta_max)
    additions = torch.zeros_like(base).scatter_add(
        1,
        safe_indices,
        applied.masked_fill(~candidate_mask, 0.0).to(base.dtype),
    )
    adjusted = base + additions
    teachers = batch["teacher_actions"].to(torch.long)
    teacher_in_range = (teachers >= 0) & (teachers < adjusted.shape[1])
    safe_teachers = teachers.clamp(min=0, max=max(0, adjusted.shape[1] - 1))
    teacher_is_valid = full_valid.gather(1, safe_teachers[:, None]).squeeze(1)
    eligible = candidate_mask.any(dim=1) & teacher_in_range & teacher_is_valid
    if bool(eligible.any()):
        loss_sum = F.cross_entropy(
            adjusted[eligible],
            teachers[eligible],
            reduction="sum",
        )
    else:
        loss_sum = deltas.sum() * 0.0
    return NativeAdjustedLossResult(
        loss_sum=loss_sum,
        row_count=int(eligible.sum().detach().cpu()),
        adjusted_logits=adjusted,
        deltas=deltas,
    )


def native_adjusted_row_count(batch: Mapping[str, torch.Tensor]) -> int:
    required = (
        "full_valid_mask",
        "topk_global_indices",
        "topk_valid_mask",
        "teacher_actions",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"native adjusted replay is missing fields: {missing}")
    full_valid = batch["full_valid_mask"].to(torch.bool)
    indices = batch["topk_global_indices"].to(torch.long)
    topk_valid = batch["topk_valid_mask"].to(torch.bool)
    candidate_mask = topk_valid & (indices >= 0) & (indices < full_valid.shape[1])
    teachers = batch["teacher_actions"].to(torch.long)
    teacher_in_range = (teachers >= 0) & (teachers < full_valid.shape[1])
    safe_teachers = teachers.clamp(min=0, max=max(0, full_valid.shape[1] - 1))
    teacher_is_valid = full_valid.gather(1, safe_teachers[:, None]).squeeze(1)
    return int(
        (candidate_mask.any(dim=1) & teacher_in_range & teacher_is_valid)
        .sum()
        .detach()
        .cpu()
    )


def normalized_native_adjusted_loss(
    result: NativeAdjustedLossResult,
    *,
    global_row_count: float,
    world_size: int,
    loss_weight: float,
) -> torch.Tensor:
    if global_row_count <= 0:
        return result.loss_sum * 0.0
    return (
        result.loss_sum
        * float(world_size)
        * float(loss_weight)
        / float(global_row_count)
    )


def make_e24_joint_dummy_batch(
    *,
    topk: int,
    feature_dim: int = 768,
    token_count: int = 257,
    native_cls: bool = False,
) -> dict[str, torch.Tensor]:
    batch = {
        "owner_embeddings": torch.zeros(1, topk, feature_dim),
        "text_tokens": torch.zeros(1, 1, feature_dim),
        "text_token_mask": torch.ones(1, 1, dtype=torch.bool),
        "future_tokens": torch.zeros(1, topk, token_count, feature_dim),
        "topk_slot_mask": torch.zeros(1, topk, dtype=torch.bool),
        "topk_valid_mask": torch.zeros(1, topk, dtype=torch.bool),
        "candidate_q0_geometry": torch.zeros(1, topk, 3),
        "base_logits": torch.zeros(1, 1),
        "ghost_valid_mask": torch.ones(1, 1, dtype=torch.bool),
        "topk_base_indices": torch.full((1, topk), -1, dtype=torch.long),
        "teacher_rank_in_topk": torch.full((1,), -1, dtype=torch.long),
        "teacher_valid": torch.zeros(1, dtype=torch.bool),
        "teacher_stop": torch.zeros(1, dtype=torch.bool),
        "no_vp_left": torch.ones(1, dtype=torch.bool),
        "base_stop": torch.zeros(1, dtype=torch.bool),
        "teacher_base_index": torch.full((1,), -1, dtype=torch.long),
    }
    if native_cls:
        batch["q1_conditions"] = torch.zeros(1, topk, 4)
        batch["full_base_logits"] = torch.zeros(1, 1)
        batch["full_valid_mask"] = torch.ones(1, 1, dtype=torch.bool)
        batch["topk_global_indices"] = torch.full(
            (1, topk), -1, dtype=torch.long
        )
        batch["teacher_actions"] = torch.full(
            (1,), -100, dtype=torch.long
        )
    return batch


def e24_joint_denominators(
    batch: Mapping[str, torch.Tensor], config: OfflineDecisionLossConfig
) -> dict[str, float]:
    valid = batch["topk_valid_mask"].to(torch.bool)
    ranks = batch["teacher_rank_in_topk"].to(torch.long)
    teacher_valid = batch["teacher_valid"].to(torch.bool)
    teacher_stop = batch["teacher_stop"].to(torch.bool)
    no_vp = batch["no_vp_left"].to(torch.bool)
    base_stop = batch["base_stop"].to(torch.bool)
    teachers = batch["teacher_base_index"].to(torch.long)
    safe_ranks = ranks.clamp(min=0, max=max(0, valid.shape[1] - 1))
    positive_valid = (ranks >= 0) & valid.gather(1, safe_ranks[:, None]).squeeze(1)
    eligible = teacher_valid & ~teacher_stop & ~no_vp & ~base_stop & valid.any(dim=1)
    signed_rows = eligible & ((ranks < 0) | positive_valid)
    decision_rows = eligible & positive_valid & (teachers >= 0)
    absent_rows = eligible & ~positive_valid & (teachers >= 0)
    masked = batch["base_logits"].float().masked_fill(
        ~batch["ghost_valid_mask"].to(torch.bool), -torch.inf
    )
    base_actions = masked.argmax(dim=1)
    row_weights = torch.where(
        base_actions == teachers,
        torch.full_like(teachers, config.correct_row_weight, dtype=torch.float32),
        torch.full_like(teachers, config.wrong_row_weight, dtype=torch.float32),
    )
    return {
        "signed": float(signed_rows.sum()),
        "decision_weight": float(row_weights[decision_rows].sum()),
        "regularization": float((valid & decision_rows[:, None]).sum()),
        "absent_noop": float((valid & absent_rows[:, None]).sum()),
    }


def e24_joint_diagnostic_totals(
    batch: Mapping[str, torch.Tensor],
    deltas: torch.Tensor,
    *,
    delta_max: float,
) -> dict[str, float]:
    """Return additive replay diagnostics suitable for distributed reduction."""

    valid = batch["topk_valid_mask"].to(torch.bool)
    ranks = batch["teacher_rank_in_topk"].to(torch.long)
    teachers = batch["teacher_base_index"].to(torch.long)
    teacher_valid = batch["teacher_valid"].to(torch.bool)
    eligible = (
        teacher_valid
        & ~batch["teacher_stop"].to(torch.bool)
        & ~batch["no_vp_left"].to(torch.bool)
        & ~batch["base_stop"].to(torch.bool)
        & (teachers >= 0)
        & valid.any(dim=1)
    )
    safe_ranks = ranks.clamp(min=0, max=max(0, valid.shape[1] - 1))
    selected_slots = batch["topk_slot_mask"].to(torch.bool)
    teacher_in_topk = eligible & (ranks >= 0) & selected_slots.gather(
        1, safe_ranks[:, None]
    ).squeeze(1)

    ghost_valid = batch["ghost_valid_mask"].to(torch.bool)
    base = batch["base_logits"].float().masked_fill(~ghost_valid, -torch.inf)
    adjusted = base.clone()
    rows, slots = valid.nonzero(as_tuple=True)
    selected = batch["topk_base_indices"][rows, slots].to(torch.long)
    if rows.numel():
        adjusted[rows, selected] += deltas.detach().float()[rows, slots]
    base_correct = base.argmax(dim=1) == teachers
    adjusted_correct = adjusted.argmax(dim=1) == teachers
    eligible_count = int(eligible.sum().detach().cpu())
    valid_delta = deltas.detach()[valid]
    saturation_threshold = float(delta_max) * (1.0 - 1e-3)
    return {
        "teacher_eligible": float(eligible_count),
        "teacher_in_top5": float(teacher_in_topk.sum().detach().cpu()),
        "future_slots": float(selected_slots.sum().detach().cpu()),
        "future_valid": float(valid.sum().detach().cpu()),
        "fix": float((eligible & ~base_correct & adjusted_correct).sum().detach().cpu()),
        "harm": float((eligible & base_correct & ~adjusted_correct).sum().detach().cpu()),
        "delta_sum": float(valid_delta.float().sum().cpu()),
        "delta_abs_sum": float(valid_delta.float().abs().sum().cpu()),
        "delta_count": float(valid_delta.numel()),
        "delta_saturated": float(
            (valid_delta.float().abs() >= saturation_threshold).sum().cpu()
        ),
    }


def normalized_e24_joint_loss(
    result,
    *,
    global_denominators: Mapping[str, float],
    config: OfflineDecisionLossConfig,
    world_size: int,
    loss_weight: float,
) -> torch.Tensor:
    zero = result.loss * 0.0
    scale = float(world_size) * float(loss_weight)
    signed_count = result.teacher_present_count + result.teacher_absent_count
    total = zero
    if global_denominators["signed"] > 0:
        total = total + (
            config.signed_weight
            * result.signed_loss
            * signed_count
            / global_denominators["signed"]
        )
    if global_denominators["decision_weight"] > 0:
        total = total + (
            config.final_weight * result.final_loss * result.decision_weight_sum
            + config.pair_weight * result.pair_loss * result.decision_weight_sum
        ) / global_denominators["decision_weight"]
    if global_denominators["regularization"] > 0:
        total = total + (
            config.regularization_weight
            * result.regularization_loss
            * result.regularization_candidate_count
            / global_denominators["regularization"]
        )
    if global_denominators["absent_noop"] > 0:
        total = total + (
            config.absent_noop_weight
            * result.absent_noop_loss
            * result.absent_noop_candidate_count
            / global_denominators["absent_noop"]
        )
    return total * scale
