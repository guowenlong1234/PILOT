#!/usr/bin/env python3
"""Run one deterministic h=1/2/3 batch through real NWM/CWP/E24 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vlnce_baselines.models.encoders.rae_dinov2_encoder import (
    RaeDinov2RgbEncoder,
)
from vlnce_baselines.nwm.active_lookahead.dino_cwp_future import (
    build_dino_cwp_nwm_future_rollout,
    build_per_query_initial_noise,
    load_dino_cwp_predictor,
)
from vlnce_baselines.nwm.active_lookahead.joint_e24 import load_e24_joint_head
from vlnce_baselines.nwm.active_lookahead.types import CandidateQ0
from vlnce_baselines.nwm.etp_adapter import (
    RaeLatentTargetRequest,
    RaeSourceContextSnapshot,
)
from vlnce_baselines.nwm.runtime import NwmPredictionRuntime
from vlnce_baselines.nwm.raenwm_core.models import pack_cls_patch


NWM_SHA = "38b24af13b76ba8faef367559244c3a0401e0557e7c299870c273cbee8a07064"
STAT_SHA = "84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59"
CWP_SHA = "6a45291219907dd027203d224f3f8400631651a83bd01c45b1dea55d93ec0979"
E24_SHA = "bae7a9664000235dfc6fb66b43a8a0a38e7645b876e6a6369f732eb7bf404ed8"
E24_BASE_MANIFEST_SHA = (
    "a2e84e88c568633a5a86d9cb2643d1b1cfe0d9b302cc4eb3a1b66dc0802f48c6"
)


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--none-threshold", type=float, default=1.0)
    return parser.parse_args()


def _runtime_config(root: Path, seed: int):
    return SimpleNamespace(
        predict_cls_token=True,
        token_count=257,
        config_path=str(root / "configs/nwm/raenwm_mp3d_fresh_cls.yaml"),
        checkpoint_path=str(
            root / "pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar"
        ),
        checkpoint_sha256=NWM_SHA,
        stat_path=str(root / "pretrained/raenwm_stage0/stat.pt"),
        stat_sha256=STAT_SHA,
        context_size=4,
        max_buffer_size=64,
        metric_waypoint_spacing=0.24975892673356762,
        pos_eps=1.0e-3,
        yaw_eps=1.0e-3,
        static_run_k=3,
        max_horizon=64.0,
        num_steps=10,
        final_only_euler=False,
        noise_seed=int(seed),
    )


def _active_config(horizon: int, none_threshold: float):
    return SimpleNamespace(
        dino_cwp_none_threshold=float(none_threshold),
        lookahead_horizon_steps=int(horizon),
        future_aggregation="endpoint",
        rollout_failure_policy="deepest_valid",
        rollout_noise_policy="per_query_v1",
    )


def main() -> int:
    args = _arguments()
    if not 0.0 <= float(args.none_threshold) <= 1.0:
        raise ValueError("--none-threshold must be in [0,1]")
    root = args.asset_root.expanduser().resolve()
    device = torch.device(args.device)
    runtime = NwmPredictionRuntime(
        _runtime_config(root, args.noise_seed), device
    )
    runtime.reset(1)
    cwp, cwp_metadata = load_dino_cwp_predictor(
        root / "pretrained/active_lookahead/dino_cwp_best.pt",
        expected_sha256=CWP_SHA,
        device=device,
    )
    e24, e24_metadata = load_e24_joint_head(
        str(root / "pretrained/active_lookahead/e24_avg3.pth"),
        device=device,
        expected_checkpoint_sha256=E24_SHA,
        expected_source_base_manifest_sha256=E24_BASE_MANIFEST_SHA,
    )
    e24.eval()
    visual_encoder = RaeDinov2RgbEncoder(
        root / "pretrained/rae_dinov2_with_registers_base",
        device=device,
        precision="bf16",
    )
    rgb_generator = torch.Generator(device="cpu").manual_seed(20260830)
    rgb = torch.randint(
        0,
        256,
        (4, 224, 224, 3),
        dtype=torch.uint8,
        generator=rgb_generator,
    )
    raw_cls, _nav_cls, raw_patch = (
        visual_encoder.forward_with_raw_cls_and_patch_latents({"rgb": rgb})
    )
    context_tokens = pack_cls_patch(
        runtime.normalizer.normalize_cls(raw_cls),
        runtime.normalizer.normalize_patch(raw_patch),
    ).detach().cpu()
    del visual_encoder, raw_cls, raw_patch
    snapshot = RaeSourceContextSnapshot(
        source_front_vp="asset-smoke-front",
        source_high_level_step=11,
        context_latents=context_tokens,
        source_position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        source_yaw=float(np.pi),
    )
    q0_request = RaeLatentTargetRequest(
        env_index=0,
        ghost_vp="asset-smoke-ghost",
        snapshot=snapshot,
        target_position=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        target_yaw=float(np.pi),
        horizon_override=4.0,
    )
    trainer = SimpleNamespace(
        device=device,
        raenwm_runtime=runtime,
        dino_cwp_future_predictor=cwp,
        gmaps=[SimpleNamespace()],
        config=SimpleNamespace(MODEL=SimpleNamespace(task_type="r2r")),
        _active_lookahead_config=lambda: _active_config(
            1, args.none_threshold
        ),
    )
    q0_noise = build_per_query_initial_noise(
        trainer, [(0, snapshot.source_high_level_step, q0_request.ghost_vp, 0)]
    )
    q0_prediction = runtime.predict_latent_targets(
        [q0_request], initial_noise=q0_noise
    )
    q0_meta = runtime.last_batch.records[0]
    q0_patch = q0_prediction.pred_latent[0].detach().cpu().to(torch.float16)
    q0 = CandidateQ0(
        contract_version="r1_post_update_ghost_mean_cached_v1",
        position_source="r1_post_update_ghost_mean",
        ghost_vp=q0_request.ghost_vp,
        source_front_vp=snapshot.source_front_vp,
        source_high_level_step=snapshot.source_high_level_step,
        source_position=snapshot.source_position.copy(),
        source_yaw=snapshot.source_yaw,
        target_position=q0_request.target_position.copy(),
        target_yaw=float(q0_request.target_yaw),
        condition=(
            q0_meta.condition.dx,
            q0_meta.condition.dy,
            q0_meta.condition.dtheta,
            q0_meta.condition.rel_t,
        ),
        horizon=float(q0_meta.horizon),
        source_context=snapshot,
        predicted_patch_cpu_fp16=q0_patch,
        patch_quantization_max_abs=0.0,
        candidate_forward_m=1.0,
        current_view_index=0,
    )

    reports = []
    for horizon in (1, 2, 3):
        trainer._active_lookahead_config = lambda h=horizon: _active_config(
            h, args.none_threshold
        )
        repeated = []
        for _repeat in range(2):
            result = build_dino_cwp_nwm_future_rollout(
                trainer,
                active_envs=[0],
                records_by_env=[[q0]],
                topk=1,
                feature_dim=768,
                reference=torch.zeros((), device=device),
            )
            future = result.future_tokens.expand(1, 5, -1, -1).contiguous()
            valid = torch.zeros(1, 5, dtype=torch.bool, device=device)
            valid[:, 0] = result.future_valid_mask[:, 0]
            with torch.inference_mode():
                delta = e24.forward_topk_from_log_probs(
                    torch.zeros(1, 5, 768, device=device),
                    torch.zeros(1, 4, 768, device=device),
                    future,
                    torch.zeros(1, 5, device=device),
                    valid,
                    text_token_mask=torch.ones(
                        1, 4, dtype=torch.bool, device=device
                    ),
                    candidate_geometry=torch.zeros(1, 5, 3, device=device),
                ).delta
            repeated.append((result, delta))
        result, delta = repeated[0]
        repeated_result, repeated_delta = repeated[1]
        for name in (
            "future_tokens",
            "future_conditions",
            "future_valid_mask",
            "full_horizon_mask",
            "fallback_mask",
            "realized_depths",
        ):
            if not torch.equal(getattr(result, name), getattr(repeated_result, name)):
                raise RuntimeError(f"h={horizon} {name} is not deterministic")
        if not torch.equal(delta, repeated_delta):
            raise RuntimeError(f"h={horizon} E24 delta is not deterministic")
        if not bool(torch.isfinite(delta).all()):
            raise FloatingPointError(f"E24 produced non-finite h={horizon} delta")
        reports.append({
            "horizon": horizon,
            "valid": bool(result.future_valid_mask[0, 0]),
            "full_horizon": bool(result.full_horizon_mask[0, 0]),
            "fallback": bool(result.fallback_mask[0, 0]),
            "realized_depth": int(result.realized_depths[0, 0]),
            "future_abs_mean": float(result.future_tokens.float().abs().mean()),
            "e24_delta": float(delta[0, 0]),
            "repeat_exact": True,
        })
    incomplete = [
        report for report in reports
        if not report["valid"]
        or not report["full_horizon"]
        or report["realized_depth"] != report["horizon"]
    ]
    if incomplete:
        raise RuntimeError(f"real asset rollout did not reach full horizon: {incomplete}")
    print(json.dumps({
        "device": str(device),
        "nwm_checkpoint_sha256": NWM_SHA,
        "cwp_checkpoint_sha256": cwp_metadata["checkpoint_sha256"],
        "e24_checkpoint_sha256": e24_metadata["checkpoint_sha256"],
        "cwp_none_threshold": float(args.none_threshold),
        "reports": reports,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
