#!/usr/bin/env python3
"""Run a deterministic real-asset q0 -> CWP -> q1 -> E24 forward chain."""

from types import SimpleNamespace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from vlnce_baselines.nwm.active_lookahead.dino_cwp_future import (
    decode_dino_cwp_top1,
    latent_to_patch_tokens,
    load_dino_cwp_predictor,
    waypoint_to_world_position,
)
from vlnce_baselines.nwm.active_lookahead.joint_e24 import load_e24_joint_head
from vlnce_baselines.nwm.active_lookahead.online_e24 import (
    quantize_like_offline_cache,
)
from vlnce_baselines.nwm.etp_adapter import RaeLatentTargetRequest
from vlnce_baselines.nwm.runtime import NwmPredictionRuntime


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def runtime_config(root: Path):
    return SimpleNamespace(
        context_size=4,
        num_steps=10,
        final_only_euler=False,
        checkpoint_path=str(root / "pretrained/raenwm_stage0/checkpoint_step_70000.pth.tar"),
        checkpoint_sha256="392fe02045f7c11f006e1efb822914eee5826b8c6fdb2553c87e7de7c1c4df36",
        head_checkpoint_path=str(root / "pretrained/raenwm_stage0/nwm_heads.pt"),
        head_checkpoint_sha256="a4d396021b8bf670c44565c383db1c9288c3bd8988624fa0a59064b6a81dc6e2",
        stat_path=str(root / "pretrained/raenwm_stage0/stat.pt"),
        stat_sha256="84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59",
        config_path=str(root / "configs/nwm/raenwm_mp3d.yaml"),
        max_buffer_size=64,
        metric_waypoint_spacing=0.24975892673356762,
        pos_eps=1.0e-3,
        yaw_eps=1.0e-3,
        static_run_k=3,
        max_horizon=64.0,
        noise_seed=0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    device = torch.device(args.device)
    torch.manual_seed(24)

    runtime = NwmPredictionRuntime(runtime_config(root), device)
    runtime.reset(1)
    for step in range(4):
        raw = torch.full(
            (1, 768, 16, 16), 0.05 * (step + 1), dtype=torch.float32
        )
        runtime.update_contexts(
            raw,
            [np.asarray([0.0, 0.0, -0.25 * step], dtype=np.float32)],
            [float(np.pi)],
        )
    snapshot = runtime.source_context_snapshot(
        0, source_front_vp="fixed-front", source_high_level_step=3
    )
    if snapshot is None:
        raise RuntimeError("fixed four-frame source context was not created")

    q0_position = np.asarray([0.25, 0.0, -1.0], dtype=np.float32)
    q0_request = RaeLatentTargetRequest(
        env_index=0,
        ghost_vp="g-fixed",
        snapshot=snapshot,
        target_position=q0_position,
    )
    noise = torch.linspace(
        -1.0, 1.0, 768 * 16 * 16, device=device, dtype=torch.float32
    ).reshape(1, 768, 16, 16)
    q0 = runtime.predict_latent_targets([q0_request], initial_noise=noise)
    q0_repeat = runtime.predict_latent_targets([q0_request], initial_noise=noise)
    torch.testing.assert_close(q0.pred_latent, q0_repeat.pred_latent, rtol=0, atol=0)
    torch.testing.assert_close(q0.pred_cls, q0_repeat.pred_cls, rtol=0, atol=0)

    cwp, cwp_meta = load_dino_cwp_predictor(
        root / "pretrained/active_lookahead/dino_cwp_best.pt",
        expected_sha256="6a45291219907dd027203d224f3f8400631651a83bd01c45b1dea55d93ec0979",
        device=device,
    )
    patches = latent_to_patch_tokens(q0.pred_latent)
    with torch.inference_mode():
        cwp_output = cwp(patches)
    prediction = decode_dino_cwp_top1(cwp_output, none_threshold=0.3)[0]
    q0_heading_deg = 0.0
    q1_position = waypoint_to_world_position(
        q0_position,
        heading_deg=q0_heading_deg,
        local_angle_deg=prediction.local_angle_deg,
        distance_m=prediction.distance_m,
    )
    q0_record = runtime.adapter.build_target_record(
        env_index=0,
        ghost_vp="g-fixed",
        source_position=snapshot.source_position,
        source_yaw=snapshot.source_yaw,
        target_position=q0_position,
    )
    q1_request = RaeLatentTargetRequest(
        env_index=0,
        ghost_vp="g-fixed",
        snapshot=snapshot,
        target_position=q1_position,
        target_yaw=float(np.pi + np.deg2rad(prediction.local_angle_deg)),
        horizon_override=(
            q0_record.horizon
            + prediction.distance_m / runtime.adapter.config.metric_waypoint_spacing
        ),
    )
    q1 = runtime.predict_latent_targets([q1_request], initial_noise=noise)
    future = quantize_like_offline_cache(
        torch.cat(
            (
                q1.pred_cls[:, None, :],
                latent_to_patch_tokens(q1.pred_latent),
            ),
            dim=1,
        )
    )
    if tuple(future.shape) != (1, 257, 768):
        raise RuntimeError(f"wrong future token shape: {tuple(future.shape)}")

    e24, e24_meta = load_e24_joint_head(
        root / "pretrained/active_lookahead/e24_avg3.pth",
        device=device,
        expected_checkpoint_sha256="bae7a9664000235dfc6fb66b43a8a0a38e7645b876e6a6369f732eb7bf404ed8",
        expected_source_base_manifest_sha256="a2e84e88c568633a5a86d9cb2643d1b1cfe0d9b302cc4eb3a1b66dc0802f48c6",
    )
    e24.eval()
    owner = torch.linspace(-0.5, 0.5, 5 * 768, device=device).reshape(1, 5, 768)
    text = torch.linspace(-0.25, 0.25, 7 * 768, device=device).reshape(1, 7, 768)
    futures = future[:, None].expand(-1, 5, -1, -1).contiguous()
    valid = torch.ones((1, 5), dtype=torch.bool, device=device)
    base_log_probs = torch.log_softmax(
        torch.tensor([[2.0, 1.5, 1.0, 0.5, 0.0]], device=device), dim=1
    )
    geometry = torch.tensor(
        [[[0.5, 0.0, 1.0], [0.4, 0.5, 0.866], [0.3, 0.866, 0.5],
          [0.2, 1.0, 0.0], [0.1, 0.866, -0.5]]],
        device=device,
    )
    with torch.inference_mode():
        e24_output = e24.forward_topk_from_log_probs(
            owner,
            text,
            futures,
            base_log_probs,
            valid,
            text_token_mask=torch.ones((1, 7), dtype=torch.bool, device=device),
            candidate_geometry=geometry,
        )
    if not torch.isfinite(e24_output.delta).all():
        raise FloatingPointError("E24 delta is not finite")
    if float(e24_output.delta.abs().max()) > 1.0:
        raise ValueError("E24 delta exceeds [-1,1]")

    model = runtime.predictor.bundle.model
    payload = {
        "format_version": "etpr1-active-lookahead-real-assets-v1",
        "q0_latent_sha256": tensor_sha256(q0.pred_latent),
        "q0_cls_sha256": tensor_sha256(q0.pred_cls),
        "q1_latent_sha256": tensor_sha256(q1.pred_latent),
        "q1_cls_sha256": tensor_sha256(q1.pred_cls),
        "future_sha256": tensor_sha256(future),
        "e24_delta_sha256": tensor_sha256(e24_output.delta),
        "e24_delta": e24_output.delta.detach().cpu().tolist(),
        "cwp": {
            "pred_none": prediction.pred_none,
            "none_prob": prediction.none_prob,
            "local_angle_deg": prediction.local_angle_deg,
            "distance_m": prediction.distance_m,
            "checkpoint_sha256": cwp_meta["checkpoint_sha256"],
        },
        "e24": {
            "checkpoint_sha256": e24_meta["checkpoint_sha256"],
            "source_steps": e24_meta["source_steps"],
        },
        "frozen": {
            "nwm_body": all(not p.requires_grad for p in model.parameters()),
            "nwm_heads": all(
                not p.requires_grad for p in runtime.predictor.heads.parameters()
            ),
            "dino_cwp": all(not p.requires_grad for p in cwp.parameters()),
        },
        "cuda_peak_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        ),
    }
    if not all(payload["frozen"].values()):
        raise RuntimeError(f"a frozen module is trainable: {payload['frozen']}")
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
