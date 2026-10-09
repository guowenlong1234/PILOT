"""Extend cached panorama q1 predictions against their unchanged real history.

The original q1 computation and random seeds remain unchanged. Only successful
q1 rows enter this helper; later invalid predictions retain the deepest endpoint.
"""
import hashlib
import math
import numpy as np
import torch
from .dino_cwp_future import decode_dino_cwp_top1, waypoint_to_world_position
from ..etp_adapter import RaeLatentTargetRequest


def rollout_depth(config):
    value = getattr(config, 'lookahead_horizon_steps', 1)
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError('stage2 lookahead_horizon_steps must be 1, 2 or 3')
    return int(value)


def initialize_rollout_rows(rows, depth):
    for row in rows:
        mask = row['future_valid_mask']
        row.update(rollout_requested_depth=depth,
                   realized_depths=mask.long(),
                   full_horizon_mask=mask.clone() if depth == 1 else torch.zeros_like(mask),
                   fallback_mask=mask.clone() if depth > 1 else torch.zeros_like(mask),
                   endpoint_conditions=row['q1_conditions'].clone(),
                   endpoint_metadata=[dict(m) if m is not None and bool(mask[j]) else None
                                      for j, m in enumerate(row['q1_metadata'])],
                   rollout_metadata=[[dict(m, depth=1)] if m is not None and bool(mask[j]) else []
                                     for j, m in enumerate(row['q1_metadata'])],
                   rollout_failure_reason=['' if bool(v) else row['invalid_reason'][j]
                                           for j, v in enumerate(mask)])


def recursive_noise_seed(identity, depth):
    # identity is exactly the legacy q1 seed preimage. Additional layers have
    # separate seeds without depending on batching, top-k order, or global RNG.
    value = identity if depth == 1 else f'{identity}|depth={depth}'
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], 'little') % (2**63-1)


@torch.no_grad()
def extend_rollout(collector, rows, states, depth):
    """states carry (env, slot, original snapshot, latent, pose, horizon, seed)."""
    if depth == 1:
        return
    rt = collector.runtime
    device = collector.trainer.device
    rng_before = rt.generator.get_state().clone()
    active = list(states)
    for level in range(2, depth + 1):
        if not active:
            break
        collector.counts[f'rollout_depth_{level}_cwp_requested'] += len(active)
        patches = torch.stack([s['latent'] for s in active]).flatten(2).transpose(1, 2).to(device).float()
        # Unexpected model/runtime exceptions propagate: silently converting a
        # broken deployment or OOM into apparently valid data is unsafe.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            decoded = decode_dino_cwp_top1(collector.cwp(patches), none_threshold=0.3)
        if len(decoded) != len(active):
            raise ValueError('recursive CWP batch row count differs')
        requests, pending, noises = [], [], []
        for state, prediction in zip(active, decoded):
            row, slot = rows[state['env']], state['slot']
            reason = 'cwp_invalid' if not prediction.valid else ('cwp_none' if prediction.pred_none else '')
            horizon = state['horizon'] + prediction.distance_m / rt.adapter.config.metric_waypoint_spacing
            if not reason and not rt.adapter.config.min_horizon <= horizon <= rt.adapter.config.max_horizon:
                reason = 'horizon_out_of_range'
            if reason:
                row['rollout_failure_reason'][slot] = reason
                row['rollout_metadata'][slot].append(dict(depth=level, failure_reason=reason))
                collector.counts[f'rollout_depth_{level}_{reason}'] += 1
                continue
            position = waypoint_to_world_position(state['position'],
                heading_deg=math.degrees(state['yaw'] - math.pi),
                local_angle_deg=prediction.local_angle_deg, distance_m=prediction.distance_m)
            yaw = (state['yaw'] + math.radians(prediction.local_angle_deg) + math.pi) % (2*math.pi) - math.pi
            request = RaeLatentTargetRequest(state['env'], state['ghost'], state['snapshot'], position, yaw, horizon)
            seed = recursive_noise_seed(state['identity'], level)
            generator = torch.Generator(device=device).manual_seed(seed)
            noises.append(torch.randn(257, 768, device=device, generator=generator))
            metadata = dict(depth=level, position=position.tolist(), yaw=yaw, horizon=horizon, noise_seed=seed)
            pending.append((state, metadata))
            requests.append(request)
        collector.counts[f'rollout_depth_{level}_nwm_requested'] += len(requests)
        collector.counts['deep_queries_requested'] += len(requests)
        next_active = []
        for start in range(0, len(requests), 64):
            batch = rt.adapter.build_raenwm_latent_batch(requests[start:start+64], device=device)
            prediction = rt._predict_batch(batch, initial_noise=torch.stack(noises[start:start+64]))
            tokens = torch.cat((prediction.pred_cls[:, None], prediction.pred_tokens[:, 1:]), dim=1).cpu().half()
            latents = prediction.pred_latent.detach()
            expected = len(requests[start:start+64])
            if tokens.shape != (expected, 257, 768) or latents.shape != (expected, 768, 16, 16):
                raise ValueError('recursive NWM output shape differs')
            for j, (state, metadata) in enumerate(pending[start:start+64]):
                row, slot = rows[state['env']], state['slot']
                if not torch.isfinite(tokens[j]).all() or not torch.isfinite(latents[j]).all():
                    reason = 'nwm_nonfinite'
                    row['rollout_failure_reason'][slot] = reason
                    row['rollout_metadata'][slot].append(dict(metadata, failure_reason=reason))
                    collector.counts[f'rollout_depth_{level}_{reason}'] += 1
                    continue
                row['future_tokens'][slot] = tokens[j]
                row['endpoint_conditions'][slot] = batch.condition_tensor[j].detach().cpu()
                row['endpoint_metadata'][slot] = dict(metadata)
                row['rollout_metadata'][slot].append(metadata)
                row['realized_depths'][slot] = level
                row['full_horizon_mask'][slot] = level == depth
                row['fallback_mask'][slot] = level < depth
                collector.counts[f'rollout_depth_{level}_success'] += 1
                next_active.append(dict(state, latent=latents[j], position=np.asarray(metadata['position']),
                                        yaw=metadata['yaw'], horizon=metadata['horizon']))
        active = next_active
    if not torch.equal(rng_before, rt.generator.get_state()):
        raise RuntimeError('recursive future consumed stage1 RNG')
