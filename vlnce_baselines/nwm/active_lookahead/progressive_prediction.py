"""One physical future depth per call, with immutable observed q0 context.

No request is issued by initialization. The caller controls active decision rows
between depths; stopping a row never queries or inspects its deeper evidence.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
import time
import numpy as np
import torch
from .dino_cwp_future import decode_dino_cwp_top1, waypoint_to_world_position
from .progressive_core import successor_geometry
from .topk_query import executable_ghost_indices, stable_topk_ghost_indices
from ..etp_adapter import RaeSourceContextSnapshot, RaeLatentTargetRequest

# Existing frozen RAE/DINO contract (CLS followed by normalized patch tokens).
from .progressive_contract import FUTURE_TOKEN_COUNT, FUTURE_FEATURE_DIM
PATCH_GRID = (16, 16)


def prediction_identity(seed, scene, episode, step, candidate_id, source_step, depth):
    identity = json.dumps([int(seed), str(scene), str(episode), int(step),
                           str(candidate_id), int(source_step), int(depth)], separators=(',', ':'))
    noise_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'little') % (2**63-1)
    return identity, noise_seed


@dataclass
class PredictionDecision:
    rows: list
    branches: list
    depth: int = 0
    layers: list = field(default_factory=list)
    closed: bool = False


class ProgressivePrediction:
    def __init__(self, collector, max_future_depth=2, distance_scale=1.):
        if isinstance(max_future_depth, bool) or max_future_depth not in (1, 2, 3):
            raise ValueError('max_future_depth must be 1, 2 or 3')
        if not math.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError('distance_scale must be positive')
        self.collector = collector
        self.max_future_depth = int(max_future_depth)
        self.distance_scale = float(distance_scale)

    @torch.no_grad()
    def initialize_decision(self, nav_inputs, nav_outs, text, text_mask, no_vp_left, step):
        c = self.collector
        tr = c.trainer
        logits = nav_outs['global_logits'].detach()
        rows, branches = [], []
        for env, (ids, graph, episode) in enumerate(zip(nav_inputs['gmap_vp_ids'], tr.gmaps, tr.envs.current_episodes())):
            ghosts = list(executable_ghost_indices(ids))
            ranked = list(stable_topk_ghost_indices(ids, logits[env].cpu(), k=5))
            local = {idx: j for j, idx in enumerate(ghosts)}
            k = len(ranked)
            model_stop = int(logits[env].argmax()) == 0
            forced_stop = step == tr.max_len-1 or bool(no_vp_left[env])
            row = dict(text_tokens=text[env][text_mask[env].bool()].detach().cpu().half(), step=int(step), ghost_ids=[ids[j] for j in ghosts], global_indices=ghosts,
                       topk_base_indices=torch.tensor([local[j] for j in ranked], dtype=torch.long),
                       base_logits=logits[env, ghosts].cpu().float(),
                       owner_embeddings=nav_outs['gmap_embeds'][env, ranked].detach().cpu().half(),
                       candidate_present_mask=torch.ones(k, dtype=torch.bool),
                       candidate_q0_geometry=torch.zeros(k, 3), q0_metadata=[None]*k,
                       model_base_stop=model_stop, forced_stop=forced_stop,
                       base_stop=model_stop or forced_stop, no_vp_left=bool(no_vp_left[env]),
                       base_action=int(logits[env].argmax()))
            rows.append(row)
            for slot, idx in enumerate(ranked):
                ghost = ids[idx]
                record = getattr(graph, 'stage2_q0', {}).get(ghost)
                state = dict(env=env, slot=slot, ghost=ghost, terminal=False, reason='', path=0.)
                branches.append(state)
                if record is None:
                    row['q0_metadata'][slot] = dict(source_unavailable=True, failure_reason='no_q0')
                    state.update(terminal=True, reason='base_stop' if row['base_stop'] else 'no_q0')
                    continue
                s = record['snapshot']
                if not np.allclose(s['target_position'], graph.ghost_mean_pos[ghost], atol=1e-5, rtol=0):
                    raise ValueError('stale q0 target passed cache invalidation')
                angle = 2*math.pi*(record['view'] % 12)/12
                f = record['forward']
                row['candidate_q0_geometry'][slot] = torch.tensor([f/(1+f), math.sin(angle), math.cos(angle)])
                row['q0_metadata'][slot] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                          for key, value in s.items() if key != 'context_latents'}
                row['q0_metadata'][slot]['source_step'] = record['step']
                snapshot = RaeSourceContextSnapshot(str(ghost), record['step'], s['context_latents'].detach().clone(),
                    np.asarray(s['source_position']).copy(), float(s['source_yaw']),
                    context_source='panorama_q0', context_contract='stage2_panorama_q0_snapshot_v1')
                state.update(snapshot=snapshot, latent=record['patch'].detach().clone(),
                    position=np.asarray(s['target_position']).copy(), yaw=float(s['target_yaw']),
                    q0_position=np.asarray(s['target_position']).copy(), q0_yaw=float(s['target_yaw']),
                    horizon=float(s['horizon']), source_step=int(record['step']),
                    scene=str(episode.scene_id), episode=str(episode.episode_id), step=int(step))
                if row['base_stop']:
                    state.update(terminal=True, reason='base_stop')
                if (state['latent'].shape != (FUTURE_FEATURE_DIM, *PATCH_GRID)
                        or not torch.isfinite(state['latent']).all()
                        or not torch.isfinite(snapshot.context_latents).all()
                        or not np.isfinite(snapshot.source_position).all()
                        or not math.isfinite(snapshot.source_yaw)
                        or not np.isfinite(state['position']).all()
                        or not math.isfinite(state['yaw']) or not math.isfinite(state['horizon'])):
                    state.update(terminal=True, reason='invalid_q0')
        return PredictionDecision(rows, branches)

    @torch.no_grad()
    def predict_next_depth(self, decision, active_rows=None):
        if decision.closed or decision.depth >= self.max_future_depth:
            raise ValueError('decision is finalized or maximum depth reached')
        c, rt = self.collector, self.collector.runtime
        device = torch.device(c.trainer.device)
        active = set(range(len(decision.rows))) if active_rows is None else set(active_rows)
        if any(i < 0 or i >= len(decision.rows) for i in active):
            raise ValueError('active row outside decision')
        level = decision.depth + 1
        outputs = []
        for row in decision.rows:
            k = len(row['topk_base_indices'])
            outputs.append(dict(depth=level,
                future_tokens=torch.zeros(k, FUTURE_TOKEN_COUNT, FUTURE_FEATURE_DIM, dtype=torch.float16),
                future_geometry=torch.zeros(k, 5), future_valid_mask=torch.zeros(k, dtype=torch.bool),
                future_token_mask=torch.zeros(k, FUTURE_TOKEN_COUNT, dtype=torch.bool),
                future_queried_mask=torch.zeros(k, dtype=torch.bool),
                terminal_mask=torch.zeros(k, dtype=torch.bool), unqueried_mask=torch.ones(k, dtype=torch.bool),
                future_metadata=[None]*k, terminal_reason=['']*k, world_model_queries=0,
                prediction_seconds=0.))
        states = [s for s in decision.branches if s['env'] in active and not s['terminal']]
        rng_before = rt.generator.get_state().clone()
        # fork_rng protects navigation global RNG even if a frozen backend consumes it.
        cuda_devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
        if device.type == 'cuda': torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                self._predict(states, outputs, level, device)
        finally:
            if not torch.equal(rng_before, rt.generator.get_state()):
                rt.generator.set_state(rng_before)
                raise RuntimeError('progressive prediction consumed stage1 RNG')
            for name, tensors in getattr(c, 'versions', {}).items():
                if any(t._version != version for t, version in tensors):
                    raise RuntimeError(f'frozen tensor mutated: {name}')
        if device.type == 'cuda': torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-started
        for s in decision.branches:
            out, slot = outputs[s['env']], s['slot']
            out['terminal_mask'][slot] = s['terminal']
            out['terminal_reason'][slot] = s['reason']
            if s['terminal']: out['unqueried_mask'][slot] = False
        for env in active: outputs[env]['prediction_seconds'] = elapsed
        decision.depth = level
        decision.layers.append(outputs)
        return outputs

    def _predict(self, states, outputs, level, device):
        if not states: return
        c, rt = self.collector, self.collector.runtime
        patches = torch.stack([s['latent'] for s in states]).flatten(2).transpose(1, 2).to(device).float()
        # Fixed single-candidate batches keep kernel shapes independent of other
        # rows stopping/reordering; explicit noise alone cannot ensure that.
        with torch.autocast(device_type=device.type, enabled=False):
            decoded = [decode_dino_cwp_top1(c.cwp(patch[None]), none_threshold=0.3)[0]
                       for patch in patches]
        if len(decoded) != len(states): raise ValueError('CWP batch row count differs')
        pending, requests, noises = [], [], []
        cfg = rt.adapter.config
        if not math.isfinite(cfg.metric_waypoint_spacing) or cfg.metric_waypoint_spacing <= 0:
            raise ValueError('invalid metric_waypoint_spacing')
        for state, waypoint in zip(states, decoded):
            out, slot = outputs[state['env']], state['slot']
            out['unqueried_mask'][slot] = False
            out['future_queried_mask'][slot] = True
            reason = 'cwp_invalid' if not waypoint.valid else ('cwp_none' if waypoint.pred_none else '')
            if not reason and (not math.isfinite(waypoint.distance_m) or waypoint.distance_m < 0 or not math.isfinite(waypoint.local_angle_deg)):
                reason = 'cwp_invalid'
            horizon = state['horizon'] + waypoint.distance_m/cfg.metric_waypoint_spacing
            if not reason and not cfg.min_horizon <= horizon <= cfg.max_horizon: reason = 'horizon_out_of_range'
            if reason:
                state.update(terminal=True, reason=reason)
                out['future_metadata'][slot] = dict(depth=level, failure_reason=reason)
                continue
            position = waypoint_to_world_position(state['position'], heading_deg=math.degrees(state['yaw']-math.pi),
                                                  local_angle_deg=waypoint.local_angle_deg, distance_m=waypoint.distance_m)
            yaw = (state['yaw']+math.radians(waypoint.local_angle_deg)+math.pi) % (2*math.pi)-math.pi
            identity, seed = prediction_identity(c.cfg.seed, state['scene'], state['episode'], state['step'],
                                                 state['ghost'], state['source_step'], level)
            metadata = dict(depth=level, position=position.tolist(), yaw=yaw, horizon=horizon,
                            request_identity=identity, noise_seed=seed, cumulative_path_length=state['path']+waypoint.distance_m)
            out['future_metadata'][slot] = metadata
            pending.append((state, metadata))
            requests.append(RaeLatentTargetRequest(state['env'], state['ghost'], state['snapshot'], position, yaw, horizon))
            generator = torch.Generator(device=device).manual_seed(seed)
            noises.append(torch.randn(FUTURE_TOKEN_COUNT, FUTURE_FEATURE_DIM, device=device, generator=generator))
            out['world_model_queries'] += 1
        for start in range(0, len(requests), 1):
            batch = rt.adapter.build_raenwm_latent_batch(requests[start:start+1], device=device)
            pred = rt._predict_batch(batch, initial_noise=torch.stack(noises[start:start+1]))
            tokens = torch.cat((pred.pred_cls[:, None], pred.pred_tokens[:, 1:]), dim=1).detach().cpu().half()
            latents = pred.pred_latent.detach()
            n = len(requests[start:start+1])
            if tokens.shape != (n, FUTURE_TOKEN_COUNT, FUTURE_FEATURE_DIM) or latents.shape != (n, FUTURE_FEATURE_DIM, *PATCH_GRID):
                raise ValueError('progressive NWM output violates feature contract')
            for j, (state, metadata) in enumerate(pending[start:start+1]):
                out, slot = outputs[state['env']], state['slot']
                if not torch.isfinite(tokens[j]).all() or not torch.isfinite(latents[j]).all():
                    state.update(terminal=True, reason='nwm_nonfinite')
                    metadata['failure_reason'] = state['reason']
                    continue
                out['future_tokens'][slot] = tokens[j]
                out['future_valid_mask'][slot] = True
                out['future_token_mask'][slot] = True
                out['future_geometry'][slot] = torch.tensor(successor_geometry(state['q0_position'], state['q0_yaw'],
                    metadata['position'], metadata['yaw'], metadata['cumulative_path_length'], self.distance_scale), dtype=torch.float32)
                state.update(latent=latents[j], position=np.asarray(metadata['position']), yaw=metadata['yaw'],
                             horizon=metadata['horizon'], path=metadata['cumulative_path_length'])
        c.counts[f'progressive_depth_{level}_queries'] += len(requests)

    def finalize_decision(self, decision):
        """Return only actually executed prefixes; never mark absent depth valid."""
        decision.closed = True
        rows = []
        for env, original in enumerate(decision.rows):
            row = dict(original)
            layers = [layer[env] for layer in decision.layers]
            for key in ('future_tokens', 'future_geometry', 'future_valid_mask', 'future_token_mask', 'terminal_mask', 'unqueried_mask', 'future_queried_mask'):
                if layers: row[key] = torch.stack([layer[key] for layer in layers], dim=1)
            row['future_metadata'] = [[layer['future_metadata'][slot] for layer in layers]
                                      for slot in range(len(original['topk_base_indices']))]
            row['terminal_reason'] = [[layer['terminal_reason'][slot] for layer in layers]
                                      for slot in range(len(original['topk_base_indices']))]
            row['future_terminal_mask'] = row.get('terminal_mask', torch.zeros(len(original['topk_base_indices']), 0, dtype=torch.bool))
            k, d = len(original['topk_base_indices']), len(layers)
            row['target_positions'] = torch.zeros(k, d, 3)
            row['target_yaws'] = torch.zeros(k, d)
            row['horizons'] = torch.zeros(k, d)
            row['request_ids'] = [['']*d for _ in range(k)]
            row['termination_reasons'] = row['terminal_reason']
            for slot in range(k):
                for depth, metadata in enumerate(row['future_metadata'][slot]):
                    if metadata and 'position' in metadata:
                        row['target_positions'][slot,depth] = torch.tensor(metadata['position'])
                        row['target_yaws'][slot,depth] = metadata['yaw']
                        row['horizons'][slot,depth] = metadata['horizon']
                        row['request_ids'][slot][depth] = metadata['request_identity']
            row['invalid_reason'] = [branch['reason'] or 'valid' for branch in decision.branches if branch['env'] == env]
            row['executed_depth'] = decision.depth
            rows.append(row)
        return rows
