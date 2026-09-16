"""Frozen stage1 rollout observer. It never supplies or changes an action.

Goal-dependent labels are obtained only AFTER goal-independent future tokens.
Q1 uses explicit noise from its own stream, preserving stage1 randomness.
"""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from .base_freeze import capture_base_tensor_manifest, compare_base_tensor_manifests
from .candidate_q0 import _representative_candidate
from .dino_cwp_future import load_dino_cwp_predictor, decode_dino_cwp_top1, waypoint_to_world_position
from .topk_query import executable_ghost_indices, stable_topk_ghost_indices
from .stage2_data import EpisodeWriter, atomic_json
from ..etp_adapter import RaeSourceContextSnapshot, RaeLatentTargetRequest


class Stage2Collector:
    def __init__(self, trainer):
        self.trainer=trainer; cfg=trainer.config.MODEL.STAGE2_COLLECT
        if trainer._active_lookahead_enabled() or not trainer._ghost_concat_enabled():
            raise ValueError('stage2 collection requires ghost concat and old E24 disabled')
        if trainer._ghost_concat_memory_mode()!='persistent_node_state':
            raise ValueError('stage2 requires persistent node state')
        self.cfg=cfg;self.counts=Counter();self.runtime=None;self.runtime_before=None
        provenance=json.loads(Path(cfg.provenance).read_text())
        self.writer=EpisodeWriter(cfg.output,provenance)
        # Construction can consume global random state; preserve it explicitly.
        rng=torch.get_rng_state(); cuda=torch.cuda.get_rng_state_all()
        try:
            self.cwp,meta=load_dino_cwp_predictor(
                trainer.config.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_checkpoint_path,
                expected_sha256=trainer.config.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_checkpoint_sha256,
                device=trainer.device)
        finally:
            torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda)
        self.modules=dict(policy=trainer.policy,fusion=trainer.raenwm_rgb_fusion_adapter,
                          waypoint=trainer.waypoint_predictor,cwp=self.cwp)
        for module in self.modules.values():
            module.eval()
            for p in module.parameters():p.requires_grad_(False)
        # The legacy eval initializer creates an unused optimizer. Remove it;
        # collection owns no optimizer or scheduler and never calls backward.
        trainer.optimizer=None;trainer.scheduler=None
        self.before=capture_base_tensor_manifest(self.modules)
        self.trace=[]

    def bind_runtime(self,runtime):
        if runtime.panorama_mode!='world_exact_select': raise ValueError('stage2 requires world_exact_select')
        runtime.panorama_predictor.capture_stage2_snapshots=True
        self.runtime=runtime

    def cache_q0(self,prediction,previews,wp,step):
        rows={} if prediction is None else {
            (s['env_index'],str(s['query_id'])):(s,prediction.pred_latent[j].detach().cpu().half().contiguous())
            for j,s in enumerate(prediction.meta.get('stage2_snapshots',[]))}
        if prediction is not None and rows and len(rows)!=len(prediction.meta['records']):
            raise ValueError('snapshot/patch row alignment differs')
        for i,graph in enumerate(self.trainer.gmaps):
            cache=getattr(graph,'stage2_q0',{})
            for dead in set(cache)-set(graph.ghost_mean_pos):del cache[dead]
            observed={str(p.target_vp) for p in previews[i] if p.target_kind in ('new_ghost','existing_ghost')}
            for ghost in observed:
                cache.pop(ghost,None)
                if (i,ghost) not in rows:continue
                snap,patch=rows[(i,ghost)]
                view,forward,target=_representative_candidate(previews[i],wp['cand_img_idxes'][i],wp['cand_distances'][i],ghost)
                if not np.allclose(snap['target_position'],target,atol=1e-5,rtol=0):raise ValueError('q0 target mismatch')
                cache[ghost]=dict(snapshot=snap,patch=patch,view=view,forward=forward,step=step)
            graph.stage2_q0=cache

    @torch.no_grad()
    def collect_step(self,nav_inputs,nav_outs,text,text_mask,no_vp_left,step):
        tr=self.trainer; rt=self.runtime; episodes=tr.envs.current_episodes()
        logits=nav_outs['global_logits'].detach(); payloads=[]; requests=[]; destinations=[]; noises=[]
        prepared=[]
        for i,(ids,graph,ep) in enumerate(zip(nav_inputs['gmap_vp_ids'],tr.gmaps,episodes)):
            ghosts=list(executable_ghost_indices(ids)); ranked=list(stable_topk_ghost_indices(ids,logits[i].cpu(),k=5))
            k=len(ranked); local={idx:j for j,idx in enumerate(ghosts)}
            row=dict(step=int(step),ghost_ids=[ids[j] for j in ghosts],global_indices=ghosts,
                topk_base_indices=torch.tensor([local[j] for j in ranked],dtype=torch.long),
                base_logits=logits[i,ghosts].cpu().float(),base_stop=int(logits[i].argmax())==0,
                no_vp_left=bool(no_vp_left[i]),owner_embeddings=nav_outs['gmap_embeds'][i,ranked].detach().cpu().half(),
                future_tokens=torch.zeros(k,257,768,dtype=torch.float16),future_valid_mask=torch.zeros(k,dtype=torch.bool),
                candidate_q0_geometry=torch.zeros(k,3),q1_conditions=torch.zeros(k,4),
                q0_metadata=[None]*k,q1_metadata=[None]*k,invalid_reason=['no_q0']*k,
                base_action=int(logits[i].argmax()))
            payloads.append(row)
            for slot,j in enumerate(ranked):
                self.counts['topk_slots']+=1
                record=graph.stage2_q0.get(ids[j])
                if record is None:continue
                s=record['snapshot']
                if not np.allclose(s['target_position'],graph.ghost_mean_pos[ids[j]],atol=1e-5,rtol=0):
                    raise ValueError('stale q0 target passed cache invalidation')
                angle=2*math.pi*(record['view']%12)/12;f=record['forward']
                row['candidate_q0_geometry'][slot]=torch.tensor([f/(1+f),math.sin(angle),math.cos(angle)])
                row['q0_metadata'][slot]={key:value.tolist() if isinstance(value,np.ndarray) else value
                    for key,value in s.items() if key!='context_latents'}
                row['q0_metadata'][slot]['source_step']=record['step']
                self.counts['q0_cache_hits']+=1
                if row['base_stop']:
                    row['invalid_reason'][slot]='base_stop';continue
                prepared.append((i,slot,ids[j],record,str(ep.episode_id)))
        if prepared:
            # CWP consumes normalized patch features, matching its existing predictor path.
            patches=torch.stack([p[3]['patch'] for p in prepared]).flatten(2).transpose(1,2).to(tr.device).float()
            with torch.autocast(device_type='cuda',enabled=False):
                decoded=decode_dino_cwp_top1(self.cwp(patches),none_threshold=0.3)
            for (i,slot,ghost,record,episode),prediction in zip(prepared,decoded):
                row=payloads[i]; s=record['snapshot']
                if not prediction.valid or prediction.pred_none:
                    row['invalid_reason'][slot]='cwp_none' if prediction.pred_none else 'cwp_invalid';continue
                # Habitat yaw faces -z; waypoint helper's heading faces +z.
                heading=math.degrees(float(s['target_yaw'])-math.pi)
                q1=waypoint_to_world_position(s['target_position'],heading_deg=heading,
                    local_angle_deg=prediction.local_angle_deg,distance_m=prediction.distance_m)
                yaw=(float(s['target_yaw'])+math.radians(prediction.local_angle_deg)+math.pi)%(2*math.pi)-math.pi
                horizon=float(s['horizon'])+prediction.distance_m/rt.adapter.config.metric_waypoint_spacing
                if not rt.adapter.config.min_horizon<=horizon<=rt.adapter.config.max_horizon:
                    row['invalid_reason'][slot]='horizon_out_of_range';continue
                snapshot=RaeSourceContextSnapshot(str(ghost),record['step'],s['context_latents'],
                    np.asarray(s['source_position']),float(s['source_yaw']),context_source='panorama_q0',
                    context_contract='stage2_panorama_q0_snapshot_v1')
                requests.append(RaeLatentTargetRequest(i,ghost,snapshot,q1,yaw,horizon))
                destinations.append((i,slot))
                identity=f'{self.cfg.seed}|{episode}|{step}|{ghost}|{record["step"]}'
                seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'little')%(2**63-1)
                generator=torch.Generator(device=tr.device).manual_seed(seed)
                noises.append(torch.randn(257,768,device=tr.device,generator=generator))
                row['q1_metadata'][slot]=dict(position=q1.tolist(),yaw=yaw,horizon=horizon,noise_seed=seed)
        # Explicit noise prevents consumption of runtime.generator (stage1).
        rng_before=rt.generator.get_state().clone()
        for start in range(0,len(requests),64):
            subset=requests[start:start+64]
            batch=rt.adapter.build_raenwm_latent_batch(subset,device=tr.device)
            prediction=rt._predict_batch(batch,initial_noise=torch.stack(noises[start:start+64]))
            encoded=torch.cat((prediction.pred_cls[:,None],prediction.pred_tokens[:,1:]),dim=1).cpu().half()
            if not torch.isfinite(encoded).all():raise FloatingPointError('invalid q1 features')
            for j,(i,slot) in enumerate(destinations[start:start+64]):
                payloads[i]['future_tokens'][slot]=encoded[j]
                payloads[i]['future_valid_mask'][slot]=True
                payloads[i]['q1_conditions'][slot]=batch.condition_tensor[j].detach().cpu()
                payloads[i]['invalid_reason'][slot]='valid'
        if not torch.equal(rng_before,rt.generator.get_state()):raise RuntimeError('q1 consumed stage1 RNG')
        self.counts['q1_requested']+=len(requests)
        # Teacher sampling must not consume the navigation's Python RNG.
        state=random.getstate()
        try:
            random.seed(int(self.cfg.seed)+int(step))
            teacher=tr._teacher_action_new(nav_inputs['gmap_vp_ids'],no_vp_left,False).cpu().tolist()
        finally:random.setstate(state)
        for i,(row,ep,action) in enumerate(zip(payloads,episodes,teacher)):
            ghost_globals=row['global_indices']; local=ghost_globals.index(action) if action in ghost_globals else -1
            selected=row['topk_base_indices'].tolist()
            row.update(teacher_action=action,teacher_valid=action>=0,teacher_stop=action==0,
                teacher_base_index=local,teacher_rank_in_topk=selected.index(local) if local in selected else -1)
            self.counts.update(row['invalid_reason']);self.counts['rows']+=1
            self.writer.append(str(ep.episode_id),str(ep.scene_id),text[i][text_mask[i].bool()],row)
            if self.cfg.trace:
                self.trace.append(dict(episode=str(ep.episode_id),step=step,action=row['base_action'],
                    logits=logits[i,:len(nav_inputs['gmap_vp_ids'][i])].cpu().tolist()))

    def complete_episode(self,episode,metrics):
        r=self.writer.complete(episode,metrics); self.counts['episodes']+=1
        atomic_json(Path(self.cfg.output)/'progress.json',dict(self.counts))
        print('STAGE2_EPISODE',json.dumps(r),flush=True)

    def finish(self):
        if self.writer.pending:raise RuntimeError('unfinished episodes remain')
        after=capture_base_tensor_manifest(self.modules)
        comparison=compare_base_tensor_manifests(self.before,after)
        atomic_json(Path(self.cfg.output)/'freeze_report.json',dict(comparison=comparison,counts=dict(self.counts),
            before=self.before,after=after,optimizer_created=False,training_started=False))
        if self.cfg.trace:atomic_json(Path(self.cfg.output)/'trace.json',self.trace)
        if not comparison['exact_match']:raise RuntimeError('frozen stage1 changed')
