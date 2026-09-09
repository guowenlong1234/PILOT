#!/usr/bin/env python3
"""Repeatable real joint-training benchmark; optional stage synchronization."""
import argparse
import hashlib
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--updates', type=int, default=6)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--capture', action='store_true')
    p.add_argument('--sync-stages', action='store_true')
    p.add_argument('--audit', action='store_true')
    p.add_argument('--context-mode',choices=('front','world_exact_select'),default='world_exact_select')
    args = p.parse_args()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    root = Path(args.output).absolute();root.mkdir(parents=True, exist_ok=True)
    from rgb_only_optimization import common
    import vlnce_baselines.ss_trainer_ETP_R1 as trainer_module
    from vlnce_baselines.nwm.low_level_context import LowLevelContextSynchronizer
    import vlnce_baselines.nwm.panorama_runtime as pano
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder
    from vlnce_baselines.models.R1Policy import ETP
    import habitat, habitat_sim, transformers
    stats = defaultdict(lambda: [0., 0]);steps=[];captures=[0];trainer=[None];last=[None]
    frozen_world_before={}

    def digest(named):
        return {name:hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name,value in named}

    old_init_runtime=trainer_module.RLTrainer._initialize_raenwm_runtime
    def init_runtime(self,*a,**kw):
        result=old_init_runtime(self,*a,**kw)
        if args.audit and not frozen_world_before:
            model=self.raenwm_runtime.predictor.bundle.model
            assert not any(p.requires_grad for p in model.parameters())
            frozen_world_before.update(digest(model.named_parameters()))
        return result
    trainer_module.RLTrainer._initialize_raenwm_runtime=init_runtime

    def wrap(cls, name, label):
        old = getattr(cls,name)
        def call(*a, **kw):
            if args.sync_stages:torch.cuda.synchronize(rank)
            start=time.perf_counter()
            value=old(*a,**kw)
            if args.sync_stages:torch.cuda.synchronize(rank)
            key=label if label!='policy' else 'policy_'+str(kw.get('mode','unknown'))
            stats[key][0]+=time.perf_counter()-start;stats[key][1]+=1
            return value
        setattr(cls,name,call)

    wrap(LowLevelContextSynchronizer,'drain','context_drain')
    wrap(LowLevelContextSynchronizer,'_encode_frames','front_encode')
    wrap(pano.PanoramaPredictionRuntime,'predict','panorama_total')
    if hasattr(pano.PanoramaPredictionRuntime,'_observed_rgb_batch'):
        wrap(pano.PanoramaPredictionRuntime,'_observed_rgb_batch','rgb_batch')
    wrap(RaeDinov2RgbEncoder,'forward_raw_cls_and_patch_latents','raw_dino_encode')
    wrap(RaeNwmPredictor,'predict_time_from_etp_batch','world_model')
    wrap(ETP,'forward','policy')
    wrap(torch.Tensor,'backward','backward')
    old_view=pano.observed_view
    def view(*a,**kw):
        start=time.perf_counter();result=old_view(*a,**kw)
        stats['reprojection'][0]+=time.perf_counter()-start;stats['reprojection'][1]+=1
        return result
    pano.observed_view=view
    old_predict=pano.PanoramaPredictionRuntime.predict
    def predict(self,targets,histories,**kw):
        if args.capture and rank==0 and captures[0]<3:
            valid=[t for t in targets if len(histories[t.env_index].frames)==4]
            if len(valid)>=4:
                frames={i:[dict(frame_id=f.frame_id,segment_id=f.segment_id,position=f.position,
                    body_yaw=f.body_yaw,cube_rgb=f.cube_rgb,native_world_rgb12=f.native_world_rgb12)
                    for f in histories[i].frames] for i in {t.env_index for t in valid}}
                torch.save({'histories':frames,'targets':valid},root/f'capture{captures[0]}.pt')
                captures[0]+=1
        return old_predict(self,targets,histories,**kw)
    pano.PanoramaPredictionRuntime.predict=predict
    old_interval=trainer_module.RLTrainer._train_interval
    old_step=trainer_module.step_amp_optimizer
    def step(*a,**kw):
        result=old_step(*a,**kw);torch.cuda.synchronize(rank)
        now=time.perf_counter();steps.append(now-last[0]);last[0]=now
        return result
    trainer_module.step_amp_optimizer=step
    def interval(self,*a,**kw):
        trainer[0]=self
        groups=[dict(name=g.get('name'),params=sum(p.numel() for p in g['params']),lr=g['lr']) for g in self.optimizer.param_groups]
        if args.audit:
            assert all(p.requires_grad for g in self.optimizer.param_groups for p in g['params'])
            assert not any(p.requires_grad for p in self.waypoint_predictor.parameters())
            frozen_before=digest((n,p) for n,p in self.policy.named_parameters() if not p.requires_grad)
            waypoint_before=digest(self.waypoint_predictor.named_parameters())
            cls_before=digest((n,p) for n,p in self.policy.named_parameters() if 'cls_residual_mlp' in n)
            fusion_before=digest(self.raenwm_rgb_fusion_adapter.named_parameters())
        torch.cuda.synchronize(rank);last[0]=time.perf_counter()
        result=old_interval(self,*a,**kw)
        context={k:sum(v) for k,v in self.logs.items() if k.startswith('nwm_context_')}
        report=dict(rank=rank,world=world,steps=steps,warmup=args.warmup,
            mean_step_seconds=float(np.mean(steps[args.warmup:])),stages=dict(stats),
            synchronized_stages=args.sync_stages,optimizer=groups,context=context,
            losses={k:list(v) for k,v in self.logs.items() if 'loss' in k.lower() or 'grad_norm' in k.lower()},
            peak_memory_mib=torch.cuda.max_memory_allocated(rank)/2**20,
            versions=dict(python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                transformers=transformers.__version__,habitat=habitat.__version__,habitat_sim=habitat_sim.__version__))
        if args.audit:
            audit=dict(frozen_visual_unchanged=frozen_before==digest((n,p) for n,p in self.policy.named_parameters() if not p.requires_grad),
                frozen_waypoint_unchanged=waypoint_before==digest(self.waypoint_predictor.named_parameters()),
                frozen_world_unchanged=frozen_world_before==digest(self.raenwm_runtime.predictor.bundle.model.named_parameters()),
                cls_mapping_updated=cls_before!=digest((n,p) for n,p in self.policy.named_parameters() if 'cls_residual_mlp' in n),
                fusion_updated=fusion_before!=digest(self.raenwm_rgb_fusion_adapter.named_parameters()))
            assert all(audit.values()),audit
            report['audit']=audit
        (root/f'rank{rank}.json').write_text(json.dumps(report,indent=2))
        print('BENCHMARK '+json.dumps(report),flush=True)
        return result
    trainer_module.RLTrainer._train_interval=interval
    trainer_module.RLTrainer.save_checkpoint=lambda *a,**kw:None
    opts=common(','.join(map(str,range(world))),4)
    opts.update({'IL.freeze_navigation_backbone':False,'IL.lr':2e-6,'IL.rgb_fusion_lr':1e-5,
        'IL.iters':args.updates,'IL.log_every':args.updates,'IL.batch_size':4,
        'IL.checkpoint_sync_enabled':False,'IL.is_requeue':False,
        'MODEL.RAENWM.panorama_context_mode':args.context_mode,
        'CHECKPOINT_FOLDER':str(root/'checkpoints')+'/', 'TENSORBOARD_DIR':str(root/'tb')+'/',
        'RESULTS_DIR':str(root/'results')+'/'})
    sys.argv=['run.py','--local_rank',str(rank),'--exp_name','panorama_perf',
        '--run-type','dagger','--exp-config','run_r2r/iter_train_rae_dino_ghost_concat.yaml']
    for k,v in opts.items():sys.argv.extend([k,str(v)])
    try:runpy.run_path('run.py',run_name='__main__')
    finally:
        if trainer[0] is not None:trainer[0].envs.close()


if __name__=='__main__':main()
