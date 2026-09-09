#!/usr/bin/env python3
"""Capture bounded real navigation queries, not navigation performance.

Stops after a fixed number of ready decision points or before STOP. Saves
observed-history cube faces and separate same-pose oracle targets for scoring.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import runpy
import sys

import numpy as np
import torch


class CaptureComplete(Exception):
    pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episode',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--decisions',type=int,default=4)
    args=ap.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise RuntimeError('fresh capture directory required')
    from rgb_only_optimization import common
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer
    from vlnce_baselines.nwm.panorama_context import cube_quaternions
    from habitat.core.vector_env import VectorEnv
    import habitat,habitat_sim,transformers
    report={'episode':args.episode,'decisions':[],'status':'running','versions':{
        'python':platform.python_version(),'python_path':sys.executable,'torch':str(torch.__version__),
        'cuda':torch.version.cuda,'transformers':transformers.__version__,
        'habitat':habitat.__version__,'habitat_sim':habitat_sim.__version__},
        'selection':'first ready decisions, all queries; no appearance-based filtering'}
    trainer_ref=[None];calls=[0]
    old_predict=RLTrainer._run_raenwm_rgb_fusion_prediction
    old_step=VectorEnv.step
    def hook(self,*ca,**kw):
        trainer_ref[0]=self;calls[0]+=1
        pred=old_predict(self,*ca,**kw)
        if pred is None or pred.pred_tokens is None:return pred
        runtime=self.raenwm_runtime;batch=runtime.last_batch
        frames=runtime.adapter.buffers[0].get_context()
        pos=np.stack([f.position for f in frames]);yaws=np.array([f.yaw for f in frames])
        lookup={q.query_id:q for q in self._build_raenwm_preview_queries(ca[1],ca[2],ca[3])}
        requests=[]
        for f in frames:
            for quat in cube_quaternions():requests.append({'position':f.position.tolist(),'rotation':quat})
        # Eight additional horizontal views plus four cardinal cube faces give
        # an exact native 12-direction bank, avoiding reproject/encoder drift.
        extra_sectors=[i for i in range(12) if i%3]
        for f in frames:
            for sector in extra_sectors:
                yaw=sector*math.pi/6
                requests.append({'position':f.position.tolist(),'rotation':[0,math.sin(yaw/2),0,math.cos(yaw/2)]})
        # Direct front and world30 views are projection/encoder validation
        # references at already visited poses, not alternate model inputs.
        targets=[]
        for rec in batch.records:
            q=lookup[rec.query_id];ty=float(frames[-1].yaw+rec.condition.dtheta)
            targets.append({'query':rec.query_id,'position':q.target_position.tolist(),'yaw':ty,
                            'original_turn':rec.condition.dtheta})
        for f in frames:
            requests.append({'position':f.position.tolist(),'rotation':[0,math.sin(f.yaw/2),0,math.cos(f.yaw/2)]})
        beta=round(targets[0]['yaw']/(math.pi/6))*(math.pi/6)
        requests.append({'position':frames[-1].position.tolist(),'rotation':[0,math.sin(beta/2),0,math.cos(beta/2)]})
        for t in targets:requests.append({'position':t['position'],'rotation':[0,math.sin(t['yaw']/2),0,math.cos(t['yaw']/2)]})
        rendered=self.envs.call_at(0,'get_nwm_quality_views',{'requests':requests})
        rgb=np.stack([x['rgb'] for x in rendered])
        for t,obs in zip(targets,rendered[61:]):t['navigable']=obs['navigable']
        cube=rgb[:24].reshape(4,6,*rgb.shape[1:])
        native=np.empty((4,12,*rgb.shape[1:]),dtype=np.uint8)
        for fi in range(4):
            for sector in range(12):
                native[fi,sector]=cube[fi,sector//3] if sector%3==0 else rgb[24+fi*8+extra_sectors.index(sector)]
        did=len(report['decisions'])
        episode=self.envs.current_episodes()[0]
        payload={'episode':args.episode,'scene':str(episode.scene_id),'call':calls[0],
            'cube_format':'world_optical_center_locked_v2',
            'max_camera_center_error_m':max(x['camera_center_error_m'] for x in rendered),
            'positions':pos,'yaws':yaws,'cube_rgb':cube,'native_world_rgb12':native,
            'front_rgb':rgb[56:60],'projection_reference_rgb':rgb[60],
            'projection_reference_yaw':beta,'target_rgb':rgb[61:],'targets':targets,
            'original_context_tokens':batch.context_latent[0].detach().float().cpu(),
            'original_prediction_tokens':pred.pred_tokens.detach().float().cpu()}
        torch.save(payload,root/f'decision_{did:02d}.pt')
        report['decisions'].append({'file':f'decision_{did:02d}.pt','queries':len(targets),'scene':str(episode.scene_id)})
        report['config']=str(self.config)
        (root/'manifest.json').write_text(json.dumps(report,indent=2))
        print(f'CAPTURE episode={args.episode} decision={did} queries={len(targets)}',flush=True)
        if len(report['decisions'])>=args.decisions:raise CaptureComplete('fixed decision budget reached')
        return pred
    def step(envs,actions):
        if any(x['action']['act']==0 for x in actions):raise CaptureComplete('policy STOP before decision budget')
        return old_step(envs,actions)
    RLTrainer._run_raenwm_rgb_fusion_prediction=hook;VectorEnv.step=step
    opts=common('0',1)
    opts.update({'IL.freeze_navigation_backbone':False,'IL.lr':2e-6,'IL.rgb_fusion_lr':1e-5,
        'EVAL.CKPT_PATH_DIR':str(Path(args.checkpoint).resolve()),'EVAL.EPISODE_ID':str([args.episode]),
        'EVAL.EPISODE_COUNT':1,'EVAL.SAVE_RESULTS':False,'EVAL.USE_CKPT_CONFIG':False,'EVAL.fast_eval':False,
        'CHECKPOINT_FOLDER':str(root/'unused_checkpoints')+'/',
        'TENSORBOARD_DIR':str(root/'unused_tensorboard')+'/', 'RESULTS_DIR':str(root/'unused_results')+'/'})
    sys.argv=['run.py','--exp_name','nwm_quality_'+args.episode,'--run-type','eval','--exp-config','run_r2r/iter_train_rae_dino_ghost_concat.yaml']
    for k,v in opts.items():sys.argv.extend([k,str(v)])
    report['command']=list(sys.argv)
    try:
        runpy.run_path('run.py',run_name='__main__')
        report['stop_reason']='rollout exhausted'
    except CaptureComplete as exc:
        report['stop_reason']=str(exc)
    finally:
        RLTrainer._run_raenwm_rgb_fusion_prediction=old_predict;VectorEnv.step=old_step
        if trainer_ref[0] is not None and trainer_ref[0].envs is not None:trainer_ref[0].envs.close()
        report['status']='completed' if report.get('stop_reason') else 'failed'
        (root/'manifest.json').write_text(json.dumps(report,indent=2))
    if not report['decisions']:raise RuntimeError('episode produced no ready queries')


if __name__=='__main__':main()
