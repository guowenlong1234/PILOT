#!/usr/bin/env python3
"""Direct native rendering and paired frozen-model quality on fixed captures."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def prepare(args):
    import habitat_sim
    import quaternion
    from vlnce_baselines.nwm.panorama_context import make_context_plan
    source=Path(args.source);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    split=json.loads((source/'split.json').read_text());rows=[];sim=None;scene=None
    try:
        for job in split['jobs']:
            folder=source/'captures'/job['episode'];manifest=json.loads((folder/'manifest.json').read_text())
            for item in manifest['decisions']:
                path=folder/item['file'];capture=torch.load(path,map_location='cpu',weights_only=False)
                if capture['scene']!=scene:
                    if sim is not None:sim.close()
                    scene=capture['scene'];cfg=habitat_sim.SimulatorConfiguration()
                    cfg.scene_id=scene;cfg.gpu_device_id=0;cfg.enable_physics=False
                    sensor=habitat_sim.CameraSensorSpec();sensor.uuid='rgb'
                    sensor.sensor_type=habitat_sim.SensorType.COLOR;sensor.resolution=[224,224]
                    sensor.position=[0.,1.25,0.];sensor.hfov=90
                    agent_cfg=habitat_sim.agent.AgentConfiguration();agent_cfg.sensor_specifications=[sensor]
                    sim=habitat_sim.Simulator(habitat_sim.Configuration(cfg,[agent_cfg]))
                def render(position,yaw):
                    state=habitat_sim.AgentState();state.position=np.asarray(position)
                    state.rotation=quaternion.from_rotation_vector([0.,yaw,0.])
                    sim.get_agent(0).set_state(state)
                    return np.asarray(sim.get_sensor_observations()['rgb'])[...,:3].copy()
                # A native stored direction is an independent calibration check
                # against the original project's full simulator configuration.
                errors=[]
                for i in range(4):
                    ref=render(capture['positions'][i],0.)
                    errors.append(int(np.abs(ref.astype(int)-capture['native_world_rgb12'][i,0].astype(int)).max()))
                if max(errors)>1:raise RuntimeError(f'Native camera calibration mismatch {path}: {errors}')
                keys={};images=[]
                for target in capture['targets']:
                    plan=make_context_plan(capture['positions'],capture['yaws'],target['position'],target['yaw'],'world_exact_select')
                    for i in range(4):
                        key=(str(i),round(float(plan.view_yaws[i]%(2*np.pi)),8))
                        if key not in keys:
                            keys[key]=len(images);images.append(render(capture['positions'][i],plan.view_yaws[i]))
                dest=root/'prepared'/job['episode']/item['file'];dest.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(source=str(path),job=job,keys=keys,rgb=np.stack(images),calibration_errors=errors),dest)
                rows.append(dict(file=str(dest),scene=job['scene'],split=job['split'],queries=len(capture['targets']),views=len(images),calibration_max=max(errors)))
                save(root/'prepared.json',rows);print(json.dumps(rows[-1]),flush=True)
    finally:
        if sim is not None:sim.close()


def score(args):
    from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder
    from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    from vlnce_baselines.nwm.panorama_runtime import ObservedPanoramaFrame,PanoramaHistory,PanoramaTarget,PanoramaPredictionRuntime
    from vlnce_baselines.nwm.raenwm_core.models import pack_cls_patch
    root=Path(args.output);files=json.loads((root/'prepared.json').read_text())
    encoder=RaeDinov2RgbEncoder('pretrained/rae_dinov2_with_registers_base',device=torch.device('cuda:0'),precision='ambient').eval()
    normalizer=RaeNwmLatentNormalizer('pretrained/raenwm_stage0/stat.pt').cuda()
    predictor=RaeNwmPredictor('configs/nwm/raenwm_mp3d_fresh_cls.yaml','pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar',
        device='cuda:0',num_steps=10,enable_decoder=False,use_external_context_latents=True)
    variants=[('cube_fp32','cube','float32',16,8),('direct_fp32','direct','float32',16,8),
              ('direct_large_fp32','direct','float32',64,64),('direct_large_bf16','direct','bf16',64,64),
              ('direct_large_fp16','direct','fp16',64,64)]
    rows=[]
    for entry in files:
        data=torch.load(entry['file'],map_location='cpu',weights_only=False)
        capture=torch.load(data['source'],map_location='cpu',weights_only=False)
        histories={0:PanoramaHistory()}
        for i in range(4):
            histories[0].append(ObservedPanoramaFrame(str(i),'segment',capture['positions'][i],capture['yaws'][i],
                capture['cube_rgb'][i],capture['native_world_rgb12'][i],render_token=str(i)))
        targets=[PanoramaTarget(0,t['query'],tuple(t['position']),t['yaw']) for t in capture['targets']]
        def render(requests):
            return np.stack([data['rgb'][data['keys'][(r['frame_id'],round(float(r['yaw']%(2*np.pi)),8))]] for r in requests])
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            cls,patch=encoder.forward_raw_cls_and_patch_latents({'rgb':capture['target_rgb']})
            truth=pack_cls_patch(normalizer.normalize_cls(cls),normalizer.normalize_patch(patch))
        seeds=[11] if entry['split']=='development' else [11,29,47]
        for name,source,precision,eb,pb in variants:
            runtime=PanoramaPredictionRuntime(encoder=encoder,normalizer=normalizer,predictor=predictor,
                mode='world_exact_select',observation_source=source,visual_precision=precision,
                encode_batch_size=eb,prediction_batch_size=pb,render_observed=render)
            for seed in seeds:
                noise=torch.randn(len(targets),257,768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
                torch.cuda.synchronize();start=time.perf_counter()
                pred=runtime.predict(targets,histories,initial_noise=noise);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                assert torch.isfinite(pred.pred_tokens).all()
                gt_cls=normalizer.denormalize_cls(truth[:,0]);pcls=pred.pred_cls
                cls_score=F.cosine_similarity(pcls,gt_cls,dim=-1)
                patch_score=F.cosine_similarity(pred.pred_tokens[:,1:],truth[:,1:],dim=-1).mean(-1)
                rmse=(pcls-gt_cls).square().mean(-1).sqrt()
                for i,target in enumerate(targets):
                    rows.append(dict(scene=entry['scene'],split=entry['split'],capture=entry['file'],query=target.query_id,seed=seed,variant=name,
                        cls_cosine=cls_score[i].item(),patch_cosine=patch_score[i].item(),cls_rmse=rmse[i].item(),seconds=elapsed/len(targets)))
            # Release precision-specific caches before scoring the next mode.
            for h in histories.values():
                for frame in h.frames:frame._cache.clear()
        save(root/'quality_rows.json',rows);print('scored '+entry['file'],flush=True)
    result={}
    for part in ['development','confirmation']:
        result[part]={}
        for name,*_ in variants:
            selected=[r for r in rows if r['variant']==name and r['split']==part]
            scenes=sorted({r['scene'] for r in selected})
            result[part][name]={metric:float(np.mean([np.mean([r[metric] for r in selected if r['scene']==scene]) for scene in scenes]))
                for metric in ['cls_cosine','patch_cosine','cls_rmse']}
            result[part][name].update(scenes=len(scenes),query_seed_pairs=len(selected))
    save(root/'quality_summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','score']);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a) if a.action=='prepare' else score(a)
