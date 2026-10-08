#!/usr/bin/env python3
"""Paired camera-domain diagnostic; frozen models, no navigation training."""
import argparse, collections, hashlib, json, math, pickle, random, subprocess, time
from pathlib import Path
import numpy as np

CAMERAS = {'original': (90., 1.25), 'height_only': (90., .88),
           'fov_only': (63., 1.25), 'rxr': (63., .88)}

def dump(p, obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)

def prepare(a):
    root=Path(a.source); groups=collections.defaultdict(list)
    for line in (root/'collection_manifest.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['source_split']=='val_unseen' and r['filtered_len']>=45:groups[r['scene_id']].append(r)
    rng=random.Random(20261008);scenes=sorted(groups);rng.shuffle(scenes);jobs=[]
    selected_scenes=scenes if a.scenes == 0 else scenes[:a.scenes]
    for scene in selected_scenes:
        rows=sorted(groups[scene],key=lambda r:r['traj_name']);rng.shuffle(rows)
        for r in rows[:a.trajectories]:
            with (root/'data/mp3d'/r['traj_name']/'traj_data.pkl').open('rb') as f:d=pickle.load(f)
            pos=np.asarray(d['position']); yaw=np.asarray(d['yaw'])-math.pi
            # Raw collection yaw is Habitat yaw + pi. Use true low-level positions.
            for t in [len(pos)//3,2*len(pos)//3]:
                for h in [4,8]:
                    ids=list(range(t-3,t+1)); target=t+h
                    jobs.append(dict(id=f"{r['traj_name']}_{t}_{h}",scene=scene,trajectory=r['traj_name'],
                        positions=pos[ids].tolist(),yaws=yaw[ids].tolist(),target=pos[target].tolist(),
                        target_yaw=float(yaw[target]),horizon_frames=h))
    assert jobs and len({j["id"] for j in jobs})==len(jobs)
    out=Path(a.output);assert not out.exists()
    dump(out,dict(seed=20261008,scenes=selected_scenes,jobs=jobs,trajectories_per_scene=a.trajectories,cameras=CAMERAS,seeds=[11,29,47],
         severe_flags=dict(cls_cosine_drop=.05,patch_cosine_drop=.05,cls_rmse_increase_fraction=.2,cwp_clear_drop=.10),
         scope='Matched recorded RxR paths; direct rendering; no scene-generalization claim; geometric CWP checks are not teacher accuracy.'))
    print('prepared',len(jobs),'queries',flush=True)

class Renderer:
    def __init__(self, scene):
        import habitat_sim as hs
        self.hs=hs; cfg=hs.SimulatorConfiguration();cfg.scene_id=str(scene);cfg.gpu_device_id=0;cfg.enable_physics=False
        sensors=[]
        for name,(fov,height) in CAMERAS.items():
            s=hs.CameraSensorSpec();s.uuid=name;s.sensor_type=hs.SensorType.COLOR;s.sensor_subtype=hs.SensorSubType.PINHOLE
            s.resolution=[224,224];s.hfov=fov;s.position=[0.0,float(height),0.0];sensors.append(s)
        ac=hs.agent.AgentConfiguration();ac.sensor_specifications=sensors;ac.height=.88;ac.radius=.18
        self.sim=hs.Simulator(hs.Configuration(cfg,[ac]));assert self.sim.pathfinder.is_loaded
    def render(self,pos,yaw):
        from habitat_sim.utils.common import quat_from_angle_axis
        state=self.hs.AgentState();state.position=np.asarray(pos);state.rotation=quat_from_angle_axis(yaw,np.array([0.,1.,0.]))
        self.sim.get_agent(0).set_state(state,reset_sensors=True)
        obs=self.sim.get_sensor_observations()
        for name,(_,height) in CAMERAS.items():
            actual=self.sim.get_agent(0).get_state().sensor_states[name].position
            assert np.allclose(actual,np.asarray(pos)+[0,height,0],atol=2e-5)
        return {k:np.asarray(v)[...,:3].copy() for k,v in obs.items()}
    def clear(self,pos,yaw,p):
        from vlnce_baselines.nwm.active_lookahead.dino_cwp_future import waypoint_to_world_position
        if not p.valid or p.pred_none:return None
        target=waypoint_to_world_position(pos,heading_deg=math.degrees(yaw-math.pi),local_angle_deg=p.local_angle_deg,distance_m=p.distance_m)
        pf=self.sim.pathfinder;current=np.asarray(pf.snap_point(np.asarray(pos,dtype=np.float32))).copy()
        if not np.isfinite(current).all():return False
        origin=current.copy()
        for frac in np.linspace(0,1,max(2,int(p.distance_m/.05)+1))[1:]:
            wish=origin+(target-origin)*frac
            step=np.asarray(pf.try_step_no_sliding(current,wish)).copy()
            if np.linalg.norm((step-wish)[[0,2]])>.08:return False
            current=step
        return bool(np.linalg.norm((current-target)[[0,2]])<.08)

def run(a):
    import torch, torch.nn.functional as F
    import habitat,habitat_sim,transformers
    from nwm_quality_benchmark import make_encoder
    from vlnce_baselines.nwm.panorama_context import make_context_plan
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    from vlnce_baselines.nwm.active_lookahead.dino_cwp_future import load_dino_cwp_predictor,decode_dino_cwp_top1
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    spec=json.loads(Path(a.manifest).read_text()); scenes=spec['scenes'][a.shard::a.shards]
    jobs=[j for j in spec['jobs'] if j['scene'] in scenes]
    if a.limit:jobs=jobs[:a.limit]
    versions=dict(python=__import__('sys').version,torch=torch.__version__,transformers=transformers.__version__,cuda=torch.version.cuda,
                  habitat=getattr(habitat,'__version__','unknown'),habitat_sim=getattr(habitat_sim,'__version__','unknown'),
                  git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gpu=torch.cuda.get_device_name(0))
    hashes={}
    for file in ['pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar','pretrained/raenwm_stage0/stat.pt','pretrained/active_lookahead/dino_cwp_best.pt']:
        h=hashlib.sha256()
        with open(file,'rb') as f:
            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
        hashes[file]=h.hexdigest()
    assert hashes['pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar']=='38b24af13b76ba8faef367559244c3a0401e0557e7c299870c273cbee8a07064'
    assert hashes['pretrained/raenwm_stage0/stat.pt']=='84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59'
    dump(out/'metadata.json',dict(versions=versions,hashes=hashes,scenes=scenes,manifest_sha=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()))
    enc,norm=make_encoder(); predictor=RaeNwmPredictor('configs/nwm/raenwm_mp3d_fresh_cls.yaml','pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar',device='cuda:0',use_external_context_latents=True)
    cwp,_=load_dino_cwp_predictor('pretrained/active_lookahead/dino_cwp_best.pt',expected_sha256='6a45291219907dd027203d224f3f8400631651a83bd01c45b1dea55d93ec0979',device=torch.device('cuda:0'))
    rows=[];cr=[];render=None;last_scene=None
    for number,j in enumerate(jobs):
        if j['scene']!=last_scene:
            if render:render.sim.close()
            render=Renderer(Path('data/scene_datasets')/j['scene']);last_scene=j['scene']
        plan=make_context_plan(j['positions'],j['yaws'],j['target'],j['target_yaw'],'world_exact_select')
        frames=[render.render(j['positions'][i],plan.view_yaws[i]) for i in plan.order]+[render.render(j['target'],j['target_yaw'])]
        images=[f[k] for k in CAMERAS for f in frames]; tokens=enc(images).reshape(4,5,257,768).float()
        assert torch.isfinite(tokens).all()
        np.savez_compressed(out/(j['id']+'_rgb.npz'),**{k:np.stack([f[k] for f in frames]) for k in CAMERAS})
        with torch.inference_mode():
            truth_cwp=decode_dino_cwp_top1(cwp(tokens[:,4,1:].cuda()),none_threshold=.3)
        for k,p in zip(CAMERAS,truth_cwp):
            cr.append(dict(id=j['id'],scene=j['scene'],camera=k,input='real',seed=None,none=p.pred_none,
                angle=p.local_angle_deg,distance=p.distance_m,clear=render.clear(j['target'],j['target_yaw'],p)))
        saved=[]
        for seed in spec['seeds']:
            gen=torch.Generator(device='cuda:0').manual_seed(int.from_bytes(hashlib.sha256(f"{j['id']}:{seed}".encode()).digest()[:8],'little')%(2**63-1))
            noise=torch.randn(1,257,768,device='cuda:0',generator=gen).repeat(4,1,1)
            with torch.inference_mode():
                _,pred=predictor._predict_time_from_latents(tokens[:,:4],torch.tensor([plan.delta]*4)[:,None],torch.tensor([plan.rel_t]*4),initial_noise=noise)
                pred=pred.float();assert torch.isfinite(pred).all()
                pp=decode_dino_cwp_top1(cwp(pred[:,1:]),none_threshold=.3)
                rawp=norm.denormalize_cls(pred[:,0]);rawg=norm.denormalize_cls(tokens[:,4,0].cuda());raws=norm.denormalize_cls(tokens[:,3,0].cuda())
                for ix,k in enumerate(CAMERAS):
                    gt=tokens[ix,4].cuda();p=pp[ix];true=truth_cwp[ix]
                    rows.append(dict(id=j['id'],scene=j['scene'],camera=k,seed=seed,horizon=j['horizon_frames'],
                        cls_cosine=F.cosine_similarity(rawp[ix],rawg[ix],dim=0).item(),
                        cls_rmse=(rawp[ix]-rawg[ix]).square().mean().sqrt().item(),
                        patch_cosine=F.cosine_similarity(pred[ix,1:],gt[1:],dim=-1).mean().item(),
                        source_cls_cosine=F.cosine_similarity(raws[ix],rawg[ix],dim=0).item()))
                    cr.append(dict(id=j['id'],scene=j['scene'],camera=k,input='predicted',seed=seed,none=p.pred_none,
                        angle=p.local_angle_deg,distance=p.distance_m,clear=render.clear(j['target'],j['target_yaw'],p),
                        real_none_agree=p.pred_none==true.pred_none,
                        real_angle_error=abs(p.local_angle_deg-true.local_angle_deg) if not p.pred_none and not true.pred_none else None,
                        real_distance_error=abs(p.distance_m-true.distance_m) if not p.pred_none and not true.pred_none else None))
            saved.append(pred.detach().cpu().half())
        torch.save(dict(tokens=tokens.half(),predictions=torch.stack(saved),query=j,plan=plan.__dict__),out/(j['id']+'_features.pt'))
        dump(out/'progress.json',dict(completed=number+1,total=len(jobs),last=j['id']))
        dump(out/'results.json',dict(status='running',world=rows,cwp=cr))
        print(json.dumps(dict(completed=number+1,total=len(jobs),id=j['id'])),flush=True)
    if render:render.sim.close()
    dump(out/'results.json',dict(status='completed',world=rows,cwp=cr)); print('COMPLETED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',required=True)
    p.add_argument('--scenes',type=int,default=6);p.add_argument('--trajectories',type=int,default=2)
    p.add_argument('--source',default='/home/gwl/project/RAE-NWM/tools/data/mp3d_full_h125_224_merged');p.add_argument('--manifest')
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=3);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();prepare(a) if a.action=='prepare' else run(a)
