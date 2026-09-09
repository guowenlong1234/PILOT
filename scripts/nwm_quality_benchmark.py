#!/usr/bin/env python3
"""Scene-disjoint prediction-quality benchmark; no navigation SR/SPL selection."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def prepare(args):
    root=Path(args.output);path=root/'split.json'
    if path.exists():raise RuntimeError('split already fixed')
    episodes=json.load(gzip.open(args.dataset,'rt'))['episodes']
    by_scene=defaultdict(list)
    for ep in episodes:by_scene[Path(ep['scene_id']).stem].append(str(ep['episode_id']))
    scenes=sorted(by_scene);rng=random.Random(20260909);rng.shuffle(scenes)
    # Previously visualized scene stays in development, never confirmation.
    old='zsNo4HB9uLZ'
    if old in scenes:scenes.remove(old);scenes.insert(0,old)
    cut=(len(scenes)+1)//2;jobs=[]
    for i,scene in enumerate(scenes):
        ids=sorted(by_scene[scene],key=int);rng.shuffle(ids)
        for ep in ids[:2]:jobs.append({'scene':scene,'episode':ep,'split':'development' if i<cut else 'confirmation'})
    report={'seed':20260909,'jobs':jobs,'development_scenes':scenes[:cut],
        'confirmation_scenes':scenes[cut:],'decisions_per_episode':4,
        'primary_metric':'mean over scenes of raw CLS cosine (queries averaged within each scene)',
        'secondary_metric':'normalized patch cosine; candidate must not lose >0.01 vs baseline on development',
        'selection_seed':11,'confirmation_seeds':[11,29,47],
        'selection_modes':['front','relative12','world30','world_exact','world30_reverse','world30_select','world_exact_select'],
        'selection_rule':'highest development primary among secondary-eligible; confirmation not used to select'}
    dump(path,report);print(json.dumps(report),flush=True)


def collect(args):
    root=Path(args.output);spec=json.loads((root/'split.json').read_text())
    jobs=spec['jobs'][args.shard::args.shards];statuses=[]
    for job in jobs:
        dest=root/'captures'/job['episode'];manifest=dest/'manifest.json'
        if manifest.exists():
            m=json.loads(manifest.read_text())
            if m['status']=='completed' and m['decisions']:
                statuses.append({'episode':job['episode'],'exit_code':0,'reused':True});continue
            raise RuntimeError('failed prior capture requires a fresh output: '+str(dest))
        log=root/'capture_logs'/f"{job['episode']}.log";log.parent.mkdir(parents=True,exist_ok=True)
        command=['bash','scripts/rgb_only_optimization_runtime.sh','server','scripts/nwm_quality_collect.py',
            '--episode',job['episode'],'--decisions',str(spec['decisions_per_episode']),
            '--output',str(dest),'--checkpoint',args.checkpoint]
        started=time.time()
        with log.open('w') as f:rc=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        status={**job,'exit_code':rc,'seconds':time.time()-started};statuses.append(status)
        dump(root/f'collect_shard{args.shard}.json',statuses);print(json.dumps(status),flush=True)
        if rc:raise RuntimeError('capture failed: '+str(log))
    dump(root/f'collect_shard{args.shard}.json',statuses)


def make_encoder():
    from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder
    from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer
    from vlnce_baselines.nwm.raenwm_core.models import pack_cls_patch
    encoder=RaeDinov2RgbEncoder('pretrained/rae_dinov2_with_registers_base',precision='float32').cuda().eval()
    encoder.requires_grad_(False)
    normalizer=RaeNwmLatentNormalizer('pretrained/raenwm_stage0/stat.pt').cuda()
    def encode(images):
        values=[]
        for start in range(0,len(images),16):
            with torch.no_grad():
                cls,patch=encoder.forward_raw_cls_and_patch_latents({'rgb':np.stack(images[start:start+16])})
                values.append(pack_cls_patch(normalizer.normalize_cls(cls),normalizer.normalize_patch(patch)).cpu())
        return torch.cat(values)
    return encode,normalizer


def encode(args):
    from vlnce_baselines.nwm.panorama_context import MODES,make_context_plan,perspective_from_cube
    root=Path(args.output);spec=json.loads((root/'split.json').read_text());enc,norm=make_encoder()
    jobs=spec['jobs'][args.shard::args.shards];checks=[]
    for job in jobs:
        capture=root/'captures'/job['episode'];manifest=json.loads((capture/'manifest.json').read_text())
        if manifest['status']!='completed':raise RuntimeError('incomplete capture')
        for decision in manifest['decisions']:
            source=capture/decision['file'];dest=root/'encoded'/job['episode']/decision['file']
            if dest.exists():continue
            data=torch.load(source,map_location='cpu',weights_only=False)
            images=list(data['front_rgb']);keys={('front',i):i for i in range(4)}
            plans=[]
            for qi,target in enumerate(data['targets']):
                variants={}
                for mode in MODES:
                    plan=make_context_plan(data['positions'],data['yaws'],target['position'],target['yaw'],mode)
                    ids=[]
                    for i in plan.order:
                        if mode=='front':ids.append(keys[('front',i)]);continue
                        yaw=plan.view_yaws[i]
                        key=(i,round(float(yaw%(2*np.pi)),8))
                        if key not in keys:
                            keys[key]=len(images);images.append(perspective_from_cube(data['cube_rgb'][i],yaw))
                        ids.append(keys[key])
                    variants[mode]={'ids':ids,'delta':plan.delta,'rel_t':plan.rel_t,
                        'source_index':plan.source_index,'source_yaw':plan.source_yaw,
                        'order':plan.order,'view_yaws':plan.view_yaws,'fallback':plan.fallback}
                plans.append({'id':job['episode']+'_'+source.stem+'_'+target['query'],
                    'target':target,'variants':variants})
            features=enc(images)
            truth=enc(list(data['target_rgb']))
            # Compare cube extraction with a direct render of the same visited
            # pose. This reference is not a context option or target label.
            ref=data['projection_reference_rgb']
            projected=perspective_from_cube(data['cube_rgb'][-1],data['projection_reference_yaw'])
            ref_tokens=enc([ref,projected])
            cls=norm.denormalize_cls(ref_tokens[:,0].cuda()).cpu()
            check={'episode':job['episode'],'file':decision['file'],
                'projection_rgb_mae':float(np.abs(ref.astype(float)-projected.astype(float)).mean()/255),
                'projection_cls_cosine':F.cosine_similarity(cls[0],cls[1],dim=0).item(),
                'original_context_mean_abs':(features[:4]-data['original_context_tokens']).abs().mean().item(),
                'features':len(features),'queries':len(plans)}
            checks.append(check)
            payload={'job':job,'source':str(source),'queries':plans,
                'features':features.half(),'truth':truth.half(),'geometry':{
                    'positions':data['positions'],'yaws':data['yaws']},'checks':check}
            dest.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,dest)
            print(json.dumps(check),flush=True)
            dump(root/f'encode_shard{args.shard}.json',checks)
    dump(root/f'encode_shard{args.shard}.json',checks)


def score(args):
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer
    root=Path(args.output);spec=json.loads((root/'split.json').read_text())
    path=root/(args.name+'.json')
    if path.exists():raise RuntimeError('fresh score name required')
    modes=args.modes.split(',');seeds=[int(s) for s in args.seeds.split(',')]
    predictor=RaeNwmPredictor('configs/nwm/raenwm_mp3d_fresh_cls.yaml','pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar',device='cuda:0',use_external_context_latents=True)
    normalizer=RaeNwmLatentNormalizer('pretrained/raenwm_stage0/stat.pt')
    rows=[];batch_counter=0
    for job in spec['jobs']:
        if job['split']!=args.split:continue
        files=sorted((root/'encoded'/job['episode']).glob('decision_*.pt'))
        if not files:raise RuntimeError('missing encoded episode '+job['episode'])
        for file in files:
            data=torch.load(file,map_location='cpu',weights_only=False)
            work=[]
            for qi,query in enumerate(data['queries']):
                for seed in seeds:
                    noise_seed=int.from_bytes(hashlib.sha256((query['id']+':'+str(seed)).encode()).digest()[:8],'little')%(2**63-1)
                    gen=torch.Generator(device='cuda:0').manual_seed(noise_seed)
                    noise=torch.randn(257,768,device='cuda:0',generator=gen)
                    for mode in modes:work.append((qi,query,seed,mode,noise))
            for start in range(0,len(work),args.batch):
                part=work[start:start+args.batch];contexts=[];delta=[];rt=[]
                for qi,q,seed,mode,noise in part:
                    variant=q['variants'][mode]
                    contexts.append(data['features'][variant['ids']].float())
                    delta.append(variant['delta']);rt.append(variant['rel_t'])
                torch.cuda.synchronize();tic=time.perf_counter()
                with torch.no_grad():
                    _,pred=predictor._predict_time_from_latents(torch.stack(contexts),torch.tensor(delta)[:,None],torch.tensor(rt),initial_noise=torch.stack([w[4] for w in part]))
                torch.cuda.synchronize();seconds=time.perf_counter()-tic
                pred=pred.float().cpu();saved=[]
                for j,(qi,q,seed,mode,noise) in enumerate(part):
                    p=pred[j];gt=data['truth'][qi].float();v=q['variants'][mode]
                    raw_p=normalizer.denormalize_cls(p[None,0])[0]
                    raw_g=normalizer.denormalize_cls(gt[None,0])[0]
                    source=data['features'][v['ids'][-1]].float()
                    raw_s=normalizer.denormalize_cls(source[None,0])[0]
                    row={'id':q['id'],'scene':job['scene'],'episode':job['episode'],'mode':mode,'seed':seed,
                        'cls_cosine':F.cosine_similarity(raw_p,raw_g,dim=0).item(),
                        'patch_cosine':F.cosine_similarity(p[1:],gt[1:],dim=-1).mean().item(),
                        'cls_rmse':(raw_p-raw_g).square().mean().sqrt().item(),
                        'source_cls_cosine':F.cosine_similarity(raw_s,raw_g,dim=0).item(),
                        'source_patch_cosine':F.cosine_similarity(source[1:],gt[1:],dim=-1).mean().item(),
                        'original_turn_deg':q['target']['original_turn']*180/np.pi,
                        'query_turn_deg':v['delta'][2]*180/np.pi,'source_index':v['source_index'],
                        'fallback':v['fallback'],'target_navigable':q['target']['navigable'],
                        'nwm_seconds_per_query':seconds/len(part),'encoded_file':str(file)}
                    rows.append(row);saved.append({'id':q['id'],'mode':mode,'seed':seed})
                if args.save_predictions:
                    dest=root/(args.name+'_predictions')/f'batch_{batch_counter:05d}.pt';dest.parent.mkdir(exist_ok=True)
                    torch.save({'rows':saved,'tokens':pred.half()},dest)
                batch_counter+=1
            print(json.dumps({'episode':job['episode'],'file':file.name,'scored':len(rows)}),flush=True)
            dump(root/(args.name+'_progress.json'),{'rows':len(rows),'last':str(file)})
    result={'split':args.split,'modes':modes,'seeds':seeds,'rows':rows,'summary':summarize(rows)}
    dump(path,result);print(json.dumps(result['summary']),flush=True)


def summarize(rows):
    result={}
    for mode in sorted({r['mode'] for r in rows}):
        items=[r for r in rows if r['mode']==mode];scenes=sorted({r['scene'] for r in items})
        result[mode]={'queries':len({r['id'] for r in items}),'rows':len(items),'scenes':len(scenes)}
        for key in ('cls_cosine','patch_cosine','cls_rmse','source_cls_cosine','source_patch_cosine'):
            per_scene=[float(np.mean([r[key] for r in items if r['scene']==s])) for s in scenes]
            result[mode][key]=float(np.mean(per_scene))
        result[mode]['reverse_fraction']=float(np.mean([r['source_index']==0 for r in items]))
    return result


def select(args):
    root=Path(args.output);path=root/'selection.json'
    if path.exists():raise RuntimeError('selection already frozen')
    development=json.loads((root/(args.name+'.json')).read_text())
    if development['split']!='development':raise ValueError('selection only uses development')
    summary=development['summary'];baseline=summary['front']
    eligible=[m for m,s in summary.items() if s['patch_cosine']>=baseline['patch_cosine']-.01]
    winner=max(eligible,key=lambda m:summary[m]['cls_cosine'])
    report={'winner':winner,'source':args.name+'.json','criterion':'highest scene-macro raw CLS cosine; patch guard -0.01',
        'summary':summary,'confirmation_not_read':True}
    dump(path,report);print(json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['prepare','collect','encode','score','select'])
    ap.add_argument('--output',required=True);ap.add_argument('--dataset',default='data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz')
    ap.add_argument('--checkpoint');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--split',choices=['development','confirmation'],default='development')
    ap.add_argument('--name',default='development');ap.add_argument('--modes',default='front,relative12,world30,world_exact,world30_reverse,world30_select,world_exact_select')
    ap.add_argument('--seeds',default='11');ap.add_argument('--batch',type=int,default=8);ap.add_argument('--save-predictions',action='store_true')
    args=ap.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
