#!/usr/bin/env python3
"""Frozen collection and explicitly versioned online navigation evaluation."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from rgb_only_optimization import ROOT, common, resources, runtime, sha, save, VERSIONS

BASE_REL='data/logs/ghost_concat_persistent_compiled_10k_20260911/train/ghost_concat_v1_joint_persistent_train/checkpoints/ghost_concat_v1_joint_persistent_train/ckpt.iter6400.pth'
BASE_SHA='4c729c84bf4338452da4d459fc82734dcbb5f72ac6a2b574ee8f20e1080bc2fe'

BASE_SHAS={6400:BASE_SHA,9200:'87bf7ad691a93abfe2d5030c2c314ef4e630c41d3abbe61fea3b38872686055e'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['collect','validate','baseline','online'])
    p.add_argument('--machine',choices=['server','eval'],default='server')
    p.add_argument('--gpu',default='0');p.add_argument('--environments',type=int,default=4)
    p.add_argument('--split',choices=['train','val_unseen'],default='train')
    p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--episodes',type=int,default=-1);p.add_argument('--part',type=int,default=0);p.add_argument('--parts',type=int,default=1)
    p.add_argument('--base-step',type=int,choices=[6400,9200],default=6400)
    p.add_argument('--head',default='');p.add_argument('--gain',type=float,default=1.5)
    p.add_argument('--compact-storage',action='store_true')
    p.add_argument('--deployment-mode',choices=['legacy','native_9200'],default='legacy')
    p.add_argument('--trace',action='store_true');p.add_argument('--no-compile',action='store_true')
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.deployment_mode=='native_9200' and a.base_step!=9200:p.error('native_9200 requires base-step9200')
    if a.gpu not in ('0','1') or not 0<=a.part<a.parts or a.environments<1: p.error('invalid resources/partition')
    root=Path(a.output).resolve()
    if a.action in ('online','baseline') and root.exists() and any(root.iterdir()):
        raise FileExistsError('navigation comparison requires a new empty output directory')
    if a.action=='online' and (a.split!='val_unseen' or a.gain not in ((0.,1.) if a.deployment_mode=='native_9200' else (0.,1.5))):
        raise ValueError('online experiment is fixed to val_unseen and gain0/1.5')
    root.mkdir(parents=True,exist_ok=True)
    with gzip.open(ROOT/f'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/{a.split}/{a.split}.json.gz','rt') as f: data=json.load(f)
    ids=sorted(str(ep['episode_id']) for ep in data['episodes'])
    ids=ids[a.part::a.parts]
    if a.episodes>0:
        # Smoke subsets cover scenes before taking another episode per scene,
        # so an eight-environment preflight exercises the actual worker count.
        from collections import defaultdict
        groups=defaultdict(list); allowed=set(ids)
        for ep in data['episodes']:
            if str(ep['episode_id']) in allowed:groups[str(ep['scene_id'])].append(str(ep['episode_id']))
        groups=[sorted(v) for _,v in sorted(groups.items())]
        ids=[g[i] for i in range(max(map(len,groups))) for g in groups if i<len(g)][:a.episodes]
    if a.action=='validate':
        code='from vlnce_baselines.nwm.active_lookahead.stage2_data import validate_dataset; import json; m=validate_dataset('+repr(str(root/'episodes'))+','+repr(ids)+'); print(json.dumps({k:v for k,v in m.items() if k!="entries"}))'
        return subprocess.call(runtime(a.machine,['-c',code],a.gpu),cwd=ROOT)
    base_sha=BASE_SHAS[a.base_step]
    if sha(a.checkpoint)!=base_sha:raise ValueError('stage1 SHA mismatch')
    assets={}
    for rel in ('pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar','pretrained/raenwm_stage0/stat.pt','pretrained/active_lookahead/dino_cwp_best.pt'):
        assets[rel]=sha(ROOT/rel)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    provenance=dict(format='stage2-panorama-predicted-v1',stage1_sha256=base_sha,assets=assets,
        commit=commit,split=a.split,part=a.part,parts=a.parts,episode_ids=ids,seed=20260916,
        feature_space='raw_cls+normalized_patch_fp16',context_contract='stage2_panorama_q0_snapshot_v1',
        behavior='stage1_argmax',environments=a.environments,compile=not a.no_compile)
    if a.compact_storage:
        provenance['storage_format']='valid_future_only_v1'
    if a.base_step!=6400:
        provenance['stage1_iteration']=a.base_step
    if a.base_step!=6400 and a.action=='online' and a.deployment_mode=='legacy':
        provenance.update(stage1_iteration=a.base_step,head_training_base_sha256=BASE_SHA,transfer_mode='6400_to_9200',head_retrained=False)
    if a.action=='online' and a.deployment_mode=='native_9200':
        provenance.update(head_training_base_sha256=base_sha,transfer_mode='native_9200',head_retrained=True)
    if a.action=='online':
        provenance.update(behavior='stage2_argmax_stop_isolated',head_sha256=sha(a.head),gain=a.gain,residual_bound=1.)
    prov=root/'provenance.json'
    if prov.exists() and json.loads(prov.read_text())!=provenance:raise ValueError('immutable collection provenance changed')
    save(prov,provenance)
    # A crash after tensor publication but before metadata publication leaves
    # an uncommitted shard. Preserve it, then replay the whole episode.
    episode_root=root/'episodes'
    for path in list(episode_root.glob('*.pt'))+list(episode_root.glob('*.pt.tmp')):
        if path.suffix=='.pt' and path.with_suffix('.json').exists():continue
        quarantine=root/'uncommitted';quarantine.mkdir(exist_ok=True)
        path.rename(quarantine/(str(time.time_ns())+'_'+path.name))
    done=set()
    for path in (root/'episodes').glob('*.json'):
        d=json.loads(path.read_text())
        if 'file' in d:
            if sha(path.parent/d['file'])!=d['sha256']:raise ValueError('completed shard damaged')
            done.add(d['episode_id'])
    if done-set(ids):raise ValueError('unexpected completed episodes')
    todo=[i for i in ids if i not in done]
    if not todo:
        print('All requested episodes already collected; run validate.');return 0
    opts=common(a.gpu,a.environments)
    opts.update({'IL.freeze_navigation_backbone':True,'IL.is_requeue':False,'IL.use_fused_adamw':False,
        'MODEL.RAENWM.rgb_fusion_trainable':False,'MODEL.RAENWM.torch_compile':not a.no_compile,
        'MODEL.RAENWM.compile_backend':'inductor','MODEL.RAENWM.panorama_context_mode':'world_exact_select',
        'MODEL.RAENWM.panorama_observation_source':'direct','MODEL.RAENWM.panorama_visual_precision':'fp16',
        'MODEL.RAENWM.panorama_encode_batch_size':64,'MODEL.RAENWM.panorama_prediction_batch_size':64,
        'MODEL.ACTIVE_LOOKAHEAD.dino_cwp_checkpoint_path':'pretrained/active_lookahead/dino_cwp_best.pt',
        'MODEL.ACTIVE_LOOKAHEAD.dino_cwp_checkpoint_sha256':'6a45291219907dd027203d224f3f8400631651a83bd01c45b1dea55d93ec0979',
        'MODEL.STAGE2_COLLECT.enabled':a.action=='collect', 'MODEL.STAGE2_COLLECT.output':str(root/'episodes'),
        'MODEL.STAGE2_COLLECT.provenance':str(prov),'MODEL.STAGE2_COLLECT.trace':a.trace,
        'EVAL.SPLIT':a.split,'EVAL.EPISODE_ID':todo,'EVAL.EPISODE_COUNT':-1,
        'EVAL.CKPT_PATH_DIR':str(Path(a.checkpoint).resolve()),'EVAL.SAVE_RESULTS':True,
        'EVAL.USE_CKPT_CONFIG':False,'TASK_CONFIG.DATASET.SUFFIX':'',
        'RESULTS_DIR':str(root/'results')+'/', 'TENSORBOARD_DIR':str(root/'tensorboard')+'/',
        'CHECKPOINT_FOLDER':str(root/'checkpoints')+'/'})
    if a.action=='online':
        opts.update({'MODEL.STAGE2_ONLINE.enabled':True,'MODEL.STAGE2_ONLINE.output':str(root/'online'),
            'MODEL.STAGE2_ONLINE.head':str(Path(a.head).resolve()),'MODEL.STAGE2_ONLINE.head_sha256':provenance['head_sha256'],
            'MODEL.STAGE2_ONLINE.transfer_mode':'native_9200' if a.deployment_mode=='native_9200' else ('6400_to_9200' if a.base_step==9200 else 'same'),
            'MODEL.STAGE2_ONLINE.gain':a.gain,'MODEL.STAGE2_ONLINE.trace':a.trace})
    cmd=['run.py','--exp_name','stage2_collect','--run-type','eval','--exp-config','run_r2r/iter_train_rae_dino_ghost_concat_persistent.yaml']
    for key,value in opts.items():cmd.extend([key,str(value)])
    command=runtime(a.machine,cmd,a.gpu)
    launch=dict(command=command,provenance=provenance,pending_episodes=todo,
                resume_policy='complete_episode_only; stage1 RNG restarts for remaining episodes')
    save(root/'launch.json',launch)
    save(root/'attempts'/f'{time.time_ns()}.json',launch)
    if a.dry_run:print(json.dumps(command));return 0
    with resources(a.machine,a.gpu):
        with (root/'run.log').open('a') as log:
            result=subprocess.run(runtime(a.machine,['-c',VERSIONS],a.gpu),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:return result.returncode
            save(root/'status.json',dict(status='running',pid=os.getpid(),pending=len(todo)))
            started=time.monotonic()
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            save(root/'status.json',dict(status=('collected' if a.action=='collect' else 'completed') if result.returncode==0 else 'failed',exit_code=result.returncode,elapsed_seconds=time.monotonic()-started))
            return result.returncode

if __name__=='__main__':sys.exit(main())
