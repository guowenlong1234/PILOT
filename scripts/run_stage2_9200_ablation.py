#!/usr/bin/env python3
"""Finite evaluation-host-only collection, paired training and offline comparison.

Completed stages may be skipped on explicit resume; interrupted stages require
inspection. No remote execution, downloads or model/data transfers.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from rgb_only_optimization import ROOT, save, now, sha

BASE_SHA='87bf7ad691a93abfe2d5030c2c314ef4e630c41d3abbe61fea3b38872686055e'
GIB=1024**3

def read(path):
    return json.loads(Path(path).read_text())


def completed_stage(path):
    path=Path(path)
    if not path.exists():return False
    record=read(path)
    return record.get('status')=='completed' and record.get('exit_code')==0


def required_capacity(estimates, pending):
    return sum(estimates[name] for name in pending)*1.25+40*GIB


def publish_evaluation_index(index, report, checkpoint, step):
    evaluation=read(report)
    if evaluation.get('status')!='complete' or len(evaluation.get('results',[]))!=1:
        raise ValueError(f'step {step}: incomplete evaluator report')
    payload=dict(status='completed',exit_code=0,step=step,
        report_file=Path(report).name,report_sha256=sha(report),head_sha256=sha(checkpoint))
    save(Path(index),payload)
    return payload


def training_command(train, target, mode):
    if mode not in ('full','none'):raise ValueError('invalid future mode')
    return ['scripts/train_stage2_e24.py',str(train),'--output',str(target),
        '--future-mode',mode,'--variant','baseline','--precision','bf16','--seed','2',
        '--batch-size','32','--lr','2e-5','--max-steps','6000','--max-epochs','100',
        '--save-every','250','--log-every','20']


def record_pipeline_failure(argv, exc):
    try:
        index=argv.index('--output')
        out=Path(argv[index+1]).resolve()
        if out.exists():
            save(out/'pipeline.json',dict(status='failed',error_type=type(exc).__name__,
                error=str(exc),traceback=traceback.format_exc(),updated_at=now()))
    except Exception:
        # Never replace the original pipeline exception with reporting failure.
        pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--preflight-train',required=True)
    p.add_argument('--preflight-dev',required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if str(ROOT)!='/home/a6000/gwl/ETP-R1-stage2-e24':
        raise ValueError('this pipeline is restricted to evaluation host worktree')
    out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract=dict(base_sha256=BASE_SHA,source_commit=commit,machine='eval',seed=2,
        batch_size=32,steps=6000,save_every=250,precision='bf16',lr=2e-5,gain=1.,
        future_modes=['full','none'],mask_policy='same_actual_future_valid',
        preflight_train=str(Path(a.preflight_train).resolve()),preflight_dev=str(Path(a.preflight_dev).resolve()))
    if (out/'contract.json').exists():
        if not a.resume or read(out/'contract.json')!=contract:raise ValueError('explicit same-contract resume required')
    else:save(out/'contract.json',contract)
    def stage(name,command,required=None):
        record=out/'stages'/f'{name}.json'
        if record.exists():
            old=read(record)
            if old['command']!=command or old['status']!='completed' or old['exit_code']!=0:
                raise RuntimeError(f'{name}: prior incomplete attempt requires manual inspection')
            if required and not Path(required).exists():raise RuntimeError(f'{name}: completed artifact missing')
            return
        info=dict(status='running',command=command,started_at=now())
        save(record,info);save(out/'pipeline.json',dict(status='running',stage=name,pid=os.getpid(),updated_at=now()))
        try:
            with (out/'stages'/f'{name}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                info['child_pid']=child.pid;save(record,info);code=child.wait()
        except Exception as exc:
            info.update(status='failed',error_type=type(exc).__name__,error=str(exc),ended_at=now())
            save(record,info)
            raise
        info.update(status='completed' if code==0 else 'failed',exit_code=code,ended_at=now());save(record,info)
        if code:raise RuntimeError(f'{name} failed ({code}); see {record.with_suffix(".log")}')
        if required and not Path(required).exists():
            info.update(status='failed',error='required output missing')
            save(record,info)
            raise RuntimeError(f'{name}: output missing')
    def worker(name,command,required=None):
        stage(name,[sys.executable,'scripts/stage2_offline_worker.py','--machine','eval','--gpu','0',
                    '--job-dir',str(out/'jobs'/name),'--',*command],required)
    # Short-shard measurements provide a conservative capacity gate before any
    # full collection. Reserve model/checkpoint space plus the hard 20 GiB floor.
    estimates={}
    for label,root,total in (('train',a.preflight_train,10819),('dev',a.preflight_dev,1839)):
        root=Path(root);m=read(root/'dataset_manifest.json');prov=read(root/'provenance.json')
        if m['status']!='validated' or prov['stage1_sha256']!=BASE_SHA or prov.get('storage_format')!='valid_future_only_v1':
            raise ValueError('9200 compact preflight evidence required')
        if m['episodes']<1 or m['bytes']<1:raise ValueError('empty compact preflight evidence')
        estimates[label]=m['bytes']/m['episodes']*total
    # Single GPU: all stages execute strictly sequentially.
    for split,name in (('train','train'),('val_unseen','dev')):
        destination=out/'data'/name
        stage_record=out/'stages'/f'collect_{name}.json'
        if not completed_stage(stage_record):
            pending=[candidate for candidate in ('train','dev')
                     if not completed_stage(out/'stages'/f'collect_{candidate}.json')]
            free=shutil.disk_usage(out).free
            projected=required_capacity(estimates,pending)
            save(out/'capacity.json',dict(free_bytes=free,
                estimated_dataset_bytes=sum(estimates[item] for item in pending),
                required_bytes=projected,pending_collections=pending,
                margin=1.25,model_and_free_reserve_gib=40,updated_at=now()))
            if free<projected:
                raise OSError('insufficient capacity under measured 25% margin + 40GiB reserve')
        stage('collect_'+name,[sys.executable,'scripts/run_stage2_collection.py','collect','--machine','eval',
              '--gpu','0','--environments','4','--base-step','9200','--checkpoint','stage2_assets/ckpt.iter9200.pth',
              '--compact-storage','--split',split,'--output',str(destination)],destination/'train_ready.json')
        if read(destination/'pipeline.json')['status']!='ready':raise ValueError('collection gate failed')
    train=out/'data/train/episodes';dev=out/'data/dev/episodes'
    tm=read(train/'dataset_manifest.json');dm=read(dev/'dataset_manifest.json')
    if tm['episodes']!=10819 or dm['episodes']!=1839:raise ValueError('full data coverage required')
    if {r['scene_id'] for r in tm['entries']}&{r['scene_id'] for r in dm['entries']}:raise ValueError('train/dev scenes overlap')
    save(out/'data_identity.json',dict(train_manifest=sha(train/'dataset_manifest.json'),dev_manifest=sha(dev/'dataset_manifest.json')))
    for mode in ('full','none'):
        pre=out/mode/'preflight'
        stage('resume_gate_'+mode,[sys.executable,'scripts/run_stage2_offline_preflight.py',str(train),
              '--machine','eval','--future-mode',mode,'--output',str(pre)],pre/'resume_comparison.json')
        if read(pre/'pipeline.json')['status']!='passed':raise ValueError('training resume gate failed')
    for mode in ('full','none'):
        target=out/mode/'train'
        if not (out/'stages'/('train_'+mode+'.json')).exists() and shutil.disk_usage(out).free<40*1024**3:
            raise OSError('need 40GiB free before paired training stage')
        worker('train_'+mode,training_command(train,target,mode),target/'status.json')
        status=read(target/'status.json')
        if status['status']!='completed' or status['step']!=6000:raise ValueError('training budget incomplete')
        for step in range(250,6001,250):
            report=out/mode/'eval'/f'metrics_{step:06d}.json'
            worker(f'eval_{mode}_{step:06d}',['scripts/evaluate_stage2_e24.py','--data-roots',str(dev),
                   '--checkpoints',str(target/f'head_step_{step:06d}.pt'),'--report',str(report),
                   '--gains','1','--amp','bf16','--num-workers','0','--batch-size','16',
                   *(['--verify-hashes'] if step==250 else [])],report)
            publish_evaluation_index(out/mode/'eval'/f'step_{step:06d}.json',report,
                                     target/f'head_step_{step:06d}.pt',step)
    worker('report',['scripts/report_stage2_future_ablation.py','--output',str(out)],out/'final_report/summary.json')
    stage('navigation',[sys.executable,'scripts/run_stage2_9200_navigation.py','--experiment',str(out)],
          out/'navigation/final_report/summary.json')
    save(out/'pipeline.json',dict(status='completed',stage='offline_and_navigation_comparison',exit_code=0,updated_at=now()))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        record_pipeline_failure(sys.argv[1:],exc)
        raise
