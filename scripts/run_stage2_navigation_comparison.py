"""Finite baseline / zero-gain parity / selected-E24 navigation evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from rgb_only_optimization import ROOT, save, now

OWNER=False


def read(path):
    return json.loads(Path(path).read_text())


def results(path):
    files=list((path/'results').rglob('stats_ep_*.json'))
    if len(files)!=1: raise ValueError('expected one complete per-episode result file')
    values=read(files[0]); expected=set(map(str,read(path/'provenance.json')['episode_ids']))
    if set(values)!=expected: raise ValueError('episode coverage differs')
    return values


def trace(path):
    values={}
    for line in (path/'run.log').read_text().splitlines():
        if line.startswith('STAGE2_BASE_TRACE '):
            value=json.loads(line.split(' ',1)[1]); key=(value['episode'],value['step'])
            if key in values: raise ValueError('duplicate trace decision')
            values[key]=value
    return values


def main():
    global OWNER
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--replay-report',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()) and not a.resume:raise FileExistsError('use a fresh comparison directory or explicit --resume')
    OWNER=True
    if (out/'pipeline.json').exists():
        save(out/'attempts'/(now().replace(':','-')+'.json'),read(out/'pipeline.json'))
    if read(a.replay_report)['status']!='passed':raise ValueError('offline replay gate missing')
    if str(ROOT)!='/home/a6000/gwl/ETP-R1-stage2-e24':raise ValueError('run on the evaluation host worktree')
    base='stage2_assets/ckpt.iter6400.pth'
    head='data/logs/stage2_e24/training_20260917/formal_seed2/heads/head_step_004750.pt'
    stages=[('smoke_base','baseline',16,0),('smoke_zero','online',16,0),('smoke_best','online',16,1.5),
            ('full_base','baseline',-1,0),('full_best','online',-1,1.5)]
    for name,action,episodes,gain in stages:
        command=[sys.executable,'scripts/stage2_e24_job.py',action,'--machine','eval','--gpu','0',
            '--environments','8','--split','val_unseen','--checkpoint',base,'--head',head,
            '--gain',str(gain),'--episodes',str(episodes),'--output',str(out/name)]
        if episodes>0:command+=['--trace']
        save(out/'pipeline.json',dict(status='running',stage=name,command=command,updated_at=now()))
        previous=out/name/'status.json'
        if a.resume and previous.exists() and read(previous).get('status')=='completed' and read(previous).get('exit_code')==0:
            code=0
        else:
            code=subprocess.call(command,cwd=ROOT)
        if code:raise RuntimeError(f'{name} failed with exit code {code}; inspect run.log before any retry')
        results(out/name)
        if action=='online':
            frozen=read(out/name/'online/freeze_report.json')
            if not frozen['comparison']['exact_match']:raise ValueError('model weights changed')
            if not read(out/name/'online/world_freeze.json')['exact_match']:raise ValueError('world weights changed')
        if name=='smoke_zero':
            left,right=trace(out/'smoke_base'),trace(out/'smoke_zero')
            if not left or left!=right:raise ValueError('zero-gain base actions/logits differ')
            if results(out/'smoke_base')!=results(out/'smoke_zero'):raise ValueError('zero-gain navigation metrics differ')
            decisions=[json.loads(s) for s in (out/name/'online/decisions.jsonl').read_text().splitlines()]
            if any(r['base_action']!=r['action'] or any(r['delta']) for r in decisions):raise ValueError('zero gain changed action')
            save(out/'parity.json',dict(status='passed',decisions=len(left),episodes=16,exact_logits_actions_metrics=True))
    left,right=results(out/'full_base'),results(out/'full_best')
    if set(left)!=set(right) or len(left)!=1839:raise ValueError('full comparison coverage differs')
    metrics={key:dict(base=sum(r[key] for r in left.values())/len(left),
                     e24=sum(r[key] for r in right.values())/len(right)) for key in next(iter(left.values()))}
    for item in metrics.values():item['difference']=item['e24']-item['base']
    transitions=dict(fixed=sum(left[k]['success']==0 and right[k]['success']==1 for k in left),
                     harmed=sum(left[k]['success']==1 and right[k]['success']==0 for k in left))
    timing={name:read(next((out/name/'results').rglob('timing_*.json'))) for name in ('full_base','full_best')}
    save(out/'comparison.json',dict(status='completed',episodes=len(left),metrics=metrics,success_transitions=transitions,
        timing=timing,limitation='R2R val_unseen development set; selected offline head, not independent test'))
    save(out/'pipeline.json',dict(status='completed',exit_code=0,updated_at=now()))


if __name__=='__main__':
    try:main()
    except Exception as e:
        if OWNER and '--output' in sys.argv:
            out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
            prior=read(out/'pipeline.json') if (out/'pipeline.json').exists() else {}
            save(out/'pipeline.json',dict(prior,status='failed',error=repr(e),updated_at=now()))
        raise
