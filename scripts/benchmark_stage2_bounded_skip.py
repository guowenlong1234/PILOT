#!/usr/bin/env python3
"""Paired real-navigation pruning benchmark; evaluation host, sequential GPU use."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from rgb_only_optimization import ROOT, now, save
from run_stage2_navigation_comparison import results


def read(path):
    return json.loads(Path(path).read_text())


def decisions(path):
    out={}
    for line in (path/'online/decisions.jsonl').read_text().splitlines():
        row=json.loads(line);key=(row['episode'],row['step'])
        if key in out:raise ValueError('duplicate decision')
        out[key]=row
    if not out:raise ValueError('empty trace')
    return out


def compare(left,right):
    a,b=decisions(left),decisions(right)
    mismatches=[]
    for key in a.keys()|b.keys():
        if key not in a or key not in b or any(a[key][field]!=b[key][field]
                for field in ('base_action','action','forced_stop','ghost_indices')):
            mismatches.append(key)
    metrics_equal=results(left)==results(right)
    return dict(passed=not mismatches and metrics_equal,decisions=len(a),
                mismatches=mismatches[:20],mismatch_count=len(mismatches),
                exact_episode_metrics=metrics_equal)


def percentile(values,q):
    values=sorted(values)
    return values[round((len(values)-1)*q)] if values else None


def summarize(path):
    online=read(path/'online/online_summary.json')
    timing_files=list((path/'results').rglob('timing_*.json'))
    if len(timing_files)!=1:raise ValueError('missing/ambiguous timing')
    timing=read(timing_files[0]);metrics=results(path)
    # Same fixed warm-up rule in every run, including short runs.
    warm=online['step_timings'][10:]
    durations=[r['seconds'] for r in warm]
    return dict(rollout_seconds=timing['elapsed_seconds'],online_seconds=online['online_seconds'],
        counts=online['counts'],warm_batches=len(warm),
        warm_online_seconds=sum(durations),warm_rows=sum(r['batch'] for r in warm),
        warm_batch_p50=percentile(durations,.5),warm_batch_p95=percentile(durations,.95),
        episodes=len(metrics),metrics={k:statistics.mean(r[k] for r in metrics.values())
                                      for k in next(iter(metrics.values()))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--head',required=True);p.add_argument('--episodes',type=int,default=128)
    p.add_argument('--environments',type=int,default=4)
    a=p.parse_args()
    if str(ROOT)!='/home/a6000/gwl/ETP-R1':raise ValueError('run in evaluation main workspace')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    stages=[('smoke_off','off',16),('smoke_on','on',16),
            ('bench_off_1','off',a.episodes),('bench_on_1','on',a.episodes),
            ('bench_on_2','on',a.episodes),('bench_off_2','off',a.episodes)]
    save(out/'contract.json',dict(args=vars(a),stages=stages,warmup_batches=10,
        model='native_9200_full',order='off/on/on/off',created_at=now()))
    summaries={};comparisons={}
    try:
        for name,mode,episodes in stages:
            cmd=[sys.executable,'scripts/stage2_e24_job.py','online','--machine','eval','--gpu','0',
                 '--environments',str(a.environments),'--split','val_unseen','--base-step','9200',
                 '--deployment-mode','native_9200','--checkpoint',a.checkpoint,'--head',a.head,
                 '--gain','1','--episodes',str(episodes),'--output',str(out/name),
                 '--bounded-skip',mode,'--trace','--profile']
            save(out/'pipeline.json',dict(status='running',stage=name,command=cmd,updated_at=now()))
            with (out/(name+'.log')).open('x') as log:
                code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if code:raise RuntimeError(f'{name} exit {code}')
            for file in ('freeze_report.json','world_freeze.json'):
                frozen=read(out/name/'online'/file)
                if not frozen.get('comparison',frozen)['exact_match']:raise ValueError('frozen weights changed')
            summaries[name]=summarize(out/name)
            reference='smoke_off' if name=='smoke_on' else 'bench_off_1'
            if name not in ('smoke_off','bench_off_1'):
                comparisons[name]=compare(out/reference,out/name)
                save(out/'comparisons.json',comparisons)
                if not comparisons[name]['passed']:raise ValueError('paired actions or metrics differ: '+name)
            save(out/'runs.json',summaries)
        groups={mode:[summaries[f'bench_{mode}_{i}'] for i in (1,2)] for mode in ('off','on')}
        average={mode:{key:statistics.mean(r[key] for r in runs)
                       for key in ('rollout_seconds','online_seconds','warm_batch_p50','warm_batch_p95')}
                 for mode,runs in groups.items()}
        for mode,runs in groups.items():
            average[mode]['q1_requested']=statistics.mean(r['counts'].get('q1_requested',0) for r in runs)
            average[mode]['head_rows']=statistics.mean(r['counts'].get('head_rows',0) for r in runs)
        improvement={key:1-average['on'][key]/average['off'][key]
                     for key in average['off'] if average['off'][key]}
        save(out/'summary.json',dict(status='passed',runs=summaries,comparisons=comparisons,
             average=average,reduction_fraction=improvement,
             limitation='128-route performance subset by default; not full navigation benchmark; two repeats'))
        save(out/'pipeline.json',dict(status='completed',exit_code=0,updated_at=now()))
    except Exception as error:
        save(out/'pipeline.json',dict(status='failed',error=repr(error),updated_at=now()))
        raise


if __name__=='__main__':main()
