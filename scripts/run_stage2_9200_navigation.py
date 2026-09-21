#!/usr/bin/env python3
"""Finite native 9200 base/full/none navigation on the evaluation host only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from rgb_only_optimization import ROOT,save,sha,now
from run_stage2_navigation_comparison import results,validate_parity


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args()
    if str(ROOT)!='/home/a6000/gwl/ETP-R1-stage2-e24':raise ValueError('evaluation host only')
    exp=Path(a.experiment).resolve();out=exp/'navigation'
    if out.exists():raise FileExistsError('navigation requires a fresh directory; inspect any prior attempt')
    summary=read(exp/'final_report/summary.json')
    if summary['status']!='complete':raise ValueError('offline comparison audit incomplete')
    out.mkdir()
    heads={mode:summary['selected'][mode]['checkpoint'] for mode in ('full','none')}
    for mode,path in heads.items():
        if sha(path)!=summary['selected'][mode]['checkpoint_sha256']:raise ValueError('selected head changed')
    save(out/'contract.json',dict(heads=summary['selected'],gain=1.,base_step=9200,predeclared_baseline_repetitions=4))
    def execute(name,command):
        save(out/'pipeline.json',dict(status='running',stage=name,command=command,updated_at=now()))
        with (out/(name+'.log')).open('x') as log:
            code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if code:raise RuntimeError(f'{name} failed ({code})')
    try:
        for mode in heads:
            report=out/f'replay_{mode}.json'
            execute('replay_'+mode,[sys.executable,'scripts/stage2_offline_worker.py','--machine','eval',
                    '--job-dir',str(out/'jobs'/('replay_'+mode)),'--','scripts/check_stage2_online_replay.py',
                    '--data',str(exp/'data/dev/episodes'),'--head',heads[mode],'--gain','1','--report',str(report)])
            if read(report)['status']!='passed':raise ValueError('online replay failed')
        def nav(name,action,mode='full',episodes=16,gain=0):
            cmd=[sys.executable,'scripts/stage2_e24_job.py',action,'--machine','eval','--gpu','0',
                 '--environments','4','--split','val_unseen','--base-step','9200','--deployment-mode','native_9200',
                 '--checkpoint','stage2_assets/ckpt.iter9200.pth','--head',heads[mode],'--gain',str(gain),
                 '--episodes',str(episodes),'--output',str(out/name)]
            if episodes>0:cmd+=['--trace']
            execute(name,cmd);results(out/name)
            if action=='online':
                if not read(out/name/'online/freeze_report.json')['comparison']['exact_match']:raise ValueError('head/base changed')
                if not read(out/name/'online/world_freeze.json')['exact_match']:raise ValueError('world changed')
        # Fixed before seeing any E24 output; no adaptive widening of tolerance.
        for n in range(4):nav(f'smoke_base{n}','baseline')
        for mode in heads:
            nav('smoke_zero_'+mode,'online',mode)
            parity=validate_parity(out/'smoke_base0',out/('smoke_zero_'+mode),[out/f'smoke_base{n}' for n in range(1,4)])
            save(out/f'parity_{mode}.json',parity)
            nav('smoke_'+mode,'online',mode,gain=1)
        nav('base','baseline',episodes=-1)
        for mode in heads:nav(mode,'online',mode,episodes=-1,gain=1)
        save(out/'pipeline.json',dict(status='completed',exit_code=0,updated_at=now()))
        execute('report',[sys.executable,'scripts/stage2_offline_worker.py','--machine','eval',
                '--job-dir',str(out/'jobs/report'),'--','scripts/report_stage2_9200_navigation.py','--experiment',str(exp)])
        save(out/'pipeline.json',dict(status='completed',exit_code=0,updated_at=now()))
    except Exception as e:
        save(out/'pipeline.json',dict(status='failed',error=str(e),updated_at=now()));raise

if __name__=='__main__':main()
