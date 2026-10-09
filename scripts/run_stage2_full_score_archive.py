#!/usr/bin/env python3
"""Finite eval-host full R2R score archive with preflight and automatic audit."""
import argparse
from pathlib import Path
import subprocess
import sys
from rgb_only_optimization import ROOT,save,now,runtime


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--head',required=True);a=p.parse_args()
    if str(ROOT)!='/home/a6000/gwl/ETP-R1':raise ValueError('evaluation main workspace only')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    try:
        for name,episodes in (('preflight',16),('full',-1)):
            command=[sys.executable,'scripts/stage2_e24_job.py','online','--machine','eval','--gpu','0',
                '--environments','4','--split','val_unseen','--base-step','9200','--deployment-mode','native_9200',
                '--checkpoint',a.checkpoint,'--head',a.head,'--gain','1','--episodes',str(episodes),
                '--output',str(root/name),'--bounded-skip','off','--trace','--profile']
            save(root/'pipeline.json',dict(status='running',stage=name,command=command,updated_at=now()))
            with (root/(name+'.launcher.log')).open('x') as log:
                code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if code:raise RuntimeError(f'{name} navigation exit {code}')
            audit=runtime('eval',['scripts/report_stage2_score_trace.py',str(root/name),
                '--expected-episodes',str(16 if episodes==16 else 1839)],'0')
            with (root/(name+'.audit.log')).open('x') as log:
                code=subprocess.call(audit,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if code:raise RuntimeError(f'{name} score audit exit {code}')
        save(root/'pipeline.json',dict(status='completed',exit_code=0,updated_at=now()))
    except Exception as e:
        save(root/'pipeline.json',dict(status='failed',error=repr(e),updated_at=now()));raise


if __name__=='__main__':main()
