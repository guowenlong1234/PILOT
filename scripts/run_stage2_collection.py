#!/usr/bin/env python3
"""Finite collect -> full validation -> E24 forward check, then exit. No training."""
import json
import subprocess
import sys
from pathlib import Path
from rgb_only_optimization import ROOT, runtime, save


def main():
    args=sys.argv[1:]
    if not args or args[0]!='collect':raise ValueError('only collection is supported')
    def value(key,default=None):return args[args.index(key)+1] if key in args else default
    out=Path(value('--output')).resolve();machine=value('--machine','server');gpu=value('--gpu','0')
    out.mkdir(parents=True,exist_ok=True)
    for action in ('collect','validate'):
        command=[sys.executable,str(ROOT/'scripts/stage2_e24_job.py'),action,*args[1:]]
        save(out/'pipeline.json',dict(status='running',stage=action,command=command))
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:
            save(out/'pipeline.json',dict(status='failed',stage=action,exit_code=result.returncode))
            return result.returncode
    command=runtime(machine,['scripts/check_stage2_train_ready.py',str(out/'episodes'),
        '--report',str(out/'train_ready.json')],gpu)
    save(out/'pipeline.json',dict(status='running',stage='forward_only',command=command))
    result=subprocess.run(command,cwd=ROOT)
    save(out/'pipeline.json',dict(status='ready' if result.returncode==0 else 'failed',
        stage='complete',exit_code=result.returncode,training_started=False))
    return result.returncode

if __name__=='__main__':sys.exit(main())
