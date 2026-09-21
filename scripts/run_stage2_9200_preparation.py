#!/usr/bin/env python3
"""Finite 4-environment real preflight followed by the 9200 paired experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from rgb_only_optimization import ROOT,save,now


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    if str(ROOT)!='/home/a6000/gwl/ETP-R1-stage2-e24':raise ValueError('evaluation host only')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    state=root/'preparation.json'
    if state.exists():raise FileExistsError('inspect existing preparation; no blind restart')
    def run(name,cmd):
        save(state,dict(status='running',stage=name,command=cmd,updated_at=now()))
        with (root/(name+'_supervisor.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            save(root/(name+'_launch.json'),dict(pid=child.pid,command=cmd,started_at=now()))
            code=child.wait()
        if code:raise RuntimeError(f'{name} failed: {code}')
    try:
        for split,name in [('train','preflight_train64_env4'),('val_unseen','preflight_dev64_env4')]:
            run(name,[sys.executable,'scripts/run_stage2_collection.py','collect','--machine','eval',
                '--gpu','0','--environments','4','--base-step','9200','--checkpoint','stage2_assets/ckpt.iter9200.pth',
                '--compact-storage','--split',split,'--episodes','64','--trace','--output',str(root/name)])
            if read(root/name/'pipeline.json')['status']!='ready':raise ValueError('short collection failed')
            if read(root/name/'episodes/dataset_manifest.json')['episodes']!=64:raise ValueError('short coverage incomplete')
        for mode in ('full','none'):
            name='smoke_update_'+mode
            run(name,[sys.executable,'scripts/stage2_offline_worker.py','--machine','eval','--gpu','0',
                '--job-dir',str(root/'jobs'/name),'--','scripts/train_stage2_e24.py',str(root/'preflight_train64_env4/episodes'),
                '--output',str(root/name),'--future-mode',mode,'--precision','bf16','--max-steps','2',
                '--save-every','2','--log-every','1','--batch-size','32'])
            status=read(root/name/'status.json');audit=read(root/name/'parameter_audit.json')
            if status['status']!='completed' or status['step']!=2 or audit['initial_sha256']==audit['final_sha256']:
                raise ValueError('real head updates failed')
            replay='smoke_replay_'+mode
            run(replay,[sys.executable,'scripts/stage2_offline_worker.py','--machine','eval','--gpu','0',
                '--job-dir',str(root/'jobs'/replay),'--','scripts/check_stage2_online_replay.py',
                '--data',str(root/'preflight_dev64_env4/episodes'),'--head',str(root/name/'head_step_000002.pt'),
                '--gain','1','--report',str(root/(replay+'.json'))])
            if read(root/(replay+'.json'))['status']!='passed':raise ValueError('replay failed')
        run('formal',[sys.executable,'scripts/run_stage2_9200_ablation.py','--output',str(root/'formal_v1'),
            '--preflight-train',str(root/'preflight_train64_env4/episodes'),
            '--preflight-dev',str(root/'preflight_dev64_env4/episodes')])
        save(state,dict(status='completed',exit_code=0,updated_at=now()))
    except Exception as e:
        save(state,dict(status='failed',error=str(e),updated_at=now()));raise

if __name__=='__main__':main()
