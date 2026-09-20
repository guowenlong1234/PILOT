#!/usr/bin/env python3
"""Sequential, bounded benchmark runs with exit codes and GPU peak monitoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--updates', type=int, default=18)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--cases', default='baseline,parallel,env8,env10,env12')
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cases = {'baseline':(6,2,False), 'parallel':(6,2,True),
             'baselinelean':(6,2,False), 'baselinecompile':(6,2,False),
             'baselinevisual':(6,2,False), 'env12visual':(12,1,True),
             'baselineasync':(6,2,False), 'baselinevisualasync':(6,2,False),
             'env8serial':(8,1,False), 'env10serial':(10,1,False), 'env12serial':(12,1,False),
             'env8':(8,1,True), 'env10':(10,1,True), 'env12':(12,1,True),
             'env8lean':(8,1,True), 'env10lean':(10,1,True), 'env12lean':(12,1,True),
             'env11lean':(11,1,True), 'env11visual':(11,1,True),
             'env12visualasync':(12,1,True), 'env11visualasync':(11,1,True),
             'env12native':(12,1,True), 'env11native':(11,1,True), 'env10native':(10,1,True),
             'env12stress':(12,1,True), 'env11stress':(11,1,True), 'env10stress':(10,1,True),
             'env12late':(12,1,True), 'env11late':(11,1,True),
             'env12checkpoint':(12,1,True), 'env12checkpointlate':(12,1,True),
             'env14checkpoint':(14,1,True), 'env16checkpoint':(16,1,True),
             'env16checkpointlate':(16,1,True), 'env16checkpointstress':(16,1,True),
             'env8compile':(8,1,True), 'env10compile':(10,1,True), 'env12compile':(12,1,True)}
    summary = []
    for name in args.cases.split(','):
        envs, accumulation, parallel = cases[name]
        output = root/name
        if output.exists():
            raise RuntimeError(f'Refusing to overwrite {output}')
        cmd = ['bash','scripts/run_rxr_throughput_benchmark.sh','--output',str(output),
               '--checkpoint',args.checkpoint,'--updates',str(args.updates),
               '--warmup','2','--environments',str(envs),'--accumulation',str(accumulation),
               'IL.parallel_rxr_teacher',str(parallel)]
        if name.endswith('lean'):
            cmd.append('--lean-dino')
        if name.endswith('compile'):
            cmd.append('--compile-dino')
        if name.endswith('visual'):
            cmd.extend(['--compile-dino','--compile-depth'])
        if name.endswith('async'):
            cmd.append('--async-finite')
        if name.endswith('visualasync'):
            cmd.extend(['--compile-dino','--compile-depth'])
        if name.endswith('stress'):
            cmd.append('--scene-stress')
        if 'checkpoint' in name:
            cmd.extend(['MODEL.checkpoint_navigation','True'])
        if name.endswith('late'):
            cmd.extend(['IL.sample_ratio','0.178'])
        if name.endswith(('native','stress','late','checkpoint')):
            cmd.extend(['--config','run_rxr/iter_train_rae_dino_sft_fast.yaml'])
        if args.audit:
            cmd.append('--audit')
        samples = []
        start = time.time()
        with (root/f'{name}.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                query = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True)
                samples.append({'time':time.time(),'gpus':query.stdout.strip()})
                time.sleep(2)
        item = dict(name=name,command=cmd,exit_code=proc.returncode,wall_seconds=time.time()-start)
        reports = [json.loads(f.read_text()) for f in sorted(output.glob('rank*.json'))]
        if len(reports)==2 and proc.returncode==0:
            duration = max(r['seconds'] for r in reports)
            item.update(actions_per_second=sum(r['actions'] for r in reports)/duration,
                        trajectories_per_second=(args.updates-2)*2*envs*accumulation/duration,
                        seconds_per_update=duration/(args.updates-2),
                        all_updates_applied=all(s['optimizer_stepped'] for r in reports for s in r['steps'][2:]))
        (root/f'{name}_gpu.json').write_text(json.dumps(samples))
        summary.append(item)
        (root/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(item),flush=True)
        # Child failures must release CUDA before the next trial.
        for _ in range(30):
            query = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                '--format=csv,noheader,nounits'],text=True)
            if max(map(int,query.split())) < 1024:
                break
            time.sleep(2)
        else:
            raise RuntimeError('GPU memory did not clear; refusing to overlap trials')


if __name__ == '__main__':
    main()
