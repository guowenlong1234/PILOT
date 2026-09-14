#!/usr/bin/env python3
"""Deliver one checkpoint at a time; retain originals and all evaluation results."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from rgb_only_optimization import ROOT, sha, save

HOST = 'a6000@10.10.10.2'
REMOTE = Path('/home/a6000/gwl/ETP-R1')
NAME = 'ghost_concat_v1_joint_train'


def remote_python(code, *args):
    command = shlex.join(['python3', '-c', code, *map(str, args)])
    return subprocess.check_output(['ssh', '-n', '-o', 'BatchMode=yes',
                                   '-o', 'ConnectTimeout=15', HOST, command], text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--iters', type=int, default=10000)
    p.add_argument('--every', type=int, default=200)
    p.add_argument('--memory-mode', choices=['current_step_only', 'persistent_node_state'], default='current_step_only')
    p.add_argument('--iterations', help='Explicit ascending subset of checkpoints to deliver')
    args = p.parse_args()
    if str(ROOT) != '/home/gwl/project/etpr1/ETP-R1':
        p.error('Run in the training machine project workspace')
    if Path(args.output).is_absolute() or '..' in Path(args.output).parts:
        p.error('output must be project-relative')
    if min(args.every, args.iters) < 1 or args.iters % args.every:
        p.error('iters must be a positive multiple of every')
    iterations = list(range(args.every, args.iters + 1, args.every))
    if args.iterations:
        try:
            iterations = [int(value) for value in args.iterations.split(',')]
        except ValueError:
            p.error('iterations must be comma-separated integers')
        if (not iterations or iterations != sorted(set(iterations)) or
                min(iterations) < 1 or max(iterations) > args.iters or
                any(value % args.every for value in iterations)):
            p.error('iterations must be unique, ascending checkpoints within the training range')
    prefix = 'ghost_concat_v1_joint'
    if args.memory_mode == 'persistent_node_state':
        prefix += '_persistent'
    name = prefix + '_train'
    root = ROOT / args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'delivery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    relative = Path(args.output) / 'train' / name / 'checkpoints' / name
    train_manifest = root / 'train' / name / 'manifest.json'
    for iteration in iterations:
        source = ROOT / relative / f'ckpt.iter{iteration}.pth'
        # Models are atomically published and retained; old optimizer states
        # may already be pruned while evaluation is catching up.
        while not source.is_file():
            if train_manifest.exists():
                training = json.loads(train_manifest.read_text())
                if training.get('status') in ('failed', 'completed'):
                    raise RuntimeError(f'Training ended without pending model {iteration}')
            print(f'waiting_checkpoint={iteration}', flush=True)
            time.sleep(30)
        digest = sha(source)
        destination = REMOTE / relative / source.name
        manifest = REMOTE / args.output / 'eval' / f'{prefix}_eval_iter{iteration}' / 'manifest.json'
        code = 'import pathlib,sys; p=pathlib.Path(sys.argv[1]); print(p.read_text() if p.exists() else "{}")'
        result = json.loads(remote_python(code, manifest))
        if result.get('status') != 'completed':
            subprocess.run([sys.executable, 'vlnce_baselines/common/checkpoint_sync.py',
                            '--source', str(source), '--destination', f'{HOST}:{destination.parent}'], check=True)
            while True:
                result = json.loads(remote_python(code, manifest))
                if result.get('status') == 'failed':
                    raise RuntimeError(f'Evaluation failed at {iteration}; copy retained')
                if result.get('status') == 'completed':
                    break
                print(f'waiting_evaluation={iteration}', flush=True)
                time.sleep(30)
        if result.get('checkpoint_sha256') != digest or result.get('validation', {}).get('episodes') != 1839:
            raise RuntimeError('Evaluation digest or full episode count mismatch')
        # Only this run's verified replica is temporary. Never delete originals.
        cleanup = '''import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1])
if p.exists():
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''): h.update(b)
 if h.hexdigest()!=sys.argv[2]: raise RuntimeError('Replica changed; refusing removal')
 p.unlink()
print('verified_replica_released')
'''
        print(remote_python(cleanup, destination, digest).strip(), flush=True)
        save(root / 'delivery.json', dict(iteration=iteration, sha256=digest,
             status='completed' if iteration == iterations[-1] else 'running', original=str(source), requested_iterations=iterations))


if __name__ == '__main__':
    main()
