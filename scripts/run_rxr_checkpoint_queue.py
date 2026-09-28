#!/usr/bin/env python3
"""Finite, newest-first RxR queue shared by two server GPUs and one remote GPU.

The coordinator runs on the training server. Only the active remote checkpoint
is staged; successful results stay on their execution machine. Any lane error
stops that lane and records the failure, without silently retrying a GPU task.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading

REPO = Path('/home/gwl/project/etpr1/ETP-R1')
REMOTE_REPO = Path('/home/a6000/gwl/ETP-R1')
REMOTE = 'a6000@10.10.10.2'
CONTAINER = 'gwl-etpr1-rae'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate(output, episodes):
    output = Path(output)
    files = list(output.glob('results/*/eval_results/stats_ep_ckpt_*_val_unseen_r0_w1.json'))
    summaries = list(output.glob('results/*/eval_results/stats_ckpt_*_val_unseen.json'))
    if len(files) != 1 or len(summaries) != 1:
        raise ValueError('Expected exactly one episode file and one summary')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    rows = json.loads(files[0].read_text(), object_pairs_hook=unique)
    summary = json.loads(summaries[0].read_text())
    dataset = Path('data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide.json.gz')
    with gzip.open(dataset, 'rt') as stream:
        expected = {str(row['episode_id']) for row in json.load(stream)['episodes']}
    if len(expected) != 11006:
        raise ValueError(f'Unexpected full dataset size: {len(expected)}')
    if episodes == -1 and set(rows) != expected:
        raise ValueError(f'Incomplete coverage: {len(rows)}/{len(expected)}')
    if not rows or not set(rows) <= expected or (episodes > 0 and len(rows) < episodes):
        raise ValueError('Invalid short-evaluation coverage')
    for key, value in summary.items():
        values = [float(row[key]) for row in rows.values()]
        if not all(math.isfinite(v) for v in values) or not math.isfinite(float(value)):
            raise ValueError(f'Nonfinite metric: {key}')
        if not math.isclose(sum(values) / len(values), float(value), rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f'Metric does not match episode results: {key}')
    report = dict(validated_at=now(), episodes=len(rows), full=episodes == -1,
                  dataset_sha256=sha256(dataset), metrics=summary)
    write_json(output / 'validation.json', report)
    print(json.dumps(report), flush=True)


def run_queue(args):
    if Path.cwd() != REPO:
        raise ValueError('Coordinator must run in training-server main workspace')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / 'queue.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'queue.json').exists():
        raise ValueError('Existing queue: use a new directory; no implicit restart')
    sources = {}
    for directory in args.checkpoints:
        for path in Path(directory).glob('ckpt.iter*.pth'):
            iteration = int(re.fullmatch(r'ckpt.iter(\d+)\.pth', path.name)[1])
            sources[iteration] = path.resolve()
    if set(sources) != set(range(200, 30001, 200)):
        raise ValueError('Expected all 150 checkpoints, 200..30000 by 200')
    skipped = set(args.skip)
    if not skipped <= sources.keys():
        raise ValueError('Unknown skipped checkpoint')
    pending = sorted(sources.keys() - skipped, reverse=True)
    state = dict(started_at=now(), pid=os.getpid(), status='running',
                 source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 skipped=sorted(skipped), pending=pending, jobs={}, lanes={})
    mutex = threading.Lock()
    remote_root = Path(args.remote_output)

    def save():
        state['updated_at'] = now()
        write_json(root / 'queue.json', state)

    def command(cmd, log):
        print(now(), shlex.join(map(str, cmd)), file=log, flush=True)
        subprocess.run(list(map(str, cmd)), stdout=log, stderr=subprocess.STDOUT, check=True)

    def ssh(cmd, log):
        command(['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30',
                 '-o', 'ServerAliveCountMax=6', REMOTE, shlex.join(map(str, cmd))], log)

    def worker(lane, gpu, envs):
        with open(root / (lane + '.log'), 'a', buffering=1) as log:
            while True:
                with mutex:
                    if not pending:
                        state['lanes'][lane] = 'completed'
                        save()
                        return
                    iteration = pending.pop(0)
                    job = dict(lane=lane, iteration=iteration, started_at=now(), status='running',
                               checkpoint=str(sources[iteration]), environments=envs)
                    state['jobs'][str(iteration)] = job
                    state['lanes'][lane] = iteration
                    save()
                try:
                    source = sources[iteration]
                    digest = sha256(source)
                    with mutex:
                        job['sha256'] = digest
                        save()
                    name = f'{lane}_iter{iteration}'
                    if lane == 'eval':
                        output = remote_root / name
                        staging = remote_root / 'staging'
                        target = staging / source.name
                        # Refuse to contend with the protected container or any GPU work.
                        guard = ('test "$(docker inspect -f "{{.State.Running}}" gwl-etpnav)" != true '
                                 '&& test "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -le 1024 '
                                 '&& test "$(df -Pk /home/a6000/gwl | tail -1 | awk \'{print $4}\')" -gt 20971520')
                        ssh(['bash', '-lc', guard], log)
                        ssh(['mkdir', '-p', staging], log)
                        command(['rsync', '--partial', '--protect-args', str(source),
                                 f'{REMOTE}:{target}.partial'], log)
                        check = f'test "$(sha256sum {shlex.quote(str(target)+".partial")} | cut -d " " -f1)" = {digest} && mv {shlex.quote(str(target)+".partial")} {shlex.quote(str(target))}'
                        ssh(['bash', '-lc', check], log)
                        cmd = ['docker', 'exec', CONTAINER, 'bash',
                               REMOTE_REPO / 'scripts/run_rxr_checkpoint_eval.sh',
                               'eval', '0', target, output, str(envs)]
                        with mutex:
                            job['output'] = str(output)
                            save()
                        ssh(cmd, log)
                        # Small audited summary only; raw episode results remain remote.
                        summary = subprocess.check_output(['ssh', REMOTE, shlex.join(
                            ['cat', str(output / 'validation.json')])], text=True)
                        validation = json.loads(summary)
                        if not validation['full'] or validation['episodes'] != 11006:
                            raise ValueError('Remote full evaluation validation failed')
                        write_json(root / f'iter{iteration}_remote_validation.json', validation)
                        # Only this queue's transferred copy; never a source checkpoint.
                        ssh(['rm', '--', target], log)
                    else:
                        output = root / name
                        if shutil.disk_usage(root).free < 10 * 1024**3:
                            raise RuntimeError('Less than 10GiB remaining for server results')
                        with mutex:
                            job['output'] = str(output)
                            save()
                        command(['bash', REPO / 'scripts/run_rxr_checkpoint_eval.sh',
                                 'server', str(gpu), source, output, str(envs)], log)
                        validation = json.loads((output / 'validation.json').read_text())
                    with mutex:
                        job.update(status='completed', exit_code=0, finished_at=now(), validation=validation)
                        save()
                except Exception as error:
                    with mutex:
                        job.update(status='failed', finished_at=now(), error=str(error),
                                   exit_code=getattr(error, 'returncode', 1))
                        state['lanes'][lane] = 'failed'
                        save()
                    print(now(), repr(error), file=log, flush=True)
                    return
    save()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(worker, 'server0', 0, args.server_envs),
                   pool.submit(worker, 'server1', 1, args.server_envs),
                   pool.submit(worker, 'eval', 0, args.eval_envs)]
        for future in futures:
            future.result()
    state['status'] = 'completed' if not pending and all(
        job['status'] == 'completed' for job in state['jobs'].values()) else 'failed'
    state['finished_at'] = now()
    save()
    if state['status'] != 'completed':
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    check = sub.add_parser('validate')
    check.add_argument('--output', required=True)
    check.add_argument('--episodes', type=int, default=-1)
    queue = sub.add_parser('queue')
    queue.add_argument('--checkpoints', nargs='+', required=True)
    queue.add_argument('--output', required=True)
    queue.add_argument('--remote-output', required=True)
    queue.add_argument('--skip', type=int, nargs='*', default=[])
    queue.add_argument('--server-envs', type=int, default=8)
    queue.add_argument('--eval-envs', type=int, default=4)
    args = parser.parse_args()
    if args.action == 'validate':
        validate(args.output, args.episodes)
    else:
        run_queue(args)


if __name__ == '__main__':
    main()
