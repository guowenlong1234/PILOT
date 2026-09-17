#!/usr/bin/env python3
"""Finite server-side transfer/evaluation of atomically published E24 heads.

Only heads and small result JSONs cross the private link. An ambiguous SSH or
worker failure is terminal: rerunning never silently launches the same job.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time
import traceback

from rgb_only_optimization import ROOT, now, save, sha

HOST = 'a6000@10.10.10.2'
EVAL_ROOT = '/home/a6000/gwl/ETP-R1-stage2-e24'
DEV = 'data/logs/stage2_e24/formal_dev/episodes'
DEV_MANIFEST_SHA = '25d9666e89e482186b9d6f7436dcf5c9934f82cd9265fa0658c0f1ec632ba93d'
HEAD_PATTERN = re.compile(r'head_step_(\d+)\.pt$')


class Watcher:
    def __init__(self, args):
        self.args = args
        self.train = Path(args.train_dir).resolve()
        self.out = self.train / 'offline_eval'
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + args.max_hours * 3600
        self.remote_run = str(Path(EVAL_ROOT) / args.eval_run_dir)
        self.status = dict(status='starting', host=socket.gethostname(), pid=os.getpid(),
                           cwd=str(ROOT), train_dir=str(self.train), started_at=now(),
                           eval_run_dir=self.remote_run, dev=DEV, args=vars(args))

    def event(self, **event):
        event['time'] = now()
        with (self.out / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)

    def run(self, command, timeout=120):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Finite watcher deadline exceeded')
        self.event(event='command', command=command)
        started = time.monotonic()
        try:
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                    timeout=min(timeout, remaining))
        except subprocess.TimeoutExpired:
            self.event(event='command_timeout', command=command,
                       remote_state='unknown; do not relaunch automatically')
            raise
        self.event(event='command_exit', exit_code=result.returncode,
                   duration_seconds=round(time.monotonic()-started, 3),
                   stderr=result.stderr[-6000:])
        if result.returncode:
            raise RuntimeError(f'Command failed ({result.returncode}): {shlex.join(command)}\n'
                               f'{result.stderr[-4000:]}\n{result.stdout[-4000:]}')
        return result.stdout

    def remote(self, arguments, timeout=120):
        cmd = 'cd ' + shlex.quote(EVAL_ROOT) + ' && ' + shlex.join(arguments)
        return self.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                         '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
                         HOST, cmd], timeout)

    def remote_python(self, code, *arguments):
        return self.remote(['python3', '-c', code, *map(str, arguments)])

    def identity(self):
        code = ('import json,socket,getpass,os; '
                'print(json.dumps(dict(host=socket.gethostname(),user=getpass.getuser(),cwd=os.getcwd())))')
        identity = json.loads(self.remote_python(code))
        if identity['user'] != 'a6000' or identity['cwd'] != EVAL_ROOT:
            raise RuntimeError(f'Unexpected evaluation identity: {identity}')
        self.status['eval_identity'] = identity
        self.status['source_commit'] = self.run(['git', 'rev-parse', 'HEAD']).strip()
        # Pin the evaluation side to exactly the same scripts and data location.
        remote_commit = self.remote(['git', 'rev-parse', 'HEAD']).strip()
        if remote_commit != self.status['source_commit']:
            raise RuntimeError('Training/evaluation source commits differ')
        self.remote_python('import pathlib,sys,hashlib; '
                           'm=pathlib.Path(sys.argv[1])/"dataset_manifest.json"; '
                           'assert hashlib.sha256(m.read_bytes()).hexdigest()==sys.argv[3], "Dev manifest changed"; '
                           'p=pathlib.Path(sys.argv[2]); p.mkdir(parents=True,exist_ok=True)',
                           DEV, self.remote_run, DEV_MANIFEST_SHA)
        save(self.out / 'watcher.json', self.status)

    def evaluate(self, head, step):
        record_path = self.out / f'step_{step:06d}.json'
        digest = sha(head)
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record['head_sha256'] != digest:
                raise RuntimeError(f'Previously observed head changed: {head}')
            if record['status'] == 'completed':
                report = self.out / record['report_file']
                if not report.exists() or sha(report) != record['report_sha256']:
                    raise RuntimeError(f'Completed report changed/missing: {report}')
                return
            raise RuntimeError(f'Unfinished prior evaluation at step {step}; inspect remote worker '
                               f'{record.get("job_dir")}; automatic relaunch is disabled')
        remote_head = self.remote_run + '/heads/' + head.name
        job = self.remote_run + f'/jobs/step_{step:06d}'
        report = self.remote_run + f'/results/step_{step:06d}.json'
        record = dict(status='transferring', step=step, head=str(head), head_sha256=digest,
                      remote_head=remote_head, job_dir=job, remote_report=report, started_at=now())
        save(record_path, record)
        try:
            # Refuse an existing worker attempt even if the local record was lost.
            self.remote_python('from pathlib import Path; import sys; '
                               'h,j,r=map(Path,sys.argv[1:]); '
                               'assert not j.exists(), "Existing job: manual reconciliation required"; '
                               'assert not r.exists(), "Existing result: manual reconciliation required"; '
                               'h.parent.mkdir(parents=True,exist_ok=True); '
                               'assert not h.exists(), "Existing head: manual reconciliation required"',
                               remote_head, job, report)
            self.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                      str(head), HOST + ':' + remote_head + '.partial'], timeout=600)
            # Compute SHA independently at destination before atomic publication.
            self.remote_python('from pathlib import Path; import hashlib,sys; '
                               'p=Path(sys.argv[1]); d=hashlib.sha256(p.read_bytes()).hexdigest(); '
                               'assert d==sys.argv[2], (d,sys.argv[2]); '
                               'q=Path(sys.argv[3]); assert not q.exists(); p.rename(q); print(d)',
                               remote_head + '.partial', digest, remote_head)
            record.update(status='evaluating', transferred_at=now())
            save(record_path, record)
            command = ['python3', 'scripts/stage2_offline_worker.py', '--machine', 'eval',
                       '--job-dir', job, '--', 'scripts/evaluate_stage2_e24.py',
                       '--data-roots', DEV, '--checkpoints', remote_head, '--report', report,
                       '--device', 'cuda', '--amp', self.args.amp,
                       '--num-workers', str(self.args.num_workers),
                       '--batch-size', str(self.args.batch_size)]
            record['worker_command'] = command
            save(record_path, record)
            self.remote(command, timeout=self.args.eval_timeout_hours * 3600)
            worker = json.loads(self.remote(['cat', job + '/worker.json']))
            if worker.get('status') != 'completed' or worker.get('exit_code') != 0:
                raise RuntimeError(f'Worker did not complete successfully: {worker}')
            raw = self.remote(['cat', report])
            metrics = json.loads(raw)
            validate_report(metrics, step)
            row = metrics['results'][0]
            if row['checkpoint_sha256'] != digest or row['checkpoint'] != remote_head:
                raise ValueError('Result checkpoint path/SHA differs from transferred head')
            local_report = self.out / f'metrics_step_{step:06d}.json'
            save(local_report, metrics)
            save(self.out / f'worker_step_{step:06d}.json', worker)
            record.update(status='completed', exit_code=0, completed_at=now(),
                          report_file=local_report.name, report_sha256=sha(local_report))
            save(record_path, record)
            self.curve()
        except Exception as exc:
            record.update(status='failed', failed_at=now(), error=str(exc))
            save(record_path, record)
            raise

    def curve(self):
        rows = []
        for path in sorted(self.out.glob('step_*.json')):
            record = json.loads(path.read_text())
            if record['status'] != 'completed':
                continue
            report = json.loads((self.out / record['report_file']).read_text())
            metrics = validate_report(report, record['step'])
            rows.append(dict(step=record['step'], head=record['head'],
                             head_sha256=record['head_sha256'], report_file=record['report_file'],
                             **metrics))
        rows.sort(key=lambda row: row['step'])
        save(self.out / 'curve.json', dict(gamma=1.0, points=rows))
        ranked = sorted(rows, key=lambda row: (-row['net_corrections'], row['harms'],
                                              row['cross_entropy'], row['step']))
        save(self.out / 'selection.json', dict(
            status='provisional' if self.status['status'] != 'completed' else 'completed',
            gamma=1.0, rule='net_corrections desc, harms asc, cross_entropy asc, step asc',
            evaluated_points=len(rows), selected=ranked[:2],
            positive_offline_gain=bool(ranked and ranked[0]['net_corrections'] > 0),
            interpretation='Fixed val_unseen development set; not independent test or navigation success'))

    def watch(self):
        self.identity()
        self.status['status'] = 'running'
        save(self.out / 'watcher.json', self.status)
        while True:
            if time.monotonic() >= self.deadline:
                raise TimeoutError('Finite watcher deadline exceeded')
            # Snapshot status before scanning: completed is written only after all heads publish.
            status_path = self.train / 'status.json'
            train_status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if train_status.get('status') in ('failed', 'interrupted'):
                raise RuntimeError(f'Training did not complete: {train_status}')
            heads = sorted((int(HEAD_PATTERN.fullmatch(p.name).group(1)), p)
                           for p in self.train.glob('head_step_*.pt')
                           if HEAD_PATTERN.fullmatch(p.name))
            for step, head in heads:
                self.evaluate(head, step)
            if train_status.get('status') == 'completed':
                if not heads or heads[-1][0] != int(train_status['step']):
                    raise RuntimeError('Training completed without a matching final head')
                expected = set(range(self.args.save_every, int(train_status['step']) + 1,
                                     self.args.save_every)) | {int(train_status['step'])}
                if not expected.issubset({step for step, _ in heads}):
                    raise RuntimeError('Missing periodic exported heads in completed training')
                self.status.update(status='completed', completed_at=now(),
                                   train_status=train_status, evaluated_heads=len(heads), exit_code=0)
                self.curve()
                save(self.out / 'watcher.json', self.status)
                return 0
            self.status.update(last_poll_at=now(), train_status=train_status,
                               observed_heads=len(heads))
            save(self.out / 'watcher.json', self.status)
            time.sleep(min(self.args.poll_seconds, max(0, self.deadline-time.monotonic())))


def validate_report(report, step):
    """Validate the evaluator's full-data, fixed-g=1 contract before selection."""
    if report.get('status') != 'complete' or len(report.get('results', [])) != 1:
        raise ValueError('Expected exactly one complete fixed-g=1 result')
    dataset = report.get('dataset', [])
    if len(dataset) != 1 or dataset[0]['manifest_sha256'] != DEV_MANIFEST_SHA:
        raise ValueError('Result does not cover the fixed development manifest')
    row = report['results'][0]
    if int(row['global_step']) != step or float(row['gain']) != 1.0:
        raise ValueError('Evaluation report step/gain does not match requested head')
    if int(row['primary_rows']) <= 0:
        raise ValueError('No valid development rows')
    metrics = dict(net_corrections=row['net_fixes'], harms=row['harms'],
                   cross_entropy=row['classification_loss'])
    for key, value in metrics.items():
        if not math.isfinite(float(value)):
            raise ValueError(f'Non-finite selection metric: {key}')
    return metrics



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--eval-run-dir', required=True,
                        help='New relative experiment directory under evaluation worktree')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--amp', choices=('bf16', 'fp16', 'none'), default='bf16')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-hours', type=float, default=72)
    parser.add_argument('--eval-timeout-hours', type=float, default=2)
    args = parser.parse_args()
    relative = Path(args.eval_run_dir)
    if relative.is_absolute() or '..' in relative.parts or len(relative.parts) < 2:
        parser.error('--eval-run-dir must be a new relative subdirectory without ..')
    if args.num_workers < 0:
        parser.error('--num-workers cannot be negative')
    if min(args.batch_size, args.save_every, args.poll_seconds,
           args.max_hours, args.eval_timeout_hours) <= 0:
        parser.error('All size/interval/timeout arguments must be positive')
    if str(ROOT) != '/home/gwl/project/etpr1/ETP-R1-stage2-e24':
        parser.error('Run this watcher in the training-machine stage2 worktree')
    watcher = Watcher(args)
    with (watcher.out / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return watcher.watch()
        except Exception as exc:
            watcher.status.update(status='failed', exit_code=1, error=str(exc), failed_at=now())
            save(watcher.out / 'watcher.json', watcher.status)
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
