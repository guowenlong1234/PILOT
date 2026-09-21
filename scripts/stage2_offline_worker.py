#!/usr/bin/env python3
"""Run one finite offline job with the existing project GPU lock and runtime."""
import argparse
import os
from pathlib import Path
import shlex
import socket
import subprocess
import traceback

from rgb_only_optimization import ROOT, VERSIONS, now, output, resources, runtime, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', required=True, choices=('server', 'eval'))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('a Python script and arguments are required')
    job = Path(args.job_dir).resolve()
    job.mkdir(parents=True, exist_ok=True)
    status = dict(status='starting', host=socket.gethostname(), cwd=str(ROOT),
                  pid=os.getpid(), started_at=now(), source_commit=output(['git', 'rev-parse', 'HEAD']),
                  command=runtime(args.machine, command, args.gpu))
    if (job / 'worker.json').exists():
        raise FileExistsError('Use a new job directory for each attempt')
    save(job / 'worker.json', status)
    try:
        with resources(args.machine, args.gpu), open(job / 'run.log', 'x', buffering=1) as log:
            log.write(shlex.join(status['command']) + '\n')
            version = subprocess.run(runtime(args.machine, ['-c', VERSIONS], args.gpu),
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if version.returncode:
                raise RuntimeError('Runtime import/version check failed')
            child = subprocess.Popen(status['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status.update(status='running', child_pid=child.pid)
            save(job / 'worker.json', status)
            code = child.wait()
            status.update(status='completed' if code == 0 else 'failed', exit_code=code, ended_at=now())
            save(job / 'worker.json', status)
            return code
    except Exception as exc:
        status.update(status='failed', error=str(exc), ended_at=now())
        save(job / 'worker.json', status)
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
