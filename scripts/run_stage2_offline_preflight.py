#!/usr/bin/env python3
"""Finite real-data GPU gate: continuous 20 versus 10 + new-process resume."""
import argparse
from pathlib import Path
import subprocess
import sys

from rgb_only_optimization import ROOT, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('preflight requires an empty directory')
    base = ['scripts/train_stage2_e24.py', *args.roots, '--batch-size', '32',
            '--max-steps', '20', '--save-every', '10', '--log-every', '1',
            '--precision', args.precision]
    jobs = [
        ('continuous', base + ['--output', str(out/'continuous')]),
        ('interrupted', base + ['--output', str(out/'resumed'), '--stop-after-steps', '10']),
        ('resumed', base + ['--output', str(out/'resumed'), '--resume', str(out/'resumed/state_step_000010.pt')]),
        ('comparison', ['scripts/check_stage2_resume.py', str(out/'continuous/state_step_000020.pt'),
                        str(out/'resumed/state_step_000020.pt'), '--output', str(out/'resume_comparison.json')]),
    ]
    for name, command in jobs:
        full = [sys.executable, 'scripts/stage2_offline_worker.py', '--machine', 'server',
                '--gpu', args.gpu, '--job-dir', str(out/'jobs'/name), '--', *command]
        save(out/'pipeline.json', dict(status='running', stage=name, command=full))
        result = subprocess.run(full, cwd=ROOT)
        if result.returncode:
            save(out/'pipeline.json', dict(status='failed', stage=name, exit_code=result.returncode))
            return result.returncode
    save(out/'pipeline.json', dict(status='passed', precision=args.precision, exit_code=0,
                                  continuous_steps=20, interrupted_at=10, resumed_to=20))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
