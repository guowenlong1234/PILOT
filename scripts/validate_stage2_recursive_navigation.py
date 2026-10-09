#!/usr/bin/env python3
"""Finite evaluation-host integration check; keeps all runs in a new directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from rgb_only_optimization import ROOT, now, save
from benchmark_stage2_bounded_skip import compare, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--h1-reference', required=True)
    args = parser.parse_args()
    if str(ROOT) != '/home/a6000/gwl/ETP-R1':
        raise ValueError('run on evaluation host in the main workspace')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    results = {}

    def run(name, command):
        save(out/'pipeline.json', dict(status='running', stage=name, command=command, updated_at=now()))
        with (out/(name+'.log')).open('x') as log:
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        results[name] = dict(command=command, exit_code=code)
        save(out/'commands.json', results)
        if code:
            raise RuntimeError(f'{name} exited {code}')

    try:
        for depth in (1, 2, 3):
            name = f'online_h{depth}'
            command = [sys.executable, 'scripts/stage2_e24_job.py', 'online',
                       '--machine', 'eval', '--environments', '4', '--split', 'val_unseen',
                       '--base-step', '9200', '--deployment-mode', 'native_9200',
                       '--checkpoint', args.checkpoint, '--head', args.head, '--gain', '1',
                       '--episodes', '16' if depth == 1 else '4', '--output', str(out/name),
                       '--bounded-skip', 'off', '--margin-threshold', '-1', '--trace', '--profile',
                       '--lookahead-horizon-steps', str(depth)]
            if depth > 1:
                command.append('--allow-rollout-depth-transfer')
            run(name, command)
            for filename in ('freeze_report.json', 'world_freeze.json'):
                frozen = json.loads((out/name/'online'/filename).read_text())
                if not frozen.get('comparison', frozen)['exact_match']:
                    raise ValueError('frozen weights changed')
            if depth == 1:
                parity = compare(Path(args.h1_reference), out/name)
                save(out/'h1_parity.json', parity)
                if not parity['passed']:
                    raise ValueError('h1 actions/metrics differ from pre-merge reference')
            else:
                rows = decisions(out/name)
                realized = [int(d) for row in rows.values() for d in row['realized_depths']]
                if not realized or max(realized) != depth:
                    raise ValueError(f'real navigation did not reach depth {depth}')
                save(out/f'h{depth}_coverage.json', dict(decisions=len(rows),
                     realized_depth_counts={str(d): realized.count(d) for d in range(depth+1)},
                     limitation='one-step trained head, explicitly permitted cross-depth inference'))
        collect = [sys.executable, 'scripts/run_stage2_collection.py', 'collect',
                   '--machine', 'eval', '--environments', '4', '--split', 'train',
                   '--base-step', '9200', '--checkpoint', args.checkpoint,
                   '--episodes', '4', '--output', str(out/'collect_h3'),
                   '--lookahead-horizon-steps', '3', '--compact-storage']
        run('collect_h3', collect)
        save(out/'pipeline.json', dict(status='completed', exit_code=0, updated_at=now(),
             limitation='integration smoke, not a multi-step navigation quality benchmark'))
    except Exception as error:
        save(out/'pipeline.json', dict(status='failed', error=str(error), updated_at=now()))
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
