#!/usr/bin/env python3
"""Audit completed pruning traces, deployment contracts, and numerical parity."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def traces(path):
    rows={}
    for line in path.read_text().splitlines():
        x=json.loads(line);key=(x['episode'],x['step'])
        if key in rows:raise ValueError('duplicate decision')
        rows[key]=x
    if not rows:raise ValueError('empty trace')
    return rows


def numerical_parity(left,right):
    if left.keys()!=right.keys():raise ValueError('decision identities differ')
    base_error=delta_error=0.
    for key,x in left.items():
        y=right[key]
        if any(x[f]!=y[f] for f in ('action','base_action','forced_stop','ghost_indices')):
            raise ValueError('action or candidate mismatch')
        if len(x['logits'])!=len(y['logits']):raise ValueError('logit shapes differ')
        for a,b in zip(x['logits'],y['logits']):
            if math.isfinite(a) and math.isfinite(b):base_error=max(base_error,abs(a-b))
            elif a!=b:raise ValueError('nonfinite logit masks differ')
        if y['skip_reason'] is None:
            if len(x['delta'])!=len(y['delta']):raise ValueError('residual shapes differ')
            delta_error=max(delta_error,max(abs(a-b) for a,b in zip(x['delta'],y['delta'])))
    return dict(decisions=len(left),base_logit_max_abs_difference=base_error,
                retained_residual_max_abs_difference=delta_error,action_mismatches=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path);a=p.parse_args()
    root=a.experiment
    if read(root/'pipeline.json')['status']!='completed':raise ValueError('benchmark incomplete')
    names=('bench_off_1','bench_on_1','bench_on_2','bench_off_2')
    rows={name:traces(root/name/'online/decisions.jsonl') for name in names}
    reference=read(root/names[0]/'provenance.json');reference.pop('bounded_skip')
    for name in names:
        actual=read(root/name/'provenance.json')
        if actual.pop('bounded_skip')!=('_on_' in name):raise ValueError('incorrect pruning flag')
        if actual!=reference:raise ValueError('deployment provenance differs')
        if read(root/name/'status.json')['exit_code']!=0:raise ValueError('run failed')
    certificate_counts={}
    for name in ('bench_on_1','bench_on_2'):
        count=0
        for x in rows[name].values():
            reason=x['skip_reason']
            if reason is not None:
                if any(x['delta']) or x['future_valid_slots']!=0:raise ValueError('pruned row computed a future or residual')
                if x['action']!=x['base_action']:raise ValueError('pruned row changed winner')
            if reason!='bounded_invariant':continue
            ghosts=x['ghost_indices'];s=x['logits'];winner=x['base_action']
            ranked=sorted(ghosts,key=lambda j:(-s[j],j))[:5]
            if winner!=ranked[0]:raise ValueError('certificate winner differs')
            # Independent reconstruction for this fixed ±1 deployment.
            margin=s[winner]-1-max(s[j]+(1 if j in ranked else 0) for j in ghosts if j!=winner)
            proof=x['certificate']
            if not margin>proof['numerical_guard']:raise ValueError('unsafe certificate')
            if abs(margin-proof['margin'])>1e-12:raise ValueError('recorded margin differs')
            count+=1
        summary=read(root/name/'online/online_summary.json')
        if count!=summary['counts']['skipped_bounded_invariant']:
            raise ValueError('certificate counter differs from traces')
        certificate_counts[name]=count
    diagnostics=read(root/'bench_on_1/online/episode_diagnostics.json')
    report=dict(status='passed',provenance_equal_except_pruning=True,
        scenes=len({x['scene'] for x in diagnostics.values()}),episodes=len(diagnostics),
        certificate_counts=certificate_counts,
        comparisons={name:numerical_parity(rows['bench_off_1'],rows[name]) for name in names[1:]},
        on_repeat=numerical_parity(rows['bench_on_1'],rows['bench_on_2']))
    path=root/'numeric_audit.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
