#!/usr/bin/env python3
"""Validate complete per-decision scores and count pruning opportunities offline."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path


def report(root, expected_episodes=None):
    provenance=json.loads((root/'provenance.json').read_text())
    files=list((root/'results').rglob('stats_ep_*.json'))
    if len(files)!=1:raise ValueError('expected one completed episode result')
    metrics=json.loads(files[0].read_text())
    expected=set(map(str,provenance['episode_ids']))
    if set(metrics)!=expected or (expected_episodes is not None and len(expected)!=expected_episodes):
        raise ValueError('episode coverage differs')
    status=json.loads((root/'status.json').read_text())
    if status.get('status')!='completed' or status.get('exit_code')!=0:raise ValueError('navigation incomplete')
    for name in ('freeze_report.json','world_freeze.json'):
        frozen=json.loads((root/'online'/name).read_text())
        if not frozen.get('comparison',frozen)['exact_match']:raise ValueError('weights changed')
    counts=Counter();steps=defaultdict(list);scenes=set();per_scene=defaultdict(Counter)
    trace=root/'online/decisions.jsonl';digest=hashlib.sha256()
    with trace.open('rb') as stream:
        for line in stream:
            digest.update(line);r=json.loads(line)
            if r['schema']!='stage2-decision-scores-v2':raise ValueError('incomplete trace schema')
            ep=str(r['episode']);step=int(r['step'])
            if ep not in expected:raise ValueError('unexpected episode')
            steps[ep].append(step);scenes.add(r['scene'])
            ids=r['global_vp_ids'];s=r['logits'];d=r['delta'];z=r['refined_logits'];g=r['ghost_indices']
            if not len(ids)==len(s)==len(d)==len(z):raise ValueError('score/identity shape mismatch')
            if g!=[i for i,v in enumerate(ids) if isinstance(v,str) and v.startswith('g')]:
                raise ValueError('candidate identities differ')
            if any(not math.isfinite(s[i]) for i in g):raise ValueError('nonfinite candidate scores')
            for base,delta,refined in zip(s,d,z):
                if not math.isfinite(delta) or abs(delta)>1:raise ValueError('invalid residual')
                if math.isfinite(base):
                    if not math.isclose(base+delta,refined,rel_tol=2e-7,abs_tol=2e-7):
                        raise ValueError('refined score differs from float32 sum')
                elif base!=-math.inf or refined!=base:raise ValueError('invalid masked score')
            base=max(range(len(s)),key=lambda i:(s[i],-i))
            if base!=r['base_action']:raise ValueError('base action differs')
            chosen=base if base==0 or r['forced_stop'] else max(g,key=lambda i:(z[i],-i))
            if chosen!=r['action'] or r['executed_action']!=(0 if r['forced_stop'] else chosen):
                raise ValueError('recorded action differs')
            top=sorted(g,key=lambda i:(-s[i],i))[:5]
            if top!=r['topk_global_indices']:raise ValueError('TopK differs')
            if len(top)!=len(r['future_valid_mask']) or len(top)!=len(r['invalid_reason']):
                raise ValueError('future metadata differs')
            if sum(r['future_valid_mask'])!=r['future_valid_slots']:raise ValueError('future count differs')
            if r['skip_reason'] is not None or provenance['bounded_skip']:
                raise ValueError('full-score archive must have pruning disabled')
            counts['decisions']+=1;counts['candidate_scores']+=len(g)
            counts['action_flips']+=int(chosen!=base)
            counts['future_valid_slots']+=r['future_valid_slots']
            reason=r['certificate']['skip_reason'] or 'needs_lookahead'
            counts[reason]+=1;per_scene[r['scene']][reason]+=1
            if not r['forced_stop'] and base!=0:
                counts['ordinary_move']+=1
                if len(g)>1:
                    margin=s[top[0]]-s[top[1]]
                    bucket=next((str(t) for t in (.5,1,2,3,5) if margin<=t),'over5')
                    counts['base_margin_le_'+bucket]+=1
                if reason=='bounded_invariant':
                    m=s[base]-1-max(s[j]+(1 if j in top else 0) for j in g if j!=base)
                    if not m>r['certificate']['numerical_guard']:raise ValueError('invalid certificate')
                    if chosen!=base:raise ValueError('certified winner changed')
    if set(steps)!=expected:raise ValueError('missing episode traces')
    for ep,values in steps.items():
        if values!=list(range(len(values))) or len(values)!=int(metrics[ep]['high_level_step'])+1:
            raise ValueError('missing/duplicate/nonsequential decision')
    online=json.loads((root/'online/online_summary.json').read_text())
    if counts['decisions']!=online['counts']['rows']:raise ValueError('decision totals differ')
    return dict(status='passed',episodes=len(expected),scenes=len(scenes),counts=dict(counts),
        per_scene=dict(per_scene),trace=str(trace),trace_sha256=digest.hexdigest(),
        metrics={k:sum(v[k] for v in metrics.values())/len(metrics) for k in next(iter(metrics.values()))},
        limitation='Counts apply to logged full-lookahead trajectories; another policy can visit different states.')


def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',type=Path)
    p.add_argument('--expected-episodes',type=int);a=p.parse_args()
    result=report(a.experiment,a.expected_episodes)
    (a.experiment/'score_trace_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
