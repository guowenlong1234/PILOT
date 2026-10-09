#!/usr/bin/env python3
"""Logged-state threshold sweep; never treats replay as new navigation metrics."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

GRID=(0.,.1,.2,.3,.5,.75,1.,1.25,1.5,1.75,2.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit=json.loads((a.archive/'score_trace_audit.json').read_text())
    trace=a.archive/'online/decisions.jsonl';raw=trace.read_bytes()
    if audit['status']!='passed' or hashlib.sha256(raw).hexdigest()!=audit['trace_sha256']:
        raise ValueError('archive audit/hash mismatch')
    metrics=json.loads(next((a.archive/'results').rglob('stats_ep_*.json')).read_text())
    rows=[]
    for x in map(json.loads,raw.splitlines()):
        if x['base_action']==0 or x['forced_stop'] or len(x['ghost_indices'])<2:continue
        ranked=sorted(x['ghost_indices'],key=lambda j:(-x['logits'][j],j))
        gap=x['logits'][ranked[0]]-x['logits'][ranked[1]]
        flip=x['action']!=x['base_action']
        rows.append(dict(gap=gap,flip=flip,queries=x['future_valid_slots'],episode=x['episode'],
                         scene=x['scene'],guard=x['certificate']['numerical_guard'],
                         regret=x['refined_logits'][x['action']]-x['refined_logits'][x['base_action']]))
    def sweep(data):
        total_queries=sum(x['queries'] for x in data);flips=sum(x['flip'] for x in data)
        bounded_queries=sum(x['queries'] for x in data if x['gap']<=2+x['guard'])
        out=[]
        for t in GRID:
            kept=[x for x in data if x['gap']<=t+x['guard']]
            missed=[x for x in data if x['flip'] and x['gap']>t+x['guard']]
            affected={x['episode'] for x in missed};q=sum(x['queries'] for x in kept)
            out.append(dict(threshold=t,retained_move_rows=len(kept),queries=q,
                query_reduction_vs_full=1-q/max(1,total_queries),
                query_reduction_vs_certificate=1-q/max(1,bounded_queries),
                retained_flips=flips-len(missed),missed_flips=len(missed),
                flip_retention=(flips-len(missed))/max(1,flips),affected_episodes=len(affected),
                affected_full_success_episodes=sum(metrics[e]['success'] for e in affected),
                max_logged_score_regret=max((x['regret'] for x in missed),default=0.)))
        return out
    table=sweep(rows);per_scene={};leave_one=[]
    for scene in sorted({x['scene'] for x in rows}):
        held=[x for x in rows if x['scene']==scene];train=[x for x in rows if x['scene']!=scene]
        selected=next(x['threshold'] for x in sweep(train) if x['missed_flips']==0)
        per_scene[scene]=sweep(held)
        leave_one.append(dict(scene=scene,selected_threshold=selected,
            held_out=next(x for x in per_scene[scene] if x['threshold']==selected)))
    result=dict(status='passed',archive=str(a.archive),trace_sha256=audit['trace_sha256'],
        episodes=audit['episodes'],scenes=audit['scenes'],moves=len(rows),
        actual_flips=sum(x['flip'] for x in rows),
        max_flip_gap=max(x['gap'] for x in rows if x['flip']),
        recommended_threshold=next(x['threshold'] for x in table if x['missed_flips']==0),
        selection_rule='Smallest coarse-grid gap threshold retaining every observed action flip.',
        sweep=table,per_scene=per_scene,leave_one_scene_out=leave_one,
        limitation='Exploratory val_unseen development analysis, not independent test. '
        'Queries are logged-state estimates; changed actions invalidate downstream replay. '
        'Affected successful episodes are not causal harms; no counterfactual SR/SPL is estimated.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('per_scene','leave_one_scene_out')},indent=2))


if __name__=='__main__':main()
