#!/usr/bin/env python3
"""Audit paired real progressive smoke runs without importing GPU libraries."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def records(root):
    rows=[json.loads(line) for line in (root/'online/decisions.jsonl').read_text().splitlines()]
    keyed={(r['scene'],r['episode'],r['step']):r for r in rows}
    if len(keyed)!=len(rows): raise ValueError('duplicate decision identity')
    if read(root/'status.json').get('exit_code')!=0: raise ValueError('navigation did not exit successfully')
    for name in ('freeze_report.json','world_freeze.json'):
        freeze=read(root/'online'/name)
        if not freeze.get('comparison',freeze)['exact_match']: raise ValueError('frozen model changed')
    return keyed


def audit(root):
    root=Path(root)
    full=records(root/'nav_none'); pruned=records(root/'nav_certified')
    if full.keys()!=pruned.keys(): raise ValueError('paired trajectories have different decisions')
    p=read(root/'nav_none/provenance.json'); q=read(root/'nav_certified/provenance.json')
    for key in ('head_sha256','stage1_sha256','assets','prediction_contract','feature_contract','max_future_depth'):
        if p[key]!=q[key]: raise ValueError('paired deployment contract differs: '+key)
    checked=0
    for key,short in pruned.items():
        expanded=full[key]
        for field in ('executed_action','ghost_ids','topk_base_indices'):
            if short[field]!=expanded[field]: raise ValueError('paired action or candidate mapping differs: '+str(key))
        if short['stop_reason']!='certified': continue
        scores=short['current_scores']; remaining=short['remaining_budget'][-1]
        winner=max(range(len(scores)),key=scores.__getitem__)
        lower=scores[winner]-remaining[winner]
        upper=max((s+r for i,(s,r) in enumerate(zip(scores,remaining)) if i!=winner),default=-math.inf)
        if not lower>upper+short['numerical_guard']: raise ValueError('invalid recorded stopping certificate')
        if winner!=short['final_winner'] or winner!=expanded['final_winner']:
            raise ValueError('certified greedy winner differs from full expansion')
        if len(short['per_depth_world_model_queries'])!=short['executed_depth']:
            raise ValueError('query accounting differs from executed prefix')
        checked+=1
    a=read(root/'nav_none/online/episode_diagnostics.json')
    b=read(root/'nav_certified/online/episode_diagnostics.json')
    metrics_equal=a==b
    result=dict(status='passed',decisions=len(full),episodes=len(a),actions_identical=True,
        episode_metrics_identical=metrics_equal,certificates_recomputed=checked,
        head_sha256=p['head_sha256'],navigation_quality_claim=False,runs={})
    for name,rows in (('none',full),('certified',pruned)):
        result['runs'][name]=dict(
            stop_reasons=dict(Counter(r['stop_reason'] for r in rows.values())),
            depths=dict(Counter(r['executed_depth'] for r in rows.values())),
            world_model_queries=sum(sum(r['per_depth_world_model_queries']) for r in rows.values()),
            stage2_seconds=sum(r['total_stage2_seconds'] for r in rows.values()),
            navigation_seconds=read(root/('nav_'+name)/'status.json')['elapsed_seconds'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--output',required=True)
    args=p.parse_args();result=audit(args.root);path=Path(args.output)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__': main()
