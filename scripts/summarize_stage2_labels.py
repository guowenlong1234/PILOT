#!/usr/bin/env python3
"""Audit supervision and predicted-future coverage of validated episode shards."""
import argparse
import json
from collections import Counter
from pathlib import Path
from vlnce_baselines.nwm.active_lookahead.stage2_data import load, atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    total=Counter(); reasons=Counter(); scenes={}; seen=set()
    for root in map(Path,a.roots):
        m=json.loads((root/'dataset_manifest.json').read_text())
        if m['status']!='validated':raise ValueError('requires validated data')
        part=Counter()
        for entry in m['entries']:
            if entry['episode_id'] in seen:raise ValueError('duplicate episode')
            seen.add(entry['episode_id']);ep=load(root/entry['file'])
            scene=scenes.setdefault(ep['scene_id'],Counter());scene['episodes']+=1
            for r in ep['rows']:
                values=Counter(rows=1,future_slots=len(r['future_valid_mask']),future_valid=int(r['future_valid_mask'].sum()))
                reasons.update(r['invalid_reason'])
                for key in ('base_stop','teacher_stop','forced_stop'):
                    values[key]=int(r.get(key,False))
                values['teacher_invalid']=int(not r['teacher_valid'])
                eligible=r['teacher_valid'] and not r['base_stop'] and not r['teacher_stop'] and not r['no_vp_left']
                if eligible:
                    assert 0<=r['teacher_base_index']<len(r['ghost_ids'])
                    values['move_teacher_rows']=1
                    values['with_any_future']=int(bool(r['future_valid_mask'].any()))
                    rank=r['teacher_rank_in_topk']
                    values['teacher_in_top5']=int(rank>=0)
                    values['teacher_with_future']=int(rank>=0 and bool(r['future_valid_mask'][rank]))
                    correct=int(r['base_logits'].argmax())==r['teacher_base_index']
                    values['base_correct']=int(correct)
                    values['base_wrong_with_teacher_future']=int(not correct and values['teacher_with_future'])
                part.update(values);scene.update(values)
        for name,key in [('rows','rows'),('future_slots','future_slots'),('future_valid','future_valid'),('with_any_future','trainable_rows')]:
            if part[name]!=m[key]:raise ValueError(f'coverage mismatch: {name}')
        total.update(part)
    n=total['move_teacher_rows']
    report=dict(episodes=len(seen),counts=dict(total),invalid_reasons=dict(reasons),
        definitions={'move_teacher_rows':'valid teacher, both base and teacher MOVE, executable ghosts remain',
                     'with_any_future':'move_teacher_rows with at least one valid predicted future (trainable_rows)',
                     'teacher_with_future':'teacher is in Top5 and its future is valid'},
        teacher_top5_rate=total['teacher_in_top5']/n if n else None,
        teacher_future_rate=total['teacher_with_future']/n if n else None,
        scenes={k:dict(v) for k,v in sorted(scenes.items())},training_started=False)
    atomic_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'}))
if __name__=='__main__':main()
