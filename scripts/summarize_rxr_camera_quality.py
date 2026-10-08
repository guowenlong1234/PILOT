#!/usr/bin/env python3
"""Aggregate fixed paired-camera shards with scene-level uncertainty."""
import argparse,json,collections
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--output',required=True);p.add_argument('--manifest');a=p.parse_args()
 world=[];cwp=[]
 for f in a.files:
  d=json.loads(Path(f).read_text());assert d['status']=='completed';world+=d['world'];cwp+=d['cwp']
 if a.manifest:
  spec=json.loads(Path(a.manifest).read_text());ids={j['id'] for j in spec['jobs']};seeds=spec['seeds'];cams=spec['cameras']
 else:
  ids={r['id'] for r in world};seeds=[11,29,47];cams=['original','height_only','fov_only','rxr'];assert len(ids)==48
 expected={(i,c,s) for i in ids for c in cams for s in seeds}
 assert len(world)==len(expected) and {(r['id'],r['camera'],r['seed']) for r in world}==expected
 ce={(i,c,'predicted',s) for i in ids for c in cams for s in seeds}|{(i,c,'real',None) for i in ids for c in cams}
 assert len(cwp)==len(ce) and {(r['id'],r['camera'],r['input'],r['seed']) for r in cwp}==ce
 scenes=sorted({r['scene'] for r in world})
 if a.manifest:assert set(scenes)==set(spec['scenes'])
 cameras=['original','height_only','fov_only','rxr'];metrics=['cls_cosine','patch_cosine','cls_rmse','source_cls_cosine']
 result={'world':{},'cwp':{},'paired':{},'scenes':scenes,'queries':len(ids),'cwp_scene_macro':{}}
 for cam in cameras:
  w=[r for r in world if r['camera']==cam]
  result['world'][cam]={k:float(np.mean([np.mean([r[k] for r in w if r['scene']==s]) for s in scenes])) for k in metrics}
  result['world'][cam]['gain_over_source']=result['world'][cam]['cls_cosine']-result['world'][cam]['source_cls_cosine']
  result['cwp'][cam]={}
  for kind in ['real','predicted']:
   rows=[r for r in cwp if r['camera']==cam and r['input']==kind]
   valid=[r for r in rows if not r['none']]
   summary={'rows':len(rows),'none_rate':float(np.mean([r['none'] for r in rows])),
    'proposals':len(valid),'clear_given_proposal':float(np.mean([r['clear'] for r in valid])) if valid else None,
    'clear_and_proposed_rate':float(np.mean([r['clear'] is True for r in rows]))}
   if kind=='predicted':
    for key in ['real_none_agree','real_angle_error','real_distance_error']:
     vals=[r[key] for r in rows if r.get(key) is not None];summary[key]=float(np.mean(vals)) if vals else None
   result['cwp'][cam][kind]=summary
 for cam in cameras:
  result['cwp_scene_macro'][cam]={}
  for kind in ['real','predicted']:
   by_scene={}
   for scene in scenes:
    rs=[r for r in cwp if r['camera']==cam and r['input']==kind and r['scene']==scene]
    props=[r for r in rs if not r['none']]
    by_scene[scene]={'none_rate':float(np.mean([r['none'] for r in rs])),
      'clear_and_proposed_rate':float(np.mean([r['clear'] is True for r in rs])),
      'clear_given_proposal':float(np.mean([r['clear'] for r in props])) if props else None}
   result['cwp_scene_macro'][cam][kind]={'per_scene':by_scene,'mean':{k:float(np.mean([v[k] for v in by_scene.values() if v[k] is not None])) for k in ['none_rate','clear_and_proposed_rate','clear_given_proposal']}}
 rng=np.random.default_rng(20261008)
 for cam in cameras[1:]:
  result['paired'][cam]={}
  for key in metrics+['gain_over_source']:
   differences=[]
   for s in scenes:
    def m(c):
     rs=[r for r in world if r['scene']==s and r['camera']==c]
     return np.mean([r['cls_cosine']-r['source_cls_cosine'] if key=='gain_over_source' else r[key] for r in rs])
    differences.append(float(m(cam)-m('original')))
   boot=np.array(differences)[rng.integers(0,len(scenes),size=(10000,len(scenes)))].mean(1)
   result['paired'][cam][key]={'delta':float(np.mean(differences)),'scene_deltas':dict(zip(scenes,differences)),
      'scene_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
 Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
