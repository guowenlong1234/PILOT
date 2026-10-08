#!/usr/bin/env python3
"""Aggregate fixed paired-camera shards with scene-level uncertainty."""
import argparse,json,collections
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
 world=[];cwp=[]
 for f in a.files:
  d=json.loads(Path(f).read_text());assert d['status']=='completed';world+=d['world'];cwp+=d['cwp']
 assert len(world)==576 and len(cwp)==768
 assert len({(r['id'],r['camera'],r['seed']) for r in world})==576
 scenes=sorted({r['scene'] for r in world});assert len(scenes)==6
 cameras=['original','height_only','fov_only','rxr'];metrics=['cls_cosine','patch_cosine','cls_rmse','source_cls_cosine']
 result={'world':{},'cwp':{},'paired':{},'scenes':scenes}
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
