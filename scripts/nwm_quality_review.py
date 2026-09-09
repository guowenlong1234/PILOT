#!/usr/bin/env python3
"""Summarize the locked winner on untouched scenes, with paired uncertainty."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',required=True)
    ap.add_argument('--confirmation',default='confirmation');args=ap.parse_args()
    root=Path(args.root);spec=json.loads((root/'split.json').read_text())
    selection=json.loads((root/'selection.json').read_text());winner=selection['winner']
    conf=json.loads((root/(args.confirmation+'.json')).read_text());rows=conf['rows']
    assert set(spec['development_scenes']).isdisjoint(spec['confirmation_scenes'])
    assert conf['split']=='confirmation' and set(conf['modes'])=={'front',winner}
    assert conf['seeds']==spec['confirmation_seeds']
    expected=set()
    for job in spec['jobs']:
        if job['split']!='confirmation':continue
        for file in (root/'encoded'/job['episode']).glob('decision_*.pt'):
            import torch
            d=torch.load(file,map_location='cpu',weights_only=False)
            expected.update(q['id'] for q in d['queries'])
    grouped=defaultdict(dict)
    for r in rows:
        key=(r['id'],r['mode']);assert r['seed'] not in grouped[key]
        grouped[key][r['seed']]=r
    assert {r['id'] for r in rows}==expected
    paired=[]
    for key in sorted(expected):
        a=grouped[(key,'front')];b=grouped[(key,winner)]
        assert set(a)==set(b)==set(spec['confirmation_seeds'])
        item={'id':key,'scene':a[11]['scene'],'episode':a[11]['episode'],
              'original_turn_deg':a[11]['original_turn_deg'],'winner_source_index':b[11]['source_index']}
        for metric in ['cls_cosine','patch_cosine','cls_rmse','source_cls_cosine']:
            item['base_'+metric]=float(np.mean([r[metric] for r in a.values()]))
            item['winner_'+metric]=float(np.mean([r[metric] for r in b.values()]))
            item['delta_'+metric]=item['winner_'+metric]-item['base_'+metric]
        paired.append(item)
    per_scene=[]
    for scene in spec['confirmation_scenes']:
        items=[x for x in paired if x['scene']==scene]
        assert items
        entry={'scene':scene,'queries':len(items)}
        for field in paired[0]:
            if field.startswith(('base_','winner_','delta_')):entry[field]=float(np.mean([x[field] for x in items]))
        per_scene.append(entry)
    rng=np.random.default_rng(909);uncertainty={}
    for metric in ['cls_cosine','patch_cosine','cls_rmse']:
        deltas=np.array([x['delta_'+metric] for x in per_scene])
        bootstrap=deltas[rng.integers(0,len(deltas),(10000,len(deltas)))].mean(1)
        uncertainty[metric]={'mean_delta':float(deltas.mean()),'scene_bootstrap_95':np.quantile(bootstrap,[.025,.975]).tolist()}
    by_angle={}
    for name,predicate in [('small_le45',lambda a:abs(a)<=45),('middle_45_90',lambda a:45<abs(a)<=90),('large_gt90',lambda a:abs(a)>90)]:
        subset=[x for x in paired if predicate(x['original_turn_deg'])]
        by_angle[name]={'queries':len(subset),**{field:float(np.mean([x[field] for x in subset])) for field in ['base_cls_cosine','winner_cls_cosine','delta_cls_cosine','base_patch_cosine','winner_patch_cosine']} if subset else {'queries':0}
    result={'winner':winner,'confirmation_queries':len(paired),'scenes':len(per_scene),
        'noise_seeds':spec['confirmation_seeds'],'cls_improved_queries':sum(x['delta_cls_cosine']>0 for x in paired),
        'per_scene':per_scene,'uncertainty':uncertainty,'by_original_angle':by_angle,'paired_queries':paired,
        'no_navigation_performance_claim':True}
    (root/'review.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='paired_queries'}),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    colors=['#dc7257' if abs(x['original_turn_deg'])>90 else '#3575b5' for x in paired]
    ax[0].scatter([x['base_cls_cosine'] for x in paired],[x['winner_cls_cosine'] for x in paired],c=colors,s=22,alpha=.7)
    ax[0].plot([0,1],[0,1],'k--',lw=1);ax[0].set(xlabel='Original front context: CLS cosine',ylabel='Selected panorama context: CLS cosine',title='Held-out queries (3 noise seeds averaged)',xlim=(0,1),ylim=(0,1))
    ax[1].barh([x['scene'] for x in per_scene],[x['delta_cls_cosine'] for x in per_scene],color='#3575b5')
    ax[1].axvline(0,color='black',lw=.8);ax[1].set(xlabel='Change in mean CLS cosine',title='Each held-out scene')
    fig.savefig(root/'confirmation_quality.png',dpi=160)


if __name__=='__main__':main()
