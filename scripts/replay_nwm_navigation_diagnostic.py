#!/usr/bin/env python3
"""Replay captured queries with identical noise at 10/50 integration points."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--checkpoint', required=True)
    args=parser.parse_args()
    root=Path(args.capture)
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    predictor=RaeNwmPredictor('configs/nwm/raenwm_mp3d_fresh_cls.yaml', args.checkpoint,
        device='cuda:0', use_external_context_latents=True)
    rows=[]
    for path in sorted(root.glob('sample_*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        batch=SimpleNamespace(is_empty=False,context_latent=data['context'],
            curr_delta=data['curr_delta'],rel_t=data['rel_t'],records=[],skipped={})
        outputs={}
        for steps in (10,50):
            predictor.bundle.config['transport']['num_steps']=steps
            with torch.no_grad():
                pred=predictor.predict_time_from_etp_batch(batch,initial_noise=data['initial_noise'])
            outputs[steps]=pred.pred_latent.float().cpu()[0]
        mean=data['normalizer_mean'].mean((2,3))[0]
        scale=torch.sqrt(data['normalizer_var'].mean((2,3))[0]+1e-5)
        gt=data['truth_tokens'][-1]
        row={'index':data['meta']['index'],'episode':data['meta']['episode'],
             'turn_deg':data['meta']['condition']['dtheta']*180/np.pi,
             'target_navigable':data['meta']['target_navigable']}
        for steps,p in outputs.items():
            row[f'cls_cos_{steps}']=F.cosine_similarity(p[0]*scale+mean,gt[0]*scale+mean,dim=0).item()
            row[f'patch_cos_{steps}']=F.cosine_similarity(p[1:],gt[1:],dim=-1).mean().item()
        row['replay_original_cos']=F.cosine_similarity(outputs[10].flatten(),data['pred_tokens'].flatten(),dim=0).item()
        row['replay_original_max_abs']=(outputs[10]-data['pred_tokens']).abs().max().item()
        rows.append(row)
        torch.save({'tokens_10':outputs[10],'tokens_50':outputs[50]},root/f'replay_{row["index"]:03d}.pt')
        if row['index']==0:
            torch.save({k:data[k] for k in ('context','curr_delta','rel_t','initial_noise')},root/'parity_input.pt')
            torch.save({'tokens':outputs[10][None],'cls_normalized':outputs[10][None,0],
                        'patch_tokens':outputs[10][None,1:],'versions':{'torch':str(torch.__version__)}},root/'parity_etpr1.pt')
        print(json.dumps(row),flush=True)
    result={'rows':rows,'means':{key:float(np.mean([r[key] for r in rows])) for key in
        ('cls_cos_10','cls_cos_50','patch_cos_10','patch_cos_50','replay_original_cos','replay_original_max_abs')}}
    (root/'replay.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['means']),flush=True)


if __name__=='__main__':main()
