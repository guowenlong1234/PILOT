#!/usr/bin/env python3
"""Check reusable runtime against scored feature input on a real capture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--encoded',required=True);ap.add_argument('--mode',required=True);ap.add_argument('--report',required=True)
    args=ap.parse_args()
    from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder
    from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    from vlnce_baselines.nwm.panorama_runtime import ObservedPanoramaFrame,PanoramaHistory,PanoramaTarget,PanoramaPredictionRuntime
    encoded=torch.load(args.encoded,map_location='cpu',weights_only=False)
    capture=torch.load(encoded['source'],map_location='cpu',weights_only=False)
    encoder=RaeDinov2RgbEncoder('pretrained/rae_dinov2_with_registers_base',device=torch.device('cuda:0'),precision='float32').eval()
    normalizer=RaeNwmLatentNormalizer('pretrained/raenwm_stage0/stat.pt').cuda()
    predictor=RaeNwmPredictor('configs/nwm/raenwm_mp3d_fresh_cls.yaml','pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar',device='cuda:0',use_external_context_latents=True)
    history=PanoramaHistory()
    for i in range(4):history.append(ObservedPanoramaFrame(str(i),'segment',capture['positions'][i],capture['yaws'][i],capture['cube_rgb'][i],capture['native_world_rgb12'][i],capture['front_rgb'][i]))
    targets=[PanoramaTarget(0,q['target']['query'],tuple(q['target']['position']),q['target']['yaw']) for q in encoded['queries']]
    gen=torch.Generator(device='cuda').manual_seed(771)
    noise=torch.randn(len(targets),257,768,device='cuda',generator=gen)
    runtime=PanoramaPredictionRuntime(encoder=encoder,normalizer=normalizer,predictor=predictor,mode=args.mode)
    result=runtime.predict(targets,{0:history},initial_noise=noise)
    first=dict(runtime.last_diagnostics)
    second=runtime.predict(targets,{0:history},initial_noise=noise)
    assert runtime.last_diagnostics['encoded_views']==0
    contexts=[];delta=[];rel=[]
    for q in encoded['queries']:
        v=q['variants'][args.mode];contexts.append(encoded['features'][v['ids']].float());delta.append(v['delta']);rel.append(v['rel_t'])
    with torch.no_grad():_,reference=predictor._predict_time_from_latents(torch.stack(contexts),torch.tensor(delta)[:,None],torch.tensor(rel),initial_noise=noise)
    cosine=F.cosine_similarity(reference.float().flatten(1),result.pred_tokens.float().flatten(1),dim=1)
    repeat=(second.pred_tokens-result.pred_tokens).abs()
    assert cosine.min()>.9999,cosine
    assert repeat.max()==0,repeat.max()
    assert [r.query_id for r in result.meta['records']]==[t.query_id for t in targets]
    report={'mode':args.mode,'queries':len(targets),'min_reference_cosine':cosine.min().item(),
        'reference_max_abs':(reference-result.pred_tokens).abs().max().item(),
        'repeat_max_abs':repeat.max().item(),'first_call':first,'cached_call':runtime.last_diagnostics,
        'metadata':runtime.context_metadata,'passed':True}
    Path(args.report).write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
