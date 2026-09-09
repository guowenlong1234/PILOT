#!/usr/bin/env python3
"""Paired frozen-context latency and output parity benchmark on real captures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--captures',required=True);p.add_argument('--output',required=True)
    p.add_argument('--reference');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--compile-model',action='store_true')
    p.add_argument('--legacy-inductor',action='store_true',help='diagnostic only: reproduce the rejected compiler')
    p.add_argument('--prediction-batch-size',type=int,default=8)
    args=p.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder
    from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    import vlnce_baselines.nwm.panorama_runtime as m
    encoder=RaeDinov2RgbEncoder('pretrained/rae_dinov2_with_registers_base',device=torch.device('cuda:0'),precision='float32').eval()
    normalizer=RaeNwmLatentNormalizer('pretrained/raenwm_stage0/stat.pt').cuda()
    predictor=RaeNwmPredictor('configs/nwm/raenwm_mp3d_fresh_cls.yaml','pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar',
        device='cuda:0',enable_decoder=False,num_steps=10,final_only_euler=False,use_external_context_latents=True)
    if args.compile_model:
        from vlnce_baselines.nwm.compile_runtime import compile_frozen_world_model
        predictor.bundle.model,predictor.compile_backend=compile_frozen_world_model(predictor.bundle.model)
    elif args.legacy_inductor:
        predictor.bundle.model=torch.compile(predictor.bundle.model,mode='reduce-overhead')
    runtime=m.PanoramaPredictionRuntime(encoder=encoder,normalizer=normalizer,predictor=predictor,
        mode='world_exact_select',prediction_batch_size=args.prediction_batch_size)
    stages=defaultdict(float)
    def wrap(obj,name,key):
        old=getattr(obj,name)
        def call(*a,**kw):
            torch.cuda.synchronize();start=time.perf_counter();result=old(*a,**kw);torch.cuda.synchronize()
            stages[key]+=time.perf_counter()-start;return result
        setattr(obj,name,call)
    wrap(m,'observed_view','projection')
    if hasattr(runtime,'_observed_rgb_batch'):
        wrap(runtime,'_observed_rgb_batch','projection')
    wrap(encoder,'forward_raw_cls_and_patch_latents','encoder')
    wrap(predictor,'predict_time_from_etp_batch','predictor')
    files=sorted(Path(args.captures).glob('capture*.pt'));assert files,'No captures'
    rows=[];outputs=[]
    for path in files:
        capture=torch.load(path,map_location='cpu',weights_only=False)
        histories={}
        for i,frames in capture['histories'].items():
            h=m.PanoramaHistory()
            for f in frames:h.append(m.ObservedPanoramaFrame(**f))
            histories[i]=h
        targets=capture['targets'];noise=torch.randn(len(targets),257,768,device='cuda',generator=torch.Generator(device='cuda').manual_seed(7321))
        for repeat in range(args.repeats+1):
            for h in histories.values():
                for f in h.frames:f._cache.clear()
            stages.clear();torch.cuda.synchronize();start=time.perf_counter()
            result=runtime.predict(targets,histories,initial_noise=noise);torch.cuda.synchronize();cold=time.perf_counter()-start
            cold_stages=dict(stages);diagnostics=dict(runtime.last_diagnostics)
            start=time.perf_counter();cached=runtime.predict(targets,histories,initial_noise=noise);torch.cuda.synchronize();hot=time.perf_counter()-start
            torch.testing.assert_close(result.pred_tokens,cached.pred_tokens,rtol=0,atol=0)
            if repeat:
                rows.append(dict(capture=path.name,cold=cold,cached=hot,stages=cold_stages,diagnostics=diagnostics))
        outputs.append(result.pred_tokens.cpu())
    parity=[]
    if args.reference:
        ref=torch.load(Path(args.reference)/'outputs.pt',map_location='cpu',weights_only=False)
        assert len(ref)==len(outputs)
        for a,b in zip(outputs,ref):
            assert a.shape==b.shape
            max_abs=(a-b).abs().max().item();cos=F.cosine_similarity(a.flatten(1),b.flatten(1),dim=1).min().item()
            error=None
            try:torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
            except AssertionError as exc:error=str(exc)
            parity.append(dict(max_abs=max_abs,min_cosine=cos,passed=error is None,error=error))
    torch.save(outputs,root/'outputs.pt')
    report=dict(rows=rows,mean_cold=float(np.mean([r['cold'] for r in rows])),mean_cached=float(np.mean([r['cached'] for r in rows])),
        mean_stages={k:float(np.mean([r['stages'].get(k,0) for r in rows])) for k in ('projection','encoder','predictor')},parity=parity)
    (root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    assert all(r['passed'] for r in parity),'Prediction parity failed; measurements retained in report.json'


if __name__=='__main__':main()
