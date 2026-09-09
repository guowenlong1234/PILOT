#!/usr/bin/env python3
"""Isolate compiler errors on identical real latent inputs and explicit noise."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F


def metrics(actual, expected):
    a, b = actual.float(), expected.float()
    delta = (a - b).abs()
    return dict(max_abs=delta.max().item(), rmse=delta.square().mean().sqrt().item(),
                mismatch_fraction=(delta > 1e-4 + b.abs() * 1e-4).float().mean().item(),
                cosine_min=F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1).min().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--inputs')
    p.add_argument('--capture', default='data/logs/panorama_perf_20260909/baseline_v1/capture0.pt')
    p.add_argument('--variant', choices=['eager', 'aot_eager', 'default', 'nofusion'], default='eager')
    p.add_argument('--modules', action='store_true')
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    from vlnce_baselines.nwm.predictor import RaeNwmPredictor
    predictor = RaeNwmPredictor('configs/nwm/raenwm_mp3d_fresh_cls.yaml',
        'pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar', device='cuda:0',
        num_steps=10, use_external_context_latents=True)
    model = predictor.bundle.model
    if args.inputs:
        data = torch.load(args.inputs, map_location='cuda', weights_only=False)
    else:
        from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder
        from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer
        from vlnce_baselines.nwm.panorama_runtime import (
            PanoramaHistory, ObservedPanoramaFrame, PanoramaPredictionRuntime)
        encoder = RaeDinov2RgbEncoder('pretrained/rae_dinov2_with_registers_base',
            device=torch.device('cuda:0'), precision='float32').eval()
        normalizer = RaeNwmLatentNormalizer('pretrained/raenwm_stage0/stat.pt').cuda()
        runtime = PanoramaPredictionRuntime(encoder=encoder, normalizer=normalizer,
            predictor=predictor, mode='world_exact_select')
        capture = torch.load(args.capture, map_location='cpu', weights_only=False)
        histories = {}
        for i, frames in capture['histories'].items():
            history = PanoramaHistory()
            for frame in frames: history.append(ObservedPanoramaFrame(**frame))
            histories[i] = history
        class Captured(Exception): pass
        data = {}
        original = predictor.predict_time_from_etp_batch
        def grab(batch, **kw):
            data.update(context_latent=batch.context_latent, curr_delta=batch.curr_delta,
                        rel_t=batch.rel_t, initial_noise=kw['initial_noise'])
            raise Captured()
        predictor.predict_time_from_etp_batch = grab
        noise = torch.randn(len(capture['targets']), 257, 768, device='cuda',
            generator=torch.Generator(device='cuda').manual_seed(7321))
        try: runtime.predict(capture['targets'], histories, initial_noise=noise)
        except Captured: pass
        predictor.predict_time_from_etp_batch = original
        torch.save({k:v.cpu() for k,v in data.items()}, root/'inputs.pt')
        del runtime, encoder, normalizer, histories
    import torch._inductor.config as config
    options = {'triton.cudagraphs': False}
    if args.variant == 'nofusion': options.update(max_fusion_size=1, epilogue_fusion=False)
    def compile_fn(fn):
        if args.variant == 'eager': return fn
        if args.variant == 'aot_eager': return torch.compile(fn, backend='aot_eager', fullgraph=True)
        return torch.compile(fn, fullgraph=True, options=options)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report = {'variant':args.variant, 'torch':torch.__version__, 'options':options}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        inputs = dict(x=data['initial_noise'], t=torch.ones(len(data['rel_t']), device='cuda'),
                      y=data['curr_delta'].flatten(0,1), x_cond=data['context_latent'], rel_t=data['rel_t'])
        expected = model(**inputs).clone()
        compiled = compile_fn(model)
        start = time.perf_counter(); actual = compiled(**inputs).clone(); torch.cuda.synchronize()
        report['first_call_seconds'] = time.perf_counter()-start
        report['single_forward'] = metrics(actual, expected)
        print(json.dumps(report), flush=True)
        # Compare every step at identical eager states, then measure accumulated error.
        original = predictor.bundle.model
        step_rows = []
        def paired(*a, **kw):
            eager = original(*a, **kw)
            value = compiled(*a, **kw)
            step_rows.append(metrics(value, eager))
            return eager
        predictor.bundle.model = paired
        _, expected_final = predictor._predict_time_from_latents(**data)
        predictor.bundle.model = compiled
        _, actual_final = predictor._predict_time_from_latents(**data)
        report['steps_same_input'] = step_rows
        report['final'] = metrics(actual_final, expected_final)
        print('FINAL '+json.dumps(report['final']), flush=True)
        if args.modules:
            # Feed representative submodules their exact eager inputs; compiling
            # them separately distinguishes reduction/fusion errors from drift.
            samples = {}; handles = []
            names = ['t_embedder', 'y_embedder', 'time_embedder', 'dyn_fuse',
                     'blocks.0.norm1', 'blocks.0.adaLN_modulation', 'blocks.0.attn',
                     'blocks.0.norm_cond', 'blocks.0.cttn', 'blocks.0.mlp', 'blocks.0']
            for name, module in model.named_modules():
                if name in names:
                    def capture_inputs(mod, a, kw, out, name=name):
                        samples[name] = (mod,a,kw,out)
                    handles.append(module.register_forward_hook(capture_inputs, with_kwargs=True))
            model(**inputs)
            for handle in handles: handle.remove()
            report['modules'] = {}
            for name, (module,a,kw,out) in samples.items():
                value = compile_fn(module)(*a, **kw)
                if isinstance(out, tuple): value,out=value[0],out[0]
                report['modules'][name] = metrics(value, out)
                print(name+' '+json.dumps(report['modules'][name]), flush=True)
    (root/'report.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__': main()
