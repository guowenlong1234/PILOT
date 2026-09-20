#!/usr/bin/env python3
"""Real RxR updates, stage timing and action throughput; never saves checkpoints."""
import argparse
import hashlib
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import runpy
import sys
import subprocess
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--environments', type=int, default=6)
    parser.add_argument('--accumulation', type=int, default=2)
    parser.add_argument('--updates', type=int, default=18)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sync-stages', action='store_true')
    parser.add_argument('--lean-dino', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--compile-dino', action='store_true')
    parser.add_argument('--compile-depth', action='store_true')
    parser.add_argument('--async-finite', action='store_true')
    args, overrides = parser.parse_known_args()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    torch.cuda.set_device(rank)
    import habitat, habitat_sim, transformers
    from habitat import VectorEnv
    import vlnce_baselines.ss_trainer_ETP_R1 as module
    from vlnce_baselines.models.R1Policy import ETP
    if args.lean_dino or args.compile_dino or args.async_finite:
        from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder
        original_init = RaeDinov2RgbEncoder.__init__
        def encoder_init(self, *a, **kw):
            original_init(self, *a, **kw)
            self.async_finite_checks = args.async_finite
            original_forward = self.backbone.forward
            def forward(*a, **kw):
                if args.lean_dino or args.compile_dino:
                    kw['output_hidden_states'] = False
                return original_forward(*a, **kw)
            if args.compile_dino:
                compiled = torch.compile(forward, dynamic=True)
                def dispatch(pixels, **kw):
                    # PyTorch 2.2's antialiased interpolation needs concrete
                    # image dimensions; only the image batch is dynamic.
                    torch._dynamo.mark_static(pixels, [1, 2, 3])
                    return compiled(pixels, **kw)
                self.backbone.forward = dispatch
            else:
                self.backbone.forward = forward
        RaeDinov2RgbEncoder.__init__ = encoder_init
    if args.compile_depth:
        from vlnce_baselines.models.encoders.resnet_encoders import VlnResnetDepthEncoder
        original_depth_init = VlnResnetDepthEncoder.__init__
        def depth_init(self, *a, **kw):
            original_depth_init(self, *a, **kw)
            compiled = torch.compile(self.visual_encoder.forward, dynamic=True)
            def dispatch(observations):
                torch._dynamo.mark_static(observations['depth'], [1, 2, 3])
                return compiled(observations)
            self.visual_encoder.forward = dispatch
        VlnResnetDepthEncoder.__init__ = depth_init
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(float)
    steps, actions = [], [0]
    trainer, last = [None], [None]

    def wrap(cls, name, label):
        original = getattr(cls, name)
        def call(*a, **kw):
            key = label + ('_' + kw.get('mode', 'unknown') if label == 'policy' else '')
            if args.sync_stages:
                torch.cuda.synchronize(rank)
            start = time.perf_counter()
            if label == 'environment_step':
                actions[0] += len(a[1])
            result = original(*a, **kw)
            if args.sync_stages:
                torch.cuda.synchronize(rank)
            stats[key] += time.perf_counter() - start
            return result
        setattr(cls, name, call)

    wrap(ETP, 'forward', 'policy')
    wrap(VectorEnv, 'step', 'environment_step')
    wrap(module.RLTrainer, '_teacher_action_new', 'teacher')
    wrap(torch.Tensor, 'backward', 'backward')
    original_step = module.step_amp_optimizer
    def step(*a, **kw):
        result = original_step(*a, **kw)
        torch.cuda.synchronize(rank)
        now = time.perf_counter()
        steps.append(dict(seconds=now-last[0], actions=actions[0], stages=dict(stats),
                          optimizer_stepped=bool(result)))
        actions[0] = 0
        stats.clear()
        last[0] = now
        return result
    module.step_amp_optimizer = step
    def digest(named):
        result = hashlib.sha256()
        for name, parameter in named:
            result.update(name.encode())
            result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return result.hexdigest()
    original_interval = module.RLTrainer._train_interval
    def interval(self, *a, **kw):
        trainer[0] = self
        if args.audit:
            frozen_before = digest((n,p) for n,p in self.policy.named_parameters() if not p.requires_grad)
            mapping_before = digest((n,p) for n,p in self.policy.named_parameters() if 'cls_residual_mlp' in n)
        torch.cuda.synchronize(rank)
        last[0] = time.perf_counter()
        result = original_interval(self, *a, **kw)
        measured = steps[args.warmup:]
        report = dict(rank=rank, world=world, settings=vars(args), steps=steps,
                      source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                      seconds=sum(s['seconds'] for s in measured),
                      actions=sum(s['actions'] for s in measured),
                      mean_update_seconds=sum(s['seconds'] for s in measured)/len(measured),
                      peak_allocated_mib=torch.cuda.max_memory_allocated(rank)/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(rank)/2**20,
                      losses={k:list(v) for k,v in self.logs.items() if 'loss' in k.lower()},
                      scaler=self.scaler.state_dict(),
                      versions=dict(python=platform.python_version(),torch=str(torch.__version__),
                                    cuda=torch.version.cuda,transformers=transformers.__version__,
                                    habitat=habitat.__version__,habitat_sim=habitat_sim.__version__))
        if args.audit:
            report['audit'] = dict(
                frozen_unchanged=frozen_before == digest((n,p) for n,p in self.policy.named_parameters() if not p.requires_grad),
                mapping_updated=mapping_before != digest((n,p) for n,p in self.policy.named_parameters() if 'cls_residual_mlp' in n),
                finite_losses=all(torch.isfinite(torch.tensor(v)).all().item() for k,v in report['losses'].items()),
            )
            assert all(report['audit'].values()), report['audit']
        (root/f'rank{rank}.json').write_text(json.dumps(report, indent=2))
        print('RXR_BENCHMARK '+json.dumps(report), flush=True)
        return result
    module.RLTrainer._train_interval = interval
    module.RLTrainer.save_checkpoint = lambda *a, **kw: None
    opts = {
        'SIMULATOR_GPU_IDS':list(range(world)), 'TORCH_GPU_IDS':list(range(world)),
        'GPU_NUMBERS':world, 'NUM_ENVIRONMENTS':args.environments,
        'IL.batch_size':args.environments, 'IL.gradient_accumulation_steps':args.accumulation,
        'IL.iters':args.updates, 'IL.log_every':args.updates,
        'IL.load_from_ckpt':True, 'IL.is_requeue':False, 'IL.ckpt_to_load':args.checkpoint,
        'IL.checkpoint_sync_enabled':False, 'IL.warmup_iters':1,
        'CHECKPOINT_FOLDER':str(root/'checkpoints')+'/',
        'TENSORBOARD_DIR':str(root/'tb')+'/', 'RESULTS_DIR':str(root/'results')+'/',
        'TASK_CONFIG.DATASET.ROLES':['guide'],
        'TASK_CONFIG.DATASET.SUFFIX':'_90',
        'TASK_CONFIG.DATASET.LANGUAGES':['en-US','en-IN','hi-IN','te-IN'],
        'TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING':True,
    }
    sys.argv = ['run.py','--local_rank',str(rank),'--exp_name',root.name,
                '--run-type','dagger','--exp-config','run_rxr/iter_train_rae_dino_sft.yaml']
    for k,v in opts.items():
        sys.argv.extend([k,str(v)])
    sys.argv.extend(overrides)
    try:
        runpy.run_path('run.py', run_name='__main__')
    finally:
        if trainer[0] is not None:
            trainer[0].envs.close()


if __name__ == '__main__':
    main()
