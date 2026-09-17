#!/usr/bin/env python3
"""Train a fresh E24 head on immutable, predicted-future Stage-2 train shards."""
import argparse
from contextlib import nullcontext
from dataclasses import fields
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from vlnce_baselines.nwm.active_lookahead.residual_head import InterleavedCrossModalTopKFutureLogitResidualHead
from vlnce_baselines.nwm.active_lookahead.stage2_data import atomic_json, collate_stage2
from vlnce_baselines.nwm.active_lookahead.stage2_training import (
    MODEL_CONFIG, EpisodeBlockSampler, forward_delta, compute_loss, seed_everything,
    save_training_checkpoint, restore_training_checkpoint, model_fingerprint,
)


def versions():
    result = dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda)
    for package in ('transformers', 'habitat-lab', 'habitat-sim'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = 'package metadata unavailable'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--block-episodes', type=int, default=32)
    parser.add_argument('--save-every', type=int, default=250)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--stop-after-steps', type=int, default=0,
                        help='absolute interruption step, excluded from resume contract')
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.batch_size, args.max_steps, args.max_epochs, args.save_every, args.log_every, args.threads) < 1 or args.lr <= 0:
        parser.error('step, batch, epoch, learning rate, thread and interval values must be positive')
    if args.precision != 'fp32' and not args.device.startswith('cuda'):
        parser.error('mixed precision is supported on CUDA only')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError('fresh training requires a new empty output directory')
    if args.resume and Path(args.resume).resolve().parent != output:
        raise ValueError('resume must continue in the checkpoint experiment directory')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(args.seed)
    sampler = EpisodeBlockSampler(args.roots, seed=args.seed, block_episodes=args.block_episodes)
    config = {k:v for k,v in vars(args).items() if k not in ('roots','output','resume','stop_after_steps')}
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    contract = dict(dataset=sampler.provenance, train_config=config, git_commit=git_commit,
        experiment=str(output), optimizer=dict(name='AdamW', lr=args.lr, betas=[.9,.999], eps=1e-8, weight_decay=.01),
        gradient_clip=10., scheduler=None, versions=versions(), trainable_module='E24_only', initialization='fresh_seeded')
    head = InterleavedCrossModalTopKFutureLogitResidualHead(**MODEL_CONFIG).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(.9,.999), eps=1e-8, weight_decay=.01)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    if optimizer_ids != {id(p) for p in head.parameters() if p.requires_grad}:
        raise RuntimeError('optimizer does not match E24-only parameters')
    audit_path = output/'parameter_audit.json'
    audit = json.loads(audit_path.read_text()) if args.resume and audit_path.exists() else dict(
        optimizer_exactly_e24=True, initial_sha256=model_fingerprint(head), nonzero_gradient_modules={})
    atomic_json(audit_path, audit)
    scaler = torch.cuda.amp.GradScaler() if args.precision == 'fp16' else None
    step = restore_training_checkpoint(args.resume, head, optimizer, scaler, sampler, contract) if args.resume else 0
    atomic_json(output/'contract.json', contract)
    target = min(args.max_steps, math.ceil(sampler.total_rows/args.batch_size)*args.max_epochs)
    print(json.dumps(dict(event='start', step=step, target=target, usable_rows=sampler.total_rows,
        parameters=sum(p.numel() for p in head.parameters()), contract_file=str(output/'contract.json'),
        precision=args.precision, versions=contract['versions'])), flush=True)
    atomic_json(output/'status.json', dict(status='running', step=step, target=target))
    head.train()
    start = time.monotonic()
    window_rows, window_steps = 0, 0
    last_saved = step if args.resume else -1
    sums = {}
    try:
        with (output/'train.jsonl').open('a', buffering=1) as log:
            while step < args.max_steps:
                if args.stop_after_steps and step >= args.stop_after_steps:
                    break
                rows = sampler.next_rows(args.batch_size, args.max_epochs)
                if rows is None:
                    break
                batch = {k:v.to(args.device) for k,v in collate_stage2(rows).items()}
                optimizer.zero_grad(set_to_none=True)
                ctx = torch.autocast('cuda', dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16) if args.precision != 'fp32' else nullcontext()
                with ctx:
                    delta = forward_delta(head, batch)
                # Loss reductions and unbounded base-logit comparisons stay FP32.
                result = compute_loss(delta.float(), batch)
                if not torch.isfinite(result.loss):
                    raise FloatingPointError(f'non-finite loss before update {step+1}')
                if result.valid_sample_count == 0:
                    print(json.dumps(dict(event='skip_empty_loss', sampler=sampler.state_dict())), flush=True)
                    continue
                if scaler is None:
                    result.loss.backward()
                else:
                    scaler.scale(result.loss).backward()
                    scaler.unscale_(optimizer)
                if step < 3:
                    groups = {}
                    for name, parameter in head.named_parameters():
                        group = name.split('.')[0]
                        if parameter.grad is None:
                            raise RuntimeError(f'missing E24 gradient: {name}')
                        observed = (parameter.grad != 0).any()
                        groups[group] = groups.get(group, observed) | observed
                    for group, observed in groups.items():
                        audit['nonzero_gradient_modules'][group] = audit['nonzero_gradient_modules'].get(group, False) or bool(observed)
                    atomic_json(audit_path, audit)
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 10., error_if_nonfinite=True)
                if scaler is None:
                    optimizer.step()
                else:
                    scaler.step(optimizer)
                    scaler.update()
                step += 1
                for field in fields(result):
                    value = getattr(result, field.name)
                    value = float(value.detach()) if torch.is_tensor(value) else float(value)
                    sums[field.name] = sums.get(field.name, 0.) + value
                selected = delta.detach()[batch['topk_valid_mask']].float()
                sums['gradient_norm'] = sums.get('gradient_norm', 0.) + float(norm)
                sums['delta_abs'] = sums.get('delta_abs', 0.) + float(selected.abs().mean())
                sums['delta_saturation'] = sums.get('delta_saturation', 0.) + float((selected.abs() > .99).float().mean())
                window_rows += len(rows)
                window_steps += 1
                if step % args.log_every == 0 or step == target or step == args.stop_after_steps:
                    record = dict(step=step, epoch=sampler.epoch, sampler=sampler.state_dict(),
                        **{k:v/window_steps for k,v in sums.items()}, rows_per_second=window_rows/(time.monotonic()-start),
                        seconds_per_update=(time.monotonic()-start)/window_steps, lr=optimizer.param_groups[0]['lr'],
                        cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0)
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record)+'\n')
                    start = time.monotonic()
                    sums, window_rows, window_steps = {}, 0, 0
                if step % args.save_every == 0:
                    save_training_checkpoint(output, head, optimizer, scaler, sampler, step, contract)
                    last_saved = step
                    atomic_json(output/'status.json', dict(status='running', step=step, target=target))
            if step != last_saved:
                save_training_checkpoint(output, head, optimizer, scaler, sampler, step, contract)
            audit['final_sha256'] = model_fingerprint(head)
            audit['final_step'] = step
            atomic_json(audit_path, audit)
            completed = step >= args.max_steps or sampler.epoch >= args.max_epochs
            status = dict(status='completed' if completed else 'interrupted', step=step,
                target=target, sampler=sampler.state_dict())
            atomic_json(output/'status.json', status)
            print(json.dumps(status), flush=True)
    except BaseException as error:
        atomic_json(output/'status.json', dict(status='failed', step=step, error=repr(error), target=target))
        raise


if __name__ == '__main__':
    main()
