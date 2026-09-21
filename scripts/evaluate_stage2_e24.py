"""One-pass offline dev evaluation; no simulator and no gradients."""
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from vlnce_baselines.nwm.active_lookahead.stage2_data import (
    Stage2Dataset, collate_stage2, load, sha256, atomic_json, FORMAT, SPACE)
from vlnce_baselines.nwm.active_lookahead.residual_head import InterleavedCrossModalTopKFutureLogitResidualHead
from vlnce_baselines.nwm.active_lookahead.stage2_evaluation import (
    CALIBRATION_GAINS, decision_records, build_report, selection_key)


class SequentialEpisodes(IterableDataset):
    def __init__(self, roots, verify_hash=False):
        # Validate provenance and cross-root disjointness using the shared reader.
        checked = Stage2Dataset(roots)
        self.entries = []
        self.provenance = []
        self.rows = len(checked)
        self.verify_hash = verify_hash
        for root in roots:
            manifest = json.loads((Path(root)/'dataset_manifest.json').read_text())
            provenance = json.loads((Path(root)/'provenance.json').read_text())
            if provenance.get('split') != 'val_unseen':
                raise ValueError('offline dev evaluation requires val_unseen')
            self.provenance.append(provenance)
            for entry in manifest['entries']:
                self.entries.append((str(Path(root)/entry['file']), entry))

    def __iter__(self):
        worker = get_worker_info()
        entries = self.entries if worker is None else self.entries[worker.id::worker.num_workers]
        for path, entry in entries:
            if self.verify_hash and sha256(path) != entry['sha256']:
                raise ValueError('dev shard SHA mismatch: '+path)
            obj = load(path)
            if (obj['format'] != FORMAT or obj['feature_space'] != SPACE or
                str(obj['episode_id']) != str(entry['episode_id']) or len(obj['rows']) != entry['rows']):
                raise ValueError('dev shard identity/schema mismatch: '+path)
            for row in obj['rows']:
                yield dict(row, text_tokens=obj['text_tokens'], episode_id=obj['episode_id'], scene_id=obj['scene_id'])


def collate(rows):
    batch = collate_stage2(rows)
    batch['episode_id'] = [str(r['episode_id']) for r in rows]
    batch['scene_id'] = [str(r['scene_id']) for r in rows]
    return batch


def predict(head, batch):
    logits = batch['base_logits'].masked_fill(~batch['ghost_valid_mask'], -torch.inf)
    empty = ~batch['ghost_valid_mask'].any(1)
    logits = logits.clone()
    logits[empty] = 0
    lp = torch.log_softmax(logits, 1)
    present = batch['topk_base_indices'].ge(0)
    context_mask = present if getattr(head, 'candidate_context_mode', 'future_valid') == 'all_present' else batch['topk_valid_mask']
    selected = lp.gather(1, batch['topk_base_indices'].clamp_min(0)).masked_fill(~context_mask, 0)
    return head.forward_topk_from_log_probs(batch['owner_embeddings'], batch['text_tokens'],
        batch['future_tokens'], selected, batch['topk_valid_mask'], text_token_mask=batch['text_token_mask'],
        candidate_geometry=batch['candidate_q0_geometry'], candidate_present_mask=present).delta


def validate_checkpoint(checkpoint, dev_provenance):
    if checkpoint.get('format_version') not in ('stage2-e24-head-v1', 'stage2-e24-training-v1'):
        raise ValueError('unsupported checkpoint format')
    train_datasets = checkpoint['contract']['dataset']
    if not train_datasets:
        raise ValueError('missing checkpoint data provenance')
    fields = ('stage1_sha256', 'assets', 'feature_space', 'context_contract', 'behavior')
    for source in train_datasets:
        train = source['provenance']
        if train.get('split') != 'train':
            raise ValueError('checkpoint was not trained on train split')
        for dev in dev_provenance:
            if any(train.get(key) != dev.get(key) for key in fields):
                raise ValueError('checkpoint and dev asset/context contract differs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-roots', nargs='+', required=True)
    p.add_argument('--checkpoints', nargs='+', default=[])
    p.add_argument('--zero-init', action='store_true')
    p.add_argument('--zero-init-future-mode', choices=['full', 'none'], default='full',
                   help='future-content contract for the optional zero-init control')
    p.add_argument('--gains', nargs='+', type=float, default=[1.], choices=CALIBRATION_GAINS)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--amp', choices=['none','fp16','bf16'], default='none')
    p.add_argument('--bootstrap-samples', type=int, default=2000)
    p.add_argument('--verify-hashes', action='store_true')
    p.add_argument('--report', required=True)
    args = p.parse_args()
    if not args.checkpoints and not args.zero_init: p.error('provide checkpoints or --zero-init')
    dataset = SequentialEpisodes(args.data_roots, args.verify_hashes)
    options = dict(batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)
    if args.num_workers: options.update(prefetch_factor=1, multiprocessing_context='spawn')
    loader = DataLoader(dataset, **options)
    models = []
    for path in args.checkpoints + (['zero-init'] if args.zero_init else []):
        if path == 'zero-init':
            from vlnce_baselines.nwm.active_lookahead.stage2_training import MODEL_CONFIG
            config = dict(MODEL_CONFIG, future_mode=args.zero_init_future_mode)
            step, state = 0, None
        else:
            checkpoint = load(path)
            validate_checkpoint(checkpoint, dataset.provenance)
            config, step = checkpoint['model_config'], checkpoint['global_step']
            state = checkpoint['future_head_state_dict']
        head = InterleavedCrossModalTopKFutureLogitResidualHead(**config).to(args.device).eval()
        if state is not None: head.load_state_dict(state, strict=True)
        models.append((path, step, head, config.get('future_mode', 'full')))
    records = {(path,gain): [] for path,_,_,_ in models for gain in args.gains}
    started = time.monotonic()
    rows = 0
    with torch.inference_mode():
        for batch in loader:
            device_batch = {k:v.to(args.device) if torch.is_tensor(v) else v for k,v in batch.items()}
            for path,step,head,_future_mode in models:
                with torch.autocast(device_type=torch.device(args.device).type,
                    dtype=torch.float16 if args.amp == 'fp16' else torch.bfloat16, enabled=args.amp != 'none'):
                    delta = predict(head, device_batch)
                for gain in args.gains:
                    records[path,gain].extend(decision_records(batch, delta, gain))
            rows += len(batch['episode_id'])
            if rows % (args.batch_size * 100) == 0:
                print(json.dumps(dict(rows=rows,total=dataset.rows,elapsed_seconds=time.monotonic()-started)), flush=True)
    if rows != dataset.rows: raise RuntimeError('incomplete dev evaluation')
    results = []
    for path,step,_head,future_mode in models:
        for gain in args.gains:
            report = build_report(records[path,gain], args.bootstrap_samples)
            report.update(checkpoint=path, checkpoint_sha256=sha256(path) if path != 'zero-init' else None,
                          global_step=step, gain=gain, future_mode=future_mode)
            if report['stop_consistent_rows'] != rows: raise RuntimeError('STOP changed')
            if (path == 'zero-init' or gain == 0) and report['all_action_changes']:
                raise RuntimeError('zero residual changed action')
            results.append(report)
    ranked = sorted([r for r in results if r['gain'] == 1.], key=selection_key)
    output = dict(format_version='stage2-offline-evaluation-v1', status='complete',
        dataset=[dict(root=str(Path(root).resolve()), manifest_sha256=sha256(Path(root)/'dataset_manifest.json')) for root in args.data_roots],
        config=vars(args), elapsed_seconds=time.monotonic()-started, results=results,
        selected_g1=[dict(checkpoint=r['checkpoint'],global_step=r['global_step'],gain=1.) for r in ranked[:2]],
        limitation='val_unseen development decisions; not navigation SR/SPL or independent test results')
    atomic_json(args.report, output)
    print(json.dumps(dict(report=args.report, results=[{k:r[k] for k in ('checkpoint','gain','primary_rows','fixes','harms','net_fixes','base_accuracy','accuracy')} for r in results])), flush=True)


if __name__ == '__main__':
    main()
