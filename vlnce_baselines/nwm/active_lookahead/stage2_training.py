"""Deterministic E24-only offline training; no policy or world model is loaded."""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from .stage2_data import FORMAT, SPACE, atomic_json, collate_stage2, load, sha256
from .offline_objective import OfflineDecisionLossConfig, offline_decision_aware_loss

MODEL_CONFIG = dict(input_dim=768, hidden_dim=768, num_queries=8,
    num_attention_heads=12, num_layers=1, ffn_dim=3072, dropout=0.1,
    delta_max=1.0, fusion_layers=3, delta_centering='none',
    score_context='base_bounded_margin_relative', round_weight_sharing='independent',
    residual_confidence_gate='none')
LOSS_CONFIG = OfflineDecisionLossConfig(training_stage='offline', objective='decision_aware_v1',
    margin=.25, temperature=.25, eta_absent=.5, positive_group_weight=.5,
    negative_group_weight=.5, signed_weight=.25, present_signed_weight=1.,
    absent_signed_weight=1., final_weight=1., pair_weight=.5, regularization_weight=.001,
    absent_noop_weight=.05, pair_margin=.25, pair_temperature=.25,
    correct_row_weight=2., wrong_row_weight=1., decision_row_policy='all_teacher_topk', residual_bound=1.)


def forward_delta(head, batch):
    log_probs = torch.log_softmax(batch['base_logits'].masked_fill(~batch['ghost_valid_mask'], -torch.inf), dim=1)
    selected = log_probs.gather(1, batch['topk_base_indices'].clamp_min(0)).masked_fill(~batch['topk_valid_mask'], 0)
    return head.forward_topk_from_log_probs(batch['owner_embeddings'], batch['text_tokens'],
        batch['future_tokens'], selected, batch['topk_valid_mask'],
        text_token_mask=batch['text_token_mask'], candidate_geometry=batch['candidate_q0_geometry']).delta


def compute_loss(delta, batch):
    return offline_decision_aware_loss(delta, batch['teacher_rank_in_topk'],
        **{k: batch[k] for k in ('topk_valid_mask', 'teacher_valid', 'teacher_stop',
        'no_vp_left', 'base_stop', 'base_logits', 'ghost_valid_mask',
        'topk_base_indices', 'teacher_base_index')}, config=LOSS_CONFIG)


def usable_row(row):
    return bool(row['future_valid_mask'].any()) and bool(row['teacher_valid']) and not any(
        bool(row[k]) for k in ('teacher_stop', 'base_stop', 'no_vp_left'))


class EpisodeBlockSampler:
    """Shuffle bounded blocks of episodes, then all usable rows within each block.

    An episode is read once per epoch, rather than once per sampled decision.
    No background prefetch consumes RNG/state beyond the last committed batch.
    Block order and row order are pure functions of seed/epoch/block; checkpoints
    need only three cursors. The final partial batch is kept for every epoch.
    """
    def __init__(self, roots, *, seed=2, block_episodes=32, expected_split='train'):
        if block_episodes < 1:
            raise ValueError('block_episodes must be positive')
        self.seed, self.block_episodes = seed, block_episodes
        self.entries, self.provenance = [], []
        self.verified_paths = set()
        self.total_rows = 0
        seen, identity = set(), None
        for root in roots:
            root = Path(root).resolve()
            manifest = json.loads((root / 'dataset_manifest.json').read_text())
            provenance = json.loads((root / 'provenance.json').read_text())
            if manifest['status'] != 'validated' or manifest['format'] != FORMAT or manifest['feature_space'] != SPACE:
                raise ValueError('invalid dataset manifest contract')
            if sha256(root / 'provenance.json') != manifest['provenance_sha256']:
                raise ValueError('dataset provenance changed')
            if provenance['split'] != expected_split:
                raise ValueError('wrong dataset split')
            contract = {k: provenance.get(k) for k in ('stage1_sha256', 'assets', 'split', 'feature_space', 'context_contract', 'behavior')}
            if identity is not None and identity != contract:
                raise ValueError('incompatible data roots')
            identity = contract
            actual_ids = {str(r['episode_id']) for r in manifest['entries']}
            if 'episode_ids' in provenance and actual_ids != set(map(str, provenance['episode_ids'])):
                raise ValueError('incomplete episode coverage')
            if len(actual_ids) != len(manifest['entries']) or seen.intersection(actual_ids):
                raise ValueError('duplicate episodes')
            seen.update(actual_ids)
            for item in manifest['entries']:
                path = root / item['file']
                if path.stat().st_size != item['bytes']:
                    raise ValueError(f'dataset shard size changed: {path}')
                self.entries.append((str(path), item))
            self.total_rows += manifest['trainable_rows']
            self.provenance.append(dict(root=str(root), manifest_sha256=sha256(root/'dataset_manifest.json'),
                provenance_sha256=manifest['provenance_sha256'], provenance=provenance))
        parts = [p['provenance'] for p in self.provenance]
        if any('parts' in p for p in parts):
            expected_parts = parts[0]['parts']
            if any(p.get('parts') != expected_parts for p in parts) or {p.get('part') for p in parts} != set(range(expected_parts)):
                raise ValueError('all collection partitions are required')
        if not self.entries or self.total_rows < 1:
            raise ValueError('empty training dataset')
        self.epoch = self.block = self.offset = 0
        self._rows = None

    def _order(self):
        return np.random.RandomState(self.seed + self.epoch).permutation(len(self.entries))

    def _load_block(self):
        selected = self._order()[self.block*self.block_episodes:(self.block+1)*self.block_episodes]
        rows = []
        for index in selected:
            path, metadata = self.entries[int(index)]
            if path not in self.verified_paths:
                if sha256(path) != metadata['sha256']:
                    raise ValueError(f'dataset shard SHA mismatch: {path}')
                self.verified_paths.add(path)
            episode = load(path)
            if episode['episode_id'] != metadata['episode_id'] or len(episode['rows']) != metadata['rows']:
                raise ValueError('episode shard changed')
            valid = [dict(r, text_tokens=episode['text_tokens']) for r in episode['rows'] if usable_row(r)]
            if len(valid) != metadata['trainable_rows']:
                raise ValueError('usable row count changed')
            rows.extend(valid)
        # Local generator leaves Python/NumPy/Torch training RNG untouched.
        random.Random(f'{self.seed}:{self.epoch}:{self.block}').shuffle(rows)
        self._rows = rows

    def next_rows(self, batch_size, max_epochs):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if self.epoch >= max_epochs:
            return None
        rows = []
        while len(rows) < batch_size:
            if self._rows is None:
                self._load_block()
            count = min(batch_size-len(rows), len(self._rows)-self.offset)
            rows.extend(self._rows[self.offset:self.offset+count])
            self.offset += count
            if self.offset == len(self._rows):
                self.block += 1
                self.offset = 0
                self._rows = None
                if self.block*self.block_episodes >= len(self.entries):
                    self.epoch += 1
                    self.block = 0
                    return rows or (None if self.epoch >= max_epochs else self.next_rows(batch_size, max_epochs))
        return rows

    def state_dict(self):
        return dict(epoch=self.epoch, block=self.block, offset=self.offset,
                    seed=self.seed, block_episodes=self.block_episodes)

    def load_state_dict(self, state):
        if state['seed'] != self.seed or state['block_episodes'] != self.block_episodes:
            raise ValueError('sampler configuration changed')
        self.epoch, self.block, self.offset = (int(state[k]) for k in ('epoch', 'block', 'offset'))
        if min(self.epoch, self.block, self.offset) < 0 or self.block*self.block_episodes >= len(self.entries):
            raise ValueError('invalid sampler cursor')
        self._rows = None
        self._load_block()
        if self.offset > len(self._rows):
            raise ValueError('sampler row cursor out of bounds')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def save_training_checkpoint(output, head, optimizer, scaler, sampler, step, contract):
    output = Path(output)
    common = dict(global_step=step, model_config=MODEL_CONFIG, loss_config=LOSS_CONFIG.to_dict(),
        contract=contract, future_head_state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()})
    atomic_torch_save(output/f'head_step_{step:06d}.pt', dict(common, format_version='stage2-e24-head-v1', resume_forbidden=True))
    full = dict(common, format_version='stage2-e24-training-v1', optimizer=optimizer.state_dict(),
        scheduler=None, scaler=scaler.state_dict() if scaler is not None else None,
        sampler=sampler.state_dict(), rng=rng_state())
    destination = output/f'state_step_{step:06d}.pt'
    atomic_torch_save(destination, full)
    atomic_json(output/'latest.json', dict(global_step=step, state=destination.name))
    states = sorted(output.glob('state_step_*.pt'))
    for old in states[:-3]:
        if int(old.stem.split('_')[-1]) % 2000:
            old.unlink()
    return destination


def restore_training_checkpoint(path, head, optimizer, scaler, sampler, contract):
    state = load(path)
    if state.get('format_version') != 'stage2-e24-training-v1' or state.get('resume_forbidden'):
        raise ValueError('checkpoint cannot resume training')
    if state['contract'] != contract or state['model_config'] != MODEL_CONFIG or state['loss_config'] != LOSS_CONFIG.to_dict():
        raise ValueError('resume data/model/config/source contract differs')
    if state['scheduler'] is not None or (state['scaler'] is None) != (scaler is None):
        raise ValueError('resume scheduler/scaler differs')
    head.load_state_dict(state['future_head_state_dict'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scaler is not None:
        scaler.load_state_dict(state['scaler'])
    sampler.load_state_dict(state['sampler'])
    restore_rng(state['rng'])
    return int(state['global_step'])


def model_fingerprint(head):
    digest = hashlib.sha256()
    for name, tensor in sorted(head.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
