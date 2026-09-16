"""Atomic episode shards and bounded-memory, train-ready E24 decisions.

Feature space matches native E24: raw CLS + normalized patch tokens.
No optimizer or training entry point is implemented in this collection phase.
"""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import torch
from torch.utils.data import Dataset

FORMAT = 'etpr1-stage2-predicted-episode-v1'
SPACE = 'raw_cls+normalized_patch_fp16'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def validate_row(row, text):
    n = len(row['ghost_ids']); k = len(row['topk_base_indices'])
    if len(set(row['ghost_ids'])) != n or not 0 <= k <= 5:
        raise ValueError('invalid ghost/topk identities')
    expected = {'base_logits': (n,), 'owner_embeddings': (k,768),
                'future_tokens': (k,257,768), 'candidate_q0_geometry': (k,3),
                'future_valid_mask': (k,), 'q1_conditions': (k,4)}
    for name, shape in expected.items():
        value = row[name]
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f'{name}: expected {shape}')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'{name}: non-finite')
    indices = row['topk_base_indices'].tolist()
    if len(set(indices)) != k or any(i < 0 or i >= n for i in indices):
        raise ValueError('topk index invalid')
    teacher = int(row['teacher_base_index']); rank = int(row['teacher_rank_in_topk'])
    if teacher >= n or teacher < -1 or rank < -1 or rank >= k:
        raise ValueError('teacher index out of range')
    if rank >= 0 and indices[rank] != teacher: raise ValueError('teacher mapping inconsistent')
    if text.ndim != 2 or text.shape[-1] != 768 or not torch.isfinite(text).all():
        raise ValueError('invalid language features')
    if row['future_tokens'].dtype != torch.float16: raise ValueError('future must be FP16')
    if row['future_valid_mask'].dtype != torch.bool: raise ValueError('future mask must be bool')


class EpisodeWriter:
    def __init__(self, root, provenance):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.provenance = provenance
        identity = self.root / 'provenance.json'
        if identity.exists() and json.loads(identity.read_text()) != provenance:
            raise ValueError('existing dataset provenance differs')
        atomic_json(identity, provenance)
        self.pending = {}

    def append(self, episode, scene, text, row):
        text = text.detach().cpu().half().contiguous()
        validate_row(row, text)
        item = self.pending.setdefault(episode, dict(format=FORMAT, feature_space=SPACE,
            episode_id=episode, scene_id=scene, text_tokens=text, rows=[]))
        if row['step'] != len(item['rows']): raise ValueError('non-sequential decision step')
        item['rows'].append(row)

    def complete(self, episode, metrics):
        item = self.pending.pop(episode)
        name = hashlib.sha256(episode.encode()).hexdigest()[:24] + '.pt'
        path = self.root / name
        if path.exists(): raise FileExistsError(f'duplicate episode {episode}')
        tmp = path.with_suffix('.pt.tmp'); torch.save(item, tmp); tmp.replace(path)
        record = dict(episode_id=episode, scene_id=item['scene_id'], file=name,
            rows=len(item['rows']), bytes=path.stat().st_size, sha256=sha256(path),
            future_slots=sum(len(r['future_valid_mask']) for r in item['rows']),
            future_valid=sum(int(r['future_valid_mask'].sum()) for r in item['rows']),
            trainable_rows=sum(bool(r['future_valid_mask'].any()) and r['teacher_valid']
                and not r['teacher_stop'] and not r['base_stop'] and not r['no_vp_left'] for r in item['rows']),
            metrics=metrics)
        atomic_json(path.with_suffix('.json'), record)
        return record


def validate_dataset(root, expected_ids=None):
    root = Path(root); entries = []; ids = set(); failures = []
    for meta in sorted(root.glob('*.json')):
        if meta.name in ('provenance.json','dataset_manifest.json','freeze_report.json'): continue
        r = json.loads(meta.read_text())
        if 'file' not in r: continue
        path = root / r['file']
        if sha256(path) != r['sha256']: raise ValueError(f'SHA mismatch: {path}')
        obj = load(path)
        if obj['format'] != FORMAT or obj['feature_space'] != SPACE: raise ValueError('schema/space mismatch')
        if obj['episode_id'] != r['episode_id'] or r['episode_id'] in ids: raise ValueError('episode mismatch/duplicate')
        if len(obj['rows']) != r['rows']: raise ValueError('row count mismatch')
        for step, row in enumerate(obj['rows']):
            validate_row(row, obj['text_tokens'])
            if row['step'] != step: raise ValueError('step order mismatch')
        ids.add(r['episode_id']); entries.append(r)
    if expected_ids is not None and ids != set(map(str, expected_ids)):
        raise ValueError(f'episode coverage mismatch: missing={len(set(map(str,expected_ids))-ids)}, extra={len(ids-set(map(str,expected_ids)))}')
    if not entries: raise ValueError('empty dataset')
    manifest = dict(format=FORMAT, feature_space=SPACE, status='validated', episodes=len(ids),
                    rows=sum(r['rows'] for r in entries), trainable_rows=sum(r['trainable_rows'] for r in entries),
                    future_slots=sum(r['future_slots'] for r in entries), future_valid=sum(r['future_valid'] for r in entries),
                    bytes=sum(r['bytes'] for r in entries), provenance_sha256=sha256(root/'provenance.json'), entries=entries)
    if not manifest['trainable_rows']: raise ValueError('no trainable rows')
    atomic_json(root/'dataset_manifest.json',manifest)
    return manifest


class Stage2Dataset(Dataset):
    def __init__(self, root, cache_size=2):
        self.root=Path(root); manifest=json.loads((self.root/'dataset_manifest.json').read_text())
        if manifest['status']!='validated': raise ValueError('unvalidated dataset')
        self.index=[(r['file'],i) for r in manifest['entries'] for i in range(r['rows'])]
        self.cache=OrderedDict(); self.cache_size=max(1,cache_size)
    def __len__(self): return len(self.index)
    def __getitem__(self,index):
        name,i=self.index[index]
        if name not in self.cache:
            self.cache[name]=load(self.root/name)
            while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        self.cache.move_to_end(name); episode=self.cache[name]
        return dict(episode['rows'][i],text_tokens=episode['text_tokens'])


def collate_stage2(rows):
    b=len(rows); k=5; g=max(1,max(len(r['ghost_ids']) for r in rows)); t=max(len(r['text_tokens']) for r in rows)
    out=dict(owner_embeddings=torch.zeros(b,k,768), future_tokens=torch.zeros(b,k,257,768),
        candidate_q0_geometry=torch.zeros(b,k,3), q1_conditions=torch.zeros(b,k,4),
        text_tokens=torch.zeros(b,t,768), text_token_mask=torch.zeros(b,t,dtype=torch.bool),
        base_logits=torch.zeros(b,g), ghost_valid_mask=torch.zeros(b,g,dtype=torch.bool),
        topk_base_indices=torch.full((b,k),-1,dtype=torch.long), topk_valid_mask=torch.zeros(b,k,dtype=torch.bool))
    for key in ('teacher_valid','teacher_stop','no_vp_left','base_stop'):
        out[key]=torch.tensor([r[key] for r in rows],dtype=torch.bool)
    for key in ('teacher_base_index','teacher_rank_in_topk'):
        out[key]=torch.tensor([r[key] for r in rows],dtype=torch.long)
    for i,r in enumerate(rows):
        n=len(r['ghost_ids']); q=len(r['topk_base_indices']); l=len(r['text_tokens'])
        for key in ('owner_embeddings','future_tokens','candidate_q0_geometry','q1_conditions','topk_base_indices'):
            out[key][i,:q]=r[key]
        out['topk_valid_mask'][i,:q]=r['future_valid_mask']
        out['base_logits'][i,:n]=r['base_logits'];out['ghost_valid_mask'][i,:n]=True
        out['text_tokens'][i,:l]=r['text_tokens'];out['text_token_mask'][i,:l]=True
    return out
