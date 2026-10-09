"""Independent, provenance-checked full-depth progressive episode schema.

Storage depth index 0 is physical successor d=1. Padding may contain arbitrary
values: validation only inspects evidence declared valid by the three masks.
"""
import hashlib
import json
import math
from pathlib import Path
import torch
from torch.utils.data import Dataset
from .stage2_data import atomic_json, sha256

from .progressive_contract import FORMAT, SPACE, GEOMETRY, PREDICTION_CONTRACT


def validate_provenance(p):
    if p.get('format') != FORMAT or p.get('feature_space') != SPACE:
        raise ValueError('progressive schema/feature space mismatch')
    if not isinstance(p.get('distance_scale'), (int,float)) or not math.isfinite(p['distance_scale']) or p['distance_scale'] <= 0:
        raise ValueError('geometry requires a finite positive distance_scale')
    if p.get('geometry_definition') != GEOMETRY:
        raise ValueError('progressive geometry definition mismatch')
    if type(p.get('max_future_depth')) is not int or p['max_future_depth'] not in (1,2,3):
        raise ValueError('invalid physical depth')
    for key in ('stage1_sha256', 'assets', 'context_contract', 'behavior', 'split'):
        if not p.get(key):
            raise ValueError(f'missing source provenance: {key}')
    if not isinstance(p.get('prediction_contract'),dict) or set(p['prediction_contract']) != set(PREDICTION_CONTRACT):
        raise ValueError('missing prediction runtime contract')
    digest = p['stage1_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid stage1 SHA256')
    if not isinstance(p['assets'], dict) or not all(
            isinstance(v, dict) and isinstance(v.get('sha256'), str) and len(v['sha256']) == 64 and all(c in '0123456789abcdef' for c in v['sha256']) for v in p['assets'].values()):
        raise ValueError('assets require explicit SHA256 records')
    f = p.get('feature_contract', {})
    if set(f) != {'token_count', 'feature_dim', 'owner_dim', 'text_dim'} or any(
            not isinstance(v, int) or v < 1 for v in f.values()) or f['token_count'] < 2:
        raise ValueError('feature contract requires CLS and patch dimensions')


def validate_row(row, text, provenance):
    validate_provenance(provenance)
    f = provenance['feature_contract']; d = provenance['max_future_depth']
    n = len(row['ghost_ids']); k = len(row['topk_base_indices'])
    shapes = dict(base_logits=(n,), owner_embeddings=(k, f['owner_dim']),
        candidate_q0_geometry=(k, 3), future_tokens=(k,d,f['token_count'],f['feature_dim']),
        future_geometry=(k,d,5), future_valid_mask=(k,d), future_token_mask=(k,d,f['token_count']),
        candidate_present_mask=(k,), future_terminal_mask=(k,d), future_queried_mask=(k,d),
        target_positions=(k,d,3), target_yaws=(k,d), horizons=(k,d))
    for name, shape in shapes.items():
        if not torch.is_tensor(row[name]) or tuple(row[name].shape) != shape:
            raise ValueError(f'{name}: expected {shape}')
        if name.endswith('_mask') and row[name].dtype != torch.bool:
            raise ValueError(f'{name}: expected boolean mask')
    if row['topk_base_indices'].dtype != torch.long:
        raise ValueError('topk indices must be int64')
    if len(set(row['ghost_ids'])) != n:
        raise ValueError('duplicate candidate IDs')
    present = row['candidate_present_mask']; valid = row['future_valid_mask']
    tokens = row['future_token_mask']; terminal = row['future_terminal_mask']; queried = row['future_queried_mask']
    ix = row['topk_base_indices'][present].tolist()
    if len(set(ix)) != len(ix) or any(i < 0 or i >= n for i in ix):
        raise ValueError('invalid topk mapping')
    if bool((row['topk_base_indices'][~present] != -1).any()):
        raise ValueError('padded index must be -1')
    if bool((valid & (~present[:,None] | ~queried | terminal)).any()):
        raise ValueError('invalid future availability')
    if bool((tokens & ~valid[:,:,None]).any()) or bool((valid & (~tokens[:,:,0] | ~tokens[:,:,1:].any(-1))).any()):
        raise ValueError('valid future requires valid CLS and patch')
    for depth in range(d):
        previous_terminal = terminal[:,depth-1] if depth else ~present
        if bool((previous_terminal & (~terminal[:,depth] | queried[:,depth] | valid[:,depth])).any()):
            raise ValueError('terminated branch resumed')
        if bool((present & ~previous_terminal & ~queried[:,depth] & ~terminal[:,depth]).any()):
            raise ValueError('collection must fully expand; unqueried is not terminal')
        if bool((queried[:,depth] & ~valid[:,depth] & ~terminal[:,depth]).any()):
            raise ValueError('failed query must record terminal')
    for name, mask in [('owner_embeddings',present),('candidate_q0_geometry',present),
                       ('future_tokens',tokens),('future_geometry',valid),('target_positions',valid),
                       ('target_yaws',valid),('horizons',valid)]:
        if not torch.isfinite(row[name][mask]).all():
            raise ValueError(f'{name}: non-finite valid evidence')
    if not torch.isfinite(row['base_logits']).all() or row['future_tokens'].dtype != torch.float16:
        raise ValueError('base logits must be finite; stored tokens must be fp16')
    if text.ndim != 2 or text.shape[-1] != f['text_dim'] or not torch.isfinite(text).all():
        raise ValueError('invalid language features')
    for name in ('request_ids', 'termination_reasons'):
        if len(row[name]) != k or any(len(v) != d for v in row[name]):
            raise ValueError(f'{name}: metadata shape mismatch')
    if len(row['q0_metadata']) != k:
        raise ValueError('missing q0 context metadata')
    for slot in range(k):
        if present[slot] and not row['q0_metadata'][slot]:
            raise ValueError('missing q0 source')
        for depth in range(d):
            if valid[slot,depth] and not row['request_ids'][slot][depth]:
                raise ValueError('missing reproducible request identity')
            if terminal[slot,depth] and not row['termination_reasons'][slot][depth]:
                raise ValueError('missing termination reason')
    teacher = int(row['teacher_base_index']); rank = int(row['teacher_rank_in_topk'])
    # Stored slots may contain padding, so resolve rank using original indices.
    expected = next((i for i in range(k) if present[i] and int(row['topk_base_indices'][i]) == teacher), -1)
    if teacher < -1 or teacher >= n or rank != expected:
        raise ValueError('teacher mapping mismatch')
    for key in ('teacher_valid','teacher_stop','base_stop','no_vp_left'):
        if not isinstance(row[key], bool):
            raise ValueError(f'{key}: expected bool')


class ProgressiveEpisodeWriter:
    def __init__(self, root, provenance):
        validate_provenance(provenance)
        self.root=Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.provenance=provenance; self.pending={}
        path=self.root/'provenance.json'
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise ValueError('existing provenance differs')
        atomic_json(path,provenance)

    def append(self, episode, scene, text, row):
        text=text.detach().cpu().half().contiguous()
        validate_row(row,text,self.provenance)
        item=self.pending.setdefault(str(episode),dict(format=FORMAT,episode_id=str(episode),scene_id=str(scene),text_tokens=text,rows=[]))
        if row['step'] != len(item['rows']):
            raise ValueError('non-sequential decision step')
        item['rows'].append(row)

    def complete(self, episode, metrics):
        item=self.pending[str(episode)]
        path=self.root/(hashlib.sha256(str(episode).encode()).hexdigest()[:24]+'.pt')
        if path.exists(): raise FileExistsError(path)
        tmp=path.with_suffix('.pt.tmp'); torch.save(item,tmp); tmp.replace(path)
        record=dict(episode_id=str(episode),file=path.name,rows=len(item['rows']),sha256=sha256(path),bytes=path.stat().st_size,metrics=metrics)
        atomic_json(path.with_suffix('.json'),record)
        del self.pending[str(episode)]
        return record


def validate_dataset(root):
    root=Path(root); p=json.loads((root/'provenance.json').read_text()); validate_provenance(p)
    entries=[]; seen=set()
    for path in sorted(root.glob('*.pt')):
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['sha256'] != sha256(path) or meta['bytes'] != path.stat().st_size or meta['file'] != path.name:
            raise ValueError('progressive shard SHA/size mismatch')
        obj=torch.load(path,map_location='cpu',weights_only=False)
        if obj['format'] != FORMAT or obj['episode_id'] != meta['episode_id'] or obj['episode_id'] in seen or len(obj['rows']) != meta['rows']:
            raise ValueError('progressive shard identity/schema mismatch')
        seen.add(obj['episode_id'])
        for step,row in enumerate(obj['rows']):
            validate_row(row,obj['text_tokens'],p)
            if row['step'] != step: raise ValueError('step sequence mismatch')
        entries.append(meta)
    if not entries: raise ValueError('empty progressive dataset')
    if 'episode_ids' in p and seen != set(map(str,p['episode_ids'])):
        raise ValueError('incomplete episode coverage')
    manifest=dict(format=FORMAT,status='validated',provenance_sha256=sha256(root/'provenance.json'),entries=entries,rows=sum(x['rows'] for x in entries))
    atomic_json(root/'dataset_manifest.json',manifest)
    return manifest


class ProgressiveDataset(Dataset):
    def __init__(self, root):
        self.root=Path(root)
        # Verify shard SHA and row contracts, not merely a stale validation stamp.
        self.manifest=validate_dataset(root)
        self.provenance=json.loads((self.root/'provenance.json').read_text())
        self.index=[(r['file'],i) for r in self.manifest['entries'] for i in range(r['rows'])]
        self.cached_name=None; self.cached=None
    def __len__(self): return len(self.index)
    def __getitem__(self,index):
        name,i=self.index[index]
        if name != self.cached_name:
            self.cached=torch.load(self.root/name,map_location='cpu',weights_only=False); self.cached_name=name
        return dict(self.cached['rows'][i],text_tokens=self.cached['text_tokens'])


def collate_progressive(rows, provenance):
    if not rows: raise ValueError('empty batch')
    for row in rows: validate_row(row,row['text_tokens'],provenance)
    b=len(rows); k=max(len(r['topk_base_indices']) for r in rows); g=max(1,max(len(r['ghost_ids']) for r in rows)); t=max(1,max(len(r['text_tokens']) for r in rows))
    out={}
    names=('owner_embeddings','candidate_q0_geometry','future_tokens','future_geometry','future_valid_mask','future_token_mask','candidate_present_mask','future_terminal_mask','future_queried_mask','topk_base_indices')
    for name in names:
        sample=rows[0][name]
        dtype=torch.float32 if sample.is_floating_point() else sample.dtype
        out[name]=torch.full((b,k,*sample.shape[1:]),-1 if name=='topk_base_indices' else 0,dtype=dtype)
    out.update(base_logits=torch.zeros(b,g),ghost_valid_mask=torch.zeros(b,g,dtype=torch.bool),text_tokens=torch.zeros(b,t,provenance['feature_contract']['text_dim']),text_token_mask=torch.zeros(b,t,dtype=torch.bool))
    for i,r in enumerate(rows):
        q=len(r['topk_base_indices']); n=len(r['ghost_ids']); l=len(r['text_tokens'])
        for name in names: out[name][i,:q]=r[name]
        out['base_logits'][i,:n]=r['base_logits']; out['ghost_valid_mask'][i,:n]=True
        out['text_tokens'][i,:l]=r['text_tokens']; out['text_token_mask'][i,:l]=True
    for name in ('teacher_valid','teacher_stop','base_stop','no_vp_left','teacher_base_index','teacher_rank_in_topk'):
        out[name]=torch.tensor([r[name] for r in rows],dtype=torch.long if 'index' in name or 'rank' in name else torch.bool)
    return out
