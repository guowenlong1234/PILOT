import json
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead.stage2_data import (COMPACT_STORAGE,
    STORAGE_PROVENANCE_KEY, EpisodeWriter, validate_dataset, Stage2Dataset,
    collate_stage2, load)
from vlnce_baselines.nwm.active_lookahead.offline_objective import offline_decision_aware_loss, OfflineDecisionLossConfig


def row():
    return dict(step=0,ghost_ids=['g0','g1'],topk_base_indices=torch.tensor([1,0]),
        base_logits=torch.tensor([1.,2.]),owner_embeddings=torch.ones(2,768).half(),
        future_tokens=torch.ones(2,257,768).half(),future_valid_mask=torch.tensor([True,False]),
        candidate_q0_geometry=torch.zeros(2,3),q1_conditions=torch.zeros(2,4),
        teacher_base_index=1,teacher_rank_in_topk=0,teacher_valid=True,teacher_stop=False,
        no_vp_left=False,base_stop=False)


def test_roundtrip_collate_loss_and_corruption(tmp_path):
    w=EpisodeWriter(tmp_path,{'split':'train'});w.append('a','scene',torch.ones(4,768),row());r=w.complete('a',{})
    manifest=validate_dataset(tmp_path,['a']);assert manifest['trainable_rows']==1
    d=Stage2Dataset(tmp_path);b=collate_stage2([d[0]])
    assert b['future_tokens'].shape==(1,5,257,768)
    assert b['topk_valid_mask'].tolist()==[[True,False,False,False,False]]
    loss=offline_decision_aware_loss(torch.zeros(1,5),b['teacher_rank_in_topk'],
        **{k:b[k] for k in ('topk_valid_mask','teacher_valid','teacher_stop','no_vp_left','base_stop','base_logits','ghost_valid_mask','topk_base_indices','teacher_base_index')},config=OfflineDecisionLossConfig())
    assert torch.isfinite(loss.loss)
    (tmp_path/r['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='SHA'):validate_dataset(tmp_path,['a'])


def test_identity_and_index_rejection(tmp_path):
    w=EpisodeWriter(tmp_path,{'split':'train'})
    with pytest.raises(ValueError,match='provenance'):EpisodeWriter(tmp_path,{'split':'val'})
    r=row();r['teacher_rank_in_topk']=1
    with pytest.raises(ValueError,match='mapping'):w.append('a','s',torch.ones(4,768),r)
    r=row();r['future_tokens'][0,0,0]=float('nan')
    with pytest.raises(ValueError,match='non-finite'):w.append('a','s',torch.ones(4,768),r)


def test_partial_episode_not_published(tmp_path):
    w=EpisodeWriter(tmp_path,{});w.append('a','s',torch.ones(4,768),row())
    assert not list(tmp_path.glob('*.pt'))
    with pytest.raises(ValueError,match='coverage'):validate_dataset(tmp_path,['a'])


def test_absent_future_and_stop_do_not_create_supervision(tmp_path):
    r=row();r['future_valid_mask'].zero_();r['base_stop']=True
    w=EpisodeWriter(tmp_path,{});w.append('a','s',torch.ones(4,768),r);w.complete('a',{})
    with pytest.raises(ValueError,match='no trainable'):validate_dataset(tmp_path,['a'])
    b=collate_stage2([dict(r,text_tokens=torch.ones(4,768))])
    loss=offline_decision_aware_loss(torch.zeros(1,5),b['teacher_rank_in_topk'],
        **{k:b[k] for k in ('topk_valid_mask','teacher_valid','teacher_stop','no_vp_left','base_stop','base_logits','ghost_valid_mask','topk_base_indices','teacher_base_index')},config=OfflineDecisionLossConfig())
    assert loss.loss.item()==0


def test_multi_root_rejects_train_dev_mix(tmp_path):
    roots=[]
    for split in ('train','val_unseen'):
        root=tmp_path/split;roots.append(root)
        w=EpisodeWriter(root,{'split':split});w.append(split,'s',torch.ones(4,768),row());w.complete(split,{})
        validate_dataset(root,[split])
    with pytest.raises(ValueError,match='incompatible'):Stage2Dataset(roots)


def test_coverage_metadata_is_verified_from_tensors(tmp_path):
    w=EpisodeWriter(tmp_path,{});w.append('a','s',torch.ones(4,768),row());r=w.complete('a',{})
    meta=(tmp_path/r['file']).with_suffix('.json');r['future_valid']=100;meta.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='coverage metadata'):validate_dataset(tmp_path,['a'])


def test_compact_storage_restores_exact_full_future_tensor(tmp_path):
    original = row()
    original['future_tokens'][0].copy_(
        torch.arange(257 * 768).reshape(257, 768).remainder(997).half())
    original['future_tokens'][1].zero_()
    provenance = {'split':'train', STORAGE_PROVENANCE_KEY:COMPACT_STORAGE}
    writer = EpisodeWriter(tmp_path, provenance)
    writer.append('a', 'scene', torch.ones(4,768), original)
    record = writer.complete('a', {})

    raw = torch.load(tmp_path/record['file'], map_location='cpu', weights_only=False)
    assert raw['storage_schema'] == COMPACT_STORAGE
    assert 'future_tokens' not in raw['rows'][0]
    assert raw['rows'][0]['valid_future_tokens'].shape == (1,257,768)
    restored = load(tmp_path/record['file'])
    assert torch.equal(restored['rows'][0]['future_tokens'], original['future_tokens'])
    assert validate_dataset(tmp_path, ['a'])['storage_schema'] == COMPACT_STORAGE
    assert torch.equal(Stage2Dataset(tmp_path)[0]['future_tokens'], original['future_tokens'])


def test_compact_storage_refuses_nonzero_invalid_future(tmp_path):
    bad = row()
    bad['future_tokens'][1, 0, 0] = 1
    writer = EpisodeWriter(tmp_path, {STORAGE_PROVENANCE_KEY:COMPACT_STORAGE})
    with pytest.raises(ValueError, match='exactly zero'):
        writer.append('a', 'scene', torch.ones(4,768), bad)
    assert not list(tmp_path.glob('*.pt'))


def test_load_rejects_malformed_compact_payload(tmp_path):
    payload = dict(format='etpr1-stage2-predicted-episode-v1',
        storage_schema=COMPACT_STORAGE, rows=[dict(
            future_valid_mask=torch.tensor([True, False]),
            valid_future_tokens=torch.zeros(2,257,768,dtype=torch.float16))])
    path = tmp_path/'bad.pt'; torch.save(payload, path)
    with pytest.raises(ValueError, match='compact future tokens'):
        load(path)
