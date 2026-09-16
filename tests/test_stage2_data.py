import json
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead.stage2_data import EpisodeWriter, validate_dataset, Stage2Dataset, collate_stage2
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
