import torch
import pytest
from vlnce_baselines.nwm.active_lookahead.stage2_evaluation import (
    decision_records, build_report, selection_key, paired_episode_bootstrap)


def batch():
    return dict(base_logits=torch.tensor([[1.,.8],[1.,.8],[1.,.8],[1.,.8],[1.,.8]]),
        ghost_valid_mask=torch.ones(5,2,dtype=torch.bool), topk_base_indices=torch.tensor([[0,1]]*5),
        topk_valid_mask=torch.tensor([[True,True],[True,True],[False,False],[True,True],[True,True]]),
        base_stop=torch.tensor([False,False,False,True,False]), no_vp_left=torch.zeros(5,dtype=torch.bool),
        teacher_valid=torch.tensor([True,True,True,True,False]), teacher_stop=torch.zeros(5,dtype=torch.bool),
        teacher_base_index=torch.tensor([1,0,1,1,1]), teacher_rank_in_topk=torch.tensor([1,0,1,1,1]),
        episode_id=['a','a','b','b','c'],scene_id=['s']*5)


def test_primary_denominator_includes_missing_future_and_excludes_stop_invalid():
    rows=decision_records(batch(),torch.tensor([[0.,.5]]*5))
    report=build_report(rows,100)
    assert report['primary_rows']==3
    assert (report['fixes'],report['harms'],report['net_fixes'])==(1,1,0)
    assert report['base_accuracy']==report['accuracy']==1/3
    assert report['stop_consistent_rows']==5
    assert rows[2]['action_changed']==0
    assert rows[3]['action_changed']==0
    assert report['strata']['teacher_future']['False']['primary_rows']==1


def test_zero_delta_and_zero_gain_keep_every_action():
    for delta,gain in ((torch.zeros(5,2),1.),(torch.ones(5,2),0.)):
        report=build_report(decision_records(batch(),delta,gain),20)
        assert report['net_fixes']==0
        assert report['all_action_changes']==0
        assert report['bootstrap']['ci95']==[0.,0.]


def test_clamp_and_gain_contract():
    rows=decision_records(batch(),torch.tensor([[-2.,2.]]*5),1.5)
    assert rows[0]['residual_abs_sum']==2.
    with pytest.raises(ValueError,match='grid'): decision_records(batch(),torch.zeros(5,2),1.206)
    with pytest.raises(ValueError,match='nonfinite'): decision_records(batch(),torch.full((5,2),float('nan')))


def test_selection_obeys_plan_ties():
    base=dict(gain=1.,net_fixes=2,harms=3,classification_loss=1.,global_step=500)
    rows=[base,dict(base,harms=2),dict(base,harms=2,classification_loss=.8),
          dict(base,harms=2,classification_loss=.8,global_step=250)]
    assert sorted(rows,key=selection_key)[0]['global_step']==250
    assert selection_key(dict(base,net_fixes=3))<selection_key(rows[-1])
    with pytest.raises(ValueError): selection_key(dict(base,gain=.5))


def test_bootstrap_clusters_steps_within_episode_and_is_reproducible():
    rows=decision_records(batch(),torch.tensor([[0.,.5]]*5))
    first=paired_episode_bootstrap(rows,100,2)
    assert first==paired_episode_bootstrap(rows,100,2)
    assert first['episodes']==3
    # The fix and harm belong to one episode and cancel in every paired resample.
    assert first['ci95']==[0.,0.]


def test_empty_ghost_row_is_safe():
    b=batch();b['ghost_valid_mask'][0]=False;b['topk_valid_mask'][0]=False
    rows=decision_records(b,torch.zeros(5,2))
    assert not rows[0]['primary']
    assert rows[0]['base_entropy']==0.
    assert rows[0]['stop_consistent']==1


def test_checkpoint_asset_contract_refuses_mixed_base_and_dev_training():
    from scripts.evaluate_stage2_e24 import validate_checkpoint
    provenance=dict(split='train',stage1_sha256='abc',assets={'wm':'def'},feature_space='raw',
                    context_contract='panorama',behavior='argmax')
    checkpoint=dict(format_version='stage2-e24-head-v1',contract={'dataset':[{'provenance':provenance}]})
    dev=dict(provenance,split='val_unseen')
    validate_checkpoint(checkpoint,[dev])
    with pytest.raises(ValueError,match='contract'):
        validate_checkpoint(checkpoint,[dict(dev,stage1_sha256='different')])
    checkpoint['format_version']='unknown'
    with pytest.raises(ValueError,match='format'):
        validate_checkpoint(checkpoint,[dev])
