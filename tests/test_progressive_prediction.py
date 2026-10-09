"""Deterministic CPU mocks, not real world-model/navigation evaluation."""
from collections import Counter
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead import progressive_prediction as m


def fixture(monkeypatch, *, horizon=2., stop_second=False):
    requests, noises, patches = [], [], []
    config=NS(metric_waypoint_spacing=.25,min_horizon=1.,max_horizon=64.)
    def build(req, device):
        requests.extend(req)
        return NS(n=len(req))
    def predict(batch, initial_noise):
        noises.append(initial_noise.clone())
        value=float(len(noises)+1)
        return NS(pred_cls=torch.full((batch.n,768),value),
                  pred_tokens=torch.full((batch.n,257,768),value),
                  pred_latent=torch.full((batch.n,768,16,16),value))
    def cwp(patch):
        patches.append(patch.clone())
        return patch.shape[0]
    def decode(n, **kwargs):
        return [NS(valid=True,pred_none=stop_second and len(patches)==2,local_angle_deg=90.,distance_m=.5) for _ in range(n)]
    monkeypatch.setattr(m,'decode_dino_cwp_top1',decode)
    rt=NS(generator=torch.Generator().manual_seed(123),adapter=NS(config=config,build_raenwm_latent_batch=build),_predict_batch=predict)
    snap=dict(target_position=np.array([0.,0.,-1.]),target_yaw=0.,horizon=horizon,
              context_latents=torch.ones(4,257,768),source_position=np.zeros(3),source_yaw=0.)
    graph=NS(ghost_mean_pos={'g1':np.array([0.,0.,-1.])},stage2_q0={'g1':dict(snapshot=snap,patch=torch.ones(768,16,16),view=0,forward=1.,step=1)})
    tr=NS(device=torch.device('cpu'),gmaps=[graph],max_len=20,
          envs=NS(current_episodes=lambda:[NS(episode_id='ep',scene_id='scene')]))
    c=NS(runtime=rt,trainer=tr,cwp=cwp,counts=Counter(),cfg=NS(seed=23),versions={})
    p=m.ProgressivePrediction(c)
    inputs=dict(gmap_vp_ids=[[None,'g1']])
    outputs=dict(global_logits=torch.tensor([[0.,1.]]),gmap_embeds=torch.ones(1,2,8))
    d=p.initialize_decision(inputs,outputs,torch.ones(1,3,8),torch.ones(1,3,dtype=torch.bool),[False],4)
    return p,d,requests,noises,patches,graph


def test_lazy_fixed_context_horizon_geometry_rng(monkeypatch):
    p,d,requests,noises,patches,graph=fixture(monkeypatch)
    original=graph.stage2_q0['g1']['patch'].clone()
    global_rng=torch.get_rng_state().clone(); runtime_rng=p.collector.runtime.generator.get_state().clone()
    assert requests==[]
    first=p.predict_next_depth(d)[0]
    assert len(requests)==1 and first['future_valid_mask'].tolist()==[True]
    assert first['future_geometry'][0].tolist()==pytest.approx([.5,0.,1.,0.,.5],abs=1e-6)
    p.predict_next_depth(d)
    assert len(requests)==2
    assert requests[0].snapshot is requests[1].snapshot
    assert torch.equal(requests[0].snapshot.context_latents,torch.ones(4,257,768))
    assert np.allclose(requests[0].target_position,[-.5,0,-1])
    assert np.allclose(requests[1].target_position,[-.5,0,-.5])
    assert [r.horizon_override for r in requests]==[4.,6.]
    assert patches[0].mean()==1 and patches[1].mean()==2
    assert torch.equal(global_rng,torch.get_rng_state())
    assert torch.equal(runtime_rng,p.collector.runtime.generator.get_state())
    assert torch.equal(original,graph.stage2_q0['g1']['patch'])
    row=p.finalize_decision(d)[0]
    assert row['future_tokens'].shape==(1,2,257,768)
    assert row['future_queried_mask'].tolist()==[[True,True]]
    with pytest.raises(ValueError):p.predict_next_depth(d)


@pytest.mark.parametrize('horizon,stop_second,expected',[(63.,False,0),(2.,True,1)])
def test_terminal_never_requeries(monkeypatch,horizon,stop_second,expected):
    p,d,requests,*_=fixture(monkeypatch,horizon=horizon,stop_second=stop_second)
    p.predict_next_depth(d);p.predict_next_depth(d)
    row=p.finalize_decision(d)[0]
    assert len(requests)==expected
    assert row['future_valid_mask'].sum()==expected
    assert row['future_terminal_mask'][0,-1]
    if not expected:assert not row['future_queried_mask'][0,1]


def test_stopped_row_has_no_later_world_queries(monkeypatch):
    p,d,requests,*_=fixture(monkeypatch)
    p.predict_next_depth(d)
    layer=p.predict_next_depth(d,active_rows=[])[0]
    assert len(requests)==1 and layer['unqueried_mask'].all() and not layer['terminal_mask'].any()
    assert not layer['future_valid_mask'].any()


def test_identity_and_new_decision(monkeypatch):
    a=m.prediction_identity(1,'scene','episode',3,'stable_id',1,1)
    assert a==m.prediction_identity(1,'scene','episode',3,'stable_id',1,1)
    for index,value in [(1,'other_scene'),(3,4),(4,'other_id'),(5,2),(6,2)]:
        args=[1,'scene','episode',3,'stable_id',1,1];args[index]=value
        assert a!=m.prediction_identity(*args)
    p,d,*_=fixture(monkeypatch);p.predict_next_depth(d)
    _,fresh,*_=fixture(monkeypatch)
    assert fresh.depth==0 and fresh.layers==[] and fresh.branches[0]['path']==0


def test_runtime_rng_violation_restored_and_rejected(monkeypatch):
    p,d,*_=fixture(monkeypatch)
    rt=p.collector.runtime;before=rt.generator.get_state().clone();old=rt._predict_batch
    def corrupt(*args,**kwargs):
        torch.randn(1,generator=rt.generator)
        return old(*args,**kwargs)
    rt._predict_batch=corrupt
    with pytest.raises(RuntimeError,match='stage1 RNG'):p.predict_next_depth(d)
    assert torch.equal(before,rt.generator.get_state())


def test_full_collection_schema(monkeypatch):
    from vlnce_baselines.nwm.active_lookahead.progressive_data import validate_row, FORMAT, SPACE, GEOMETRY, PREDICTION_CONTRACT
    p,d,*_=fixture(monkeypatch)
    p.predict_next_depth(d);p.predict_next_depth(d)
    row=p.finalize_decision(d)[0]
    row.update(teacher_base_index=0,teacher_rank_in_topk=0,teacher_valid=True,teacher_stop=False)
    provenance=dict(prediction_contract=dict(PREDICTION_CONTRACT),format=FORMAT,feature_space=SPACE,geometry_definition=GEOMETRY,distance_scale=1.,max_future_depth=2,
        stage1_sha256='a'*64,assets={'cwp':{'sha256':'b'*64}},context_contract='real_q0',behavior='frozen',split='mock',
        feature_contract=dict(token_count=257,feature_dim=768,owner_dim=8,text_dim=8))
    validate_row(row,row['text_tokens'],provenance)


def test_noise_independent_of_candidate_and_batch_order(monkeypatch):
    import copy
    p,d,requests,noises,*_=fixture(monkeypatch)
    first=d.branches[0]
    other=dict(first,ghost='g2',slot=1)
    d.branches.append(other)
    row=d.rows[0]
    row['topk_base_indices']=torch.tensor([0,1])
    before=copy.deepcopy(d)
    p.predict_next_depth(d)
    original={r.ghost_vp:n for r,n in zip(requests,torch.cat(noises))}
    requests.clear();noises.clear()
    before.branches.reverse()
    p.predict_next_depth(before)
    reordered={r.ghost_vp:n for r,n in zip(requests,torch.cat(noises))}
    assert all(torch.equal(original[k],reordered[k]) for k in original)


def test_real_controller_lazy_predictions_and_certified_full_argmax(monkeypatch):
    from dataclasses import replace
    from vlnce_baselines.nwm.active_lookahead.progressive_head import ProgressiveE24Head,ProgressiveHeadConfig
    from vlnce_baselines.nwm.active_lookahead.progressive_controller import ProgressiveController
    p,d,requests,noises,patches,graph=fixture(monkeypatch)
    # Real predictor + real controller + real neural scorer; only CWP/NWM mocked.
    d.rows[0].update(base_logits=torch.tensor([.5,0.]),topk_base_indices=torch.tensor([0,1]),
        owner_embeddings=torch.ones(2,768),candidate_q0_geometry=torch.zeros(2,3),
        candidate_present_mask=torch.ones(2,dtype=torch.bool),text_tokens=torch.ones(3,768))
    d.branches.append(dict(d.branches[0],ghost='g2',slot=1))
    import copy
    full_decision=copy.deepcopy(d)
    head=ProgressiveE24Head(ProgressiveHeadConfig(hidden_dim=8,num_attention_heads=2,num_queries=2,
        ffn_dim=16,fusion_layers=1,dropout=0.,budget_fractions=(.9,.1))).eval()
    before=graph.stage2_q0['g1']['patch'].clone()
    short=ProgressiveController(head,'certified').run(p,d)
    assert len(requests)==2 and short['diagnostics'][0]['executed_depth']==1
    assert short['diagnostics'][0]['stop_reason']=='certified'
    requests.clear();noises.clear();patches.clear()
    full=ProgressiveController(head,'none').run(p,full_decision)
    assert len(requests)==4 and full['diagnostics'][0]['executed_depth']==2
    assert short['scores'][0].argmax()==full['scores'][0].argmax()
    assert torch.equal(before,graph.stage2_q0['g1']['patch'])
