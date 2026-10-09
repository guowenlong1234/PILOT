from collections import Counter
from types import SimpleNamespace as NS
import math
import numpy as np
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead import stage2_rollout as m


def fixture(monkeypatch, *, failure=None, horizon=2.):
    row = dict(future_tokens=torch.ones(1,257,768,dtype=torch.float16),
               future_valid_mask=torch.tensor([True]),q1_conditions=torch.tensor([[1.,2.,3.,4.]]),
               q1_metadata=[dict(position=[0.,0.,-1.],yaw=0.,horizon=horizon,noise_seed=7)],
               invalid_reason=['valid'])
    requests=[]; noises=[]; calls=[]
    config=NS(metric_waypoint_spacing=.25,min_horizon=1.,max_horizon=64.)
    def build(req, device):
        requests.extend(req)
        return NS(condition_tensor=torch.tensor([[8.,7.,6.,5.] for _ in req]))
    def predict(batch, initial_noise):
        noises.append(initial_noise.clone())
        value=float(len(noises)+1)
        tokens=torch.full((1,257,768),value)
        if failure=='nan':tokens.fill_(float('nan'))
        return NS(pred_cls=tokens[:,0], pred_tokens=tokens,
                  pred_latent=torch.full((1,768,16,16),value))
    def cwp(patch):
        calls.append(patch.clone())
        if failure=='exception':raise RuntimeError('deployment broken')
        return None
    def decode(output, **kwargs):
        none=failure=='none' and len(calls)==2
        return [NS(valid=True,pred_none=none,local_angle_deg=90.,distance_m=.5)]
    monkeypatch.setattr(m,'decode_dino_cwp_top1',decode)
    rt=NS(generator=torch.Generator().manual_seed(123),
          adapter=NS(config=config,build_raenwm_latent_batch=build),_predict_batch=predict)
    collector=NS(runtime=rt,trainer=NS(device=torch.device('cpu')),cwp=cwp,counts=Counter())
    snapshot=object()
    state=dict(env=0,slot=0,ghost='g1',snapshot=snapshot,latent=torch.ones(768,16,16),
               position=np.array([0.,0.,-1.]),yaw=0.,horizon=horizon,identity='23|ep|4|g1|1')
    return collector,row,state,requests,noises,calls


def test_rollout_reuses_history_and_recurses_from_predicted_pose(monkeypatch):
    c,row,s,requests,noises,calls=fixture(monkeypatch)
    before=c.runtime.generator.get_state().clone()
    global_before=torch.get_rng_state().clone()
    m.initialize_rollout_rows([row],3)
    m.extend_rollout(c,[row],[s],3)
    assert len(requests)==2
    assert all(r.snapshot is s['snapshot'] for r in requests)
    assert np.allclose(requests[0].target_position,[-.5,0,-1])
    assert np.allclose(requests[1].target_position,[-.5,0,-.5])
    assert [r.horizon_override for r in requests]==[4.,6.]
    assert calls[0].mean()==1 and calls[1].mean()==2
    assert row['future_tokens'].mean()==3
    assert row['q1_conditions'].tolist()==[[1.,2.,3.,4.]]
    assert row['endpoint_conditions'].tolist()==[[8.,7.,6.,5.]]
    assert row['q1_metadata'][0]['horizon']==2
    assert row['realized_depths'].tolist()==[3]
    assert row['full_horizon_mask'].tolist()==[True]
    assert row['fallback_mask'].tolist()==[False]
    assert not torch.equal(noises[0],noises[1])
    assert torch.equal(before,c.runtime.generator.get_state())
    assert torch.equal(global_before,torch.get_rng_state())


@pytest.mark.parametrize('failure,horizon,realized,reason',[
    ('none',2.,2,'cwp_none'),('nan',2.,1,'nwm_nonfinite'),(None,63.,1,'horizon_out_of_range')])
def test_deep_failure_retains_deepest_valid(monkeypatch,failure,horizon,realized,reason):
    c,row,s,*_=fixture(monkeypatch,failure=failure,horizon=horizon)
    m.initialize_rollout_rows([row],3)
    m.extend_rollout(c,[row],[s],3)
    assert row['realized_depths'].tolist()==[realized]
    assert row['future_tokens'].mean()==realized
    assert row['future_valid_mask'].tolist()==[True]
    assert row['fallback_mask'].tolist()==[True]
    assert row['full_horizon_mask'].tolist()==[False]
    assert row['invalid_reason']==['valid']
    assert row['rollout_failure_reason']==[reason]


def test_unknown_runtime_failure_propagates(monkeypatch):
    c,row,s,*_=fixture(monkeypatch,failure='exception')
    m.initialize_rollout_rows([row],2)
    with pytest.raises(RuntimeError,match='deployment broken'):
        m.extend_rollout(c,[row],[s],2)


def test_depth_one_does_not_query_or_change_tokens(monkeypatch):
    c,row,s,req,*_=fixture(monkeypatch)
    before=row['future_tokens'].clone()
    m.initialize_rollout_rows([row],1)
    m.extend_rollout(c,[row],[s],1)
    assert req==[] and torch.equal(before,row['future_tokens'])
    assert row['full_horizon_mask'].tolist()==[True]


def test_noise_preserves_legacy_first_step_and_separates_depths():
    import hashlib
    identity='12|ep|4|g|1'
    expected=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8],'little')%(2**63-1)
    assert m.recursive_noise_seed(identity,1)==expected
    assert len({m.recursive_noise_seed(identity,k) for k in (1,2,3)})==3


@pytest.mark.parametrize('value',[0,4,True,1.5,'2'])
def test_invalid_depth_rejected(value):
    with pytest.raises(ValueError):m.rollout_depth(NS(lookahead_horizon_steps=value))


def test_missing_q1_never_becomes_valid(monkeypatch):
    c,row,s,requests,*_=fixture(monkeypatch)
    row['future_valid_mask'].fill_(False)
    row['future_tokens'].zero_()
    row['q1_metadata']=[None]
    row['invalid_reason']=['no_q0']
    m.initialize_rollout_rows([row],3)
    m.extend_rollout(c,[row],[],3)
    assert requests==[]
    assert row['realized_depths'].tolist()==[0]
    assert row['full_horizon_mask'].tolist()==[False]
    assert row['fallback_mask'].tolist()==[False]
    assert row['endpoint_metadata']==[None]
    assert row['rollout_failure_reason']==['no_q0']
