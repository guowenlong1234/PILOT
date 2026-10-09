from collections import Counter
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.nwm.active_lookahead.inference_gate import (
    deployment_residual_bound, inference_query_plans,
)
from vlnce_baselines.nwm.active_lookahead.stage2_collect import Stage2Collector


def plan(values, dtype=torch.float32, bound=1., step=0, exhausted=False):
    return inference_query_plans([[None]+[f'g{i}' for i in range(len(values)-1)]],
        torch.tensor([values],dtype=dtype),[exhausted],step,10,bound=bound)[0]


def test_actual_deployment_clip_and_disabled_gain():
    assert deployment_residual_bound(1.5,1.)==1.
    assert deployment_residual_bound(.5,1.)==.5
    assert deployment_residual_bound(0.,1.)==0.
    for gain, limit in [(float('nan'),1.),(1.,-1.)]:
        with pytest.raises(ValueError): deployment_residual_bound(gain,limit)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64])
def test_strict_certificate_and_actual_low_precision_addition(dtype):
    assert plan([-9,3,1],dtype)['skip_reason'] is None
    assert plan([-9,3.1,1],dtype)['skip_reason'] in (None,'bounded_invariant')
    # Include candidates outside Top5: their residual is exactly zero.
    values=[-100,6,2,1,.5,0,-1]
    assert plan(values,dtype)['skip_reason']=='bounded_invariant'
    for deltas in product((-1.,1.),repeat=5):
        scores=torch.tensor(values,dtype=dtype)
        scores[1:6]+=torch.tensor(deltas,dtype=dtype)
        assert scores.argmax().item()==1


def test_roundoff_guard_stop_single_and_forced_stop():
    assert plan([-1,3.0000002,1])['skip_reason'] is None
    assert plan([4,3,1])['skip_reason']=='base_stop'
    assert plan([-1,0])['skip_reason']=='single_candidate'
    assert plan([-1,3,2],step=9)['skip_reason']=='forced_stop'
    assert plan([-1,3,2],exhausted=True)['skip_reason']=='forced_stop'
    assert plan([-1,3,2],bound=0)['skip_reason']=='bounded_invariant'


def test_pruned_predictions_never_read_q0_or_call_world_model():
    class ForbiddenCache:
        def get(self,*args): raise AssertionError('pruned row read q0')
    obj=object.__new__(Stage2Collector)
    obj.prediction_only=True; obj.counts=Counter()
    obj.trainer=SimpleNamespace(device='cpu',max_len=10,
        envs=SimpleNamespace(current_episodes=lambda:[SimpleNamespace(episode_id='1')]),
        gmaps=[SimpleNamespace(stage2_q0=ForbiddenCache())])
    obj.runtime=SimpleNamespace(generator=torch.Generator())
    logits=torch.tensor([[-1.,4.,1.]])
    args=({'gmap_vp_ids':[[None,'g0','g1']]},
          {'global_logits':logits,'gmap_embeds':torch.zeros(1,3,768)},None,None,[False],0)
    rows=obj.predict_step(*args,skip_reasons=['bounded_invariant'])
    assert obj.counts['q1_requested']==0 and obj.counts['cwp_requested']==0
    assert not rows[0]['future_valid_mask'].any()
    assert rows[0]['invalid_reason']==['bounded_invariant']*2
    # Omitting the online plan retains the original collection path.
    with pytest.raises(AssertionError,match='read q0'): obj.predict_step(*args)
    obj.prediction_only=False
    with pytest.raises(ValueError,match='online-only'):
        obj.predict_step(*args,skip_reasons=['bounded_invariant'])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='actual deployment GPU required')
def test_mixed_and_all_pruned_rows_skip_head_and_scatter_correctly(monkeypatch,tmp_path):
    from vlnce_baselines.nwm.active_lookahead import stage2_online as module
    obj=object.__new__(module.Stage2Online)
    obj.cfg=SimpleNamespace(gain=1.,bounded_skip=True,profile=True)
    obj.head=SimpleNamespace(delta_max=1.)
    obj.trainer=SimpleNamespace(device=torch.device('cuda'),max_len=10,
        envs=SimpleNamespace(current_episodes=lambda:[SimpleNamespace(episode_id=str(i),scene_id='s') for i in range(3)]))
    obj.counts=Counter();obj.episode_diagnostics={};obj.step_timings=[];obj.online_seconds=0.
    obj.head_stream=torch.cuda.Stream();obj.trace_file=None
    logits=torch.tensor([[-1.,4.,1.],[-1.,2.,1.5],[4.,2.,1.]],device='cuda')
    inputs={'gmap_vp_ids':[[None,'g0','g1']]*3}
    calls=[]
    def predict(*args,skip_reasons):
        calls.append(skip_reasons)
        return [dict(future_valid_mask=torch.tensor([True,True]) if reason is None else torch.tensor([False,False]),
                     global_indices=[1,2],topk_base_indices=torch.tensor([0,1]),
                     base_stop=i==2,forced_stop=False,invalid_reason=[reason or 'valid']*2)
                for i,reason in enumerate(skip_reasons)]
    obj.predict_step=predict
    def score(head,rows,device,gain):
        assert len(rows)==1
        return torch.tensor([[-1.,1.,0,0,0]],device=device)
    monkeypatch.setattr(module,'score_rows',score)
    text=torch.zeros(3,2,768,device='cuda');mask=torch.ones(3,2,dtype=torch.bool,device='cuda')
    result=obj.score_step(inputs,{'global_logits':logits},text,mask,[False]*3,0)
    assert calls==[['bounded_invariant',None,'base_stop']]
    assert result.cpu().tolist()==[[0,0,0],[0,-1,1],[0,0,0]]
    assert obj.counts['head_rows']==1 and obj.counts['action_flips']==1
    logits[1,1]=5
    result=obj.score_step(inputs,{'global_logits':logits},text,mask,[False]*3,1)
    assert not result.any() and obj.counts['head_calls']==1
