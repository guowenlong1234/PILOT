"""Deterministic mock prediction; these are not navigation results."""
from types import SimpleNamespace
from dataclasses import replace
import copy
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead.progressive_head import ProgressiveHeadConfig,ProgressiveE24Head
from vlnce_baselines.nwm.active_lookahead.progressive_controller import ProgressiveController,initialize_row


def row(base,indices=None):
    indices=list(range(len(base))) if indices is None else indices
    k=len(indices)
    return dict(base_logits=torch.tensor(base,dtype=torch.float32),topk_base_indices=torch.tensor(indices,dtype=torch.long),
        owner_embeddings=torch.ones(k,8),candidate_present_mask=torch.ones(k,dtype=torch.bool),
        text_tokens=torch.ones(3,8),candidate_q0_geometry=torch.zeros(k,3),base_stop=False)


def model():
    return ProgressiveE24Head(ProgressiveHeadConfig(input_dim=8,hidden_dim=8,num_attention_heads=2,
        num_queries=2,ffn_dim=16,dropout=0,fusion_layers=1)).eval()


class Predictor:
    def __init__(self, terminal=False): self.calls=[]; self.terminal=terminal
    def predict_next_depth(self,decision,active_rows):
        self.calls.append(list(active_rows))
        result=[]
        for i,r in enumerate(decision.rows):
            k=len(r['topk_base_indices'])
            result.append(dict(future_tokens=torch.ones(k,3,8),future_geometry=torch.zeros(k,5),
                future_valid_mask=torch.full((k,),not self.terminal,dtype=torch.bool),
                future_token_mask=torch.ones(k,3,dtype=torch.bool),
                terminal_mask=torch.full((k,),self.terminal,dtype=torch.bool),world_model_queries=k))
        return result


def test_batch_mixed_depth_stops_and_none_never_prunes():
    head=model(); decision=SimpleNamespace(rows=[row([5.,0]),row([0.,0.])])
    short=Predictor(); full=Predictor()
    a=ProgressiveController(head).run(short,decision)
    b=ProgressiveController(head,'none').run(full,decision)
    assert short.calls==[[1],[1]] and full.calls==[[0,1],[0,1]]
    assert [x.argmax().item() for x in a['scores']]==[x.argmax().item() for x in b['scores']]
    assert a['diagnostics'][0]['executed_depth']==0
    assert a['diagnostics'][0]['stop_reason']=='certified'
    assert b['diagnostics'][0]['executed_depth']==2


def test_terminal_keeps_candidate_and_topk_outside_competes():
    head=model(); decision=SimpleNamespace(rows=[row([0.,3.],[0]),row([0.,0.])])
    p=Predictor(terminal=True)
    out=ProgressiveController(head).run(p,decision)
    assert out['diagnostics'][0]['final_winner']==1 and p.calls==[[1]]
    assert out['diagnostics'][1]['stop_reason']=='natural_terminal'
    torch.testing.assert_close(out['scores'][1],decision.rows[1]['base_logits'])


def test_prefix_replay_and_fresh_decision_state():
    torch.manual_seed(4); head=model()
    with torch.no_grad(): head.score_mlp[-1].weight.normal_(0,.2)
    r=row([.1,0.]); decision=SimpleNamespace(rows=[r]); rng=torch.get_rng_state().clone()
    out=ProgressiveController(head,'none').run(Predictor(),decision)
    state=initialize_row(head,r,'cpu'); layer=Predictor().predict_next_depth(decision,[0])[0]
    for d in range(2):
        state=head.step(state,layer['future_tokens'][None],layer['future_geometry'][None],
                        layer['future_valid_mask'][None],layer['future_token_mask'][None])
        torch.testing.assert_close(torch.tensor(out['diagnostics'][0]['per_depth_delta'][d]),state.delta[0],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(out['scores'][0],state.scores[0])
    repeat=ProgressiveController(head,'none').run(Predictor(),decision)
    torch.testing.assert_close(repeat['scores'][0],out['scores'][0])
    assert torch.equal(rng,torch.get_rng_state())


def test_winner_corrected_then_certified():
    head=model(); original=head.step
    def scripted(state,*args):
        result=original(state,*args)
        delta=torch.tensor([[-.5,.5]])
        return replace(result,delta=delta,cumulative_delta=state.cumulative_delta+delta,scores=state.scores+delta)
    head.step=scripted
    p=Predictor(); out=ProgressiveController(head).run(p,SimpleNamespace(rows=[row([.4,0.])]))
    assert out['diagnostics'][0]['prefix_winner']==[1,1]
    assert out['diagnostics'][0]['certificates'][-1]['certified']
    assert out['diagnostics'][0]['certificates'][1]['certified'] is False


def test_first_depth_correction_can_lock_with_asymmetric_budget():
    head=model()
    head.config=replace(head.config,budget_fractions=(.9,.1))
    head.budgets=(.9,.1)
    original=head.step
    def scripted(state,*args):
        result=original(state,*args)
        delta=state.scores.new_tensor([[-1.,1.]])*head.budgets[state.depth]
        return replace(result,delta=delta,cumulative_delta=state.cumulative_delta+delta,scores=state.scores+delta)
    head.step=scripted
    p=Predictor();decision=SimpleNamespace(rows=[row([.4,0.])])
    short=ProgressiveController(head).run(p,decision)
    full=ProgressiveController(head,'none').run(Predictor(),decision)
    assert len(p.calls)==1 and short['diagnostics'][0]['stop_reason']=='certified'
    assert short['diagnostics'][0]['base_winner']==0
    assert short['diagnostics'][0]['final_winner']==full['diagnostics'][0]['final_winner']==1
