import dataclasses
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead.progressive_head import ProgressiveE24Head,ProgressiveHeadConfig


def fixture(depth=2,k=3):
    torch.manual_seed(3)
    c=ProgressiveHeadConfig(input_dim=8,hidden_dim=8,num_queries=2,num_attention_heads=2,ffn_dim=16,dropout=0,fusion_layers=2,max_future_depth=depth,budget_fractions=tuple([1/depth]*depth))
    model=ProgressiveE24Head(c)
    with torch.no_grad(): model.score_mlp[-1].weight.normal_(0,.1)
    data=dict(owner_embeddings=torch.randn(2,k,8),text_tokens=torch.randn(2,4,8),base_logits=torch.randn(2,k),future_tokens=torch.randn(2,k,depth,4,8),future_geometry=torch.randn(2,k,depth,5),future_valid_mask=torch.ones(2,k,depth,dtype=torch.bool),candidate_present_mask=torch.ones(2,k,dtype=torch.bool),candidate_q0_geometry=torch.randn(2,k,3))
    return model,data


@pytest.mark.parametrize('depth,k',[(1,1),(2,3),(2,0)])
def test_incremental_bound_gradient(depth,k):
    model,x=fixture(depth,k)
    out=model(**x)
    assert out.per_depth_delta.shape==(2,k,depth)
    assert (out.cumulative_delta.abs()<=1).all()
    init={key:value for key,value in x.items() if not key.startswith('future_')}
    state=model.initialize_state(**init)
    for d in range(depth):
        state=model.step(state,x['future_tokens'][:,:,d],x['future_geometry'][:,:,d],x['future_valid_mask'][:,:,d])
        torch.testing.assert_close(state.scores,out.states[d+1].scores)
        assert (state.delta.abs()<=model.budgets[d]).all()
    if k:
        out.scores.square().sum().backward()
        assert model.memory_proj.weight.grad.abs().sum()>0
        assert model.future_proj.weight.grad.abs().sum()>0


def test_masks_nan_causality_and_permutation():
    model,x=fixture()
    x['candidate_present_mask'][:,2]=False
    x['future_valid_mask'][:,2]=False
    x['future_valid_mask'][0,1,1]=False
    out=model(**x)
    y={key:value.clone() for key,value in x.items()}
    for key in ['owner_embeddings','candidate_q0_geometry','base_logits']:
        y[key][:,2]=float('nan')
    y['future_tokens'][~y['future_valid_mask']]=float('nan')
    y['future_geometry'][~y['future_valid_mask']]=float('nan')
    nanout=model(**y)
    torch.testing.assert_close(nanout.scores,out.scores)
    torch.testing.assert_close(out.states[1].memory[0,1],out.states[2].memory[0,1])
    y['future_tokens'][:,:,1]=123
    y['future_geometry'][:,:,1]=-42
    y['future_valid_mask'][:,:,1]=False
    torch.testing.assert_close(model(**y).states[1].scores,out.states[1].scores)
    order=torch.tensor([2,0,1])
    perm={key:(value if key=='text_tokens' else value[:,order]) for key,value in x.items()}
    torch.testing.assert_close(model(**perm).scores,out.scores[:,order],atol=1e-6,rtol=1e-5)


def test_all_invalid_and_masked_tokens():
    model,x=fixture()
    x['future_valid_mask'].fill_(False)
    x['future_tokens'].fill_(float('nan'))
    out=model(**x)
    torch.testing.assert_close(out.scores,x['base_logits'])
    torch.testing.assert_close(out.memory,out.states[0].memory)
    model,x=fixture()
    x['future_token_mask']=torch.ones(2,3,2,4,dtype=torch.bool)
    x['future_token_mask'][:,:,:,-1]=False
    before=model(**x)
    x['future_tokens'][:,:,:,-1]=float('nan')
    torch.testing.assert_close(model(**x).scores,before.scores)
    x['future_token_mask'][:,:,:,0]=False
    with pytest.raises(ValueError,match='CLS'): model(**x)


def test_zero_initialization_update_save_load(tmp_path):
    model,x=fixture()
    torch.nn.init.zeros_(model.score_mlp[-1].weight)
    torch.testing.assert_close(model(**x).cumulative_delta,torch.zeros(2,3))
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    model(**x).scores.sum().backward()
    assert model.score_mlp[-1].weight.grad.abs().sum()>0
    # Zero final weights intentionally delay upstream gradients on first update.
    assert model.memory_proj.weight.grad.abs().sum()==0
    opt.step()
    path=tmp_path/'head.pt'
    torch.save(dict(config=dataclasses.asdict(model.config),model=model.state_dict()),path)
    ckpt=torch.load(path,weights_only=True)
    restored=ProgressiveE24Head(ProgressiveHeadConfig(**ckpt['config']))
    restored.load_state_dict(ckpt['model'],strict=True)
    torch.testing.assert_close(restored(**x).scores,model(**x).scores)


def test_source_detached_but_memory_keeps_cross_depth_gradient():
    model,x=fixture()
    source=torch.nn.Linear(8,8)
    for name in ('owner_embeddings','text_tokens','future_tokens'):
        x[name]=source(x[name])
    out=model(**x)
    out.states[1].memory.retain_grad()
    out.scores.sum().backward()
    assert all(p.grad is None for p in source.parameters())
    assert out.states[1].memory.grad is not None
    assert out.states[1].memory.grad.abs().sum()>0
