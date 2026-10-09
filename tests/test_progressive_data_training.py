"""Synthetic CPU contracts; these tests are not navigation acceptance results."""
from dataclasses import replace
import copy
import torch
import pytest
from vlnce_baselines.nwm.active_lookahead.progressive_data import *
from vlnce_baselines.nwm.active_lookahead.progressive_training import *
from vlnce_baselines.nwm.active_lookahead.progressive_head import ProgressiveE24Head,ProgressiveHeadConfig


def fixture():
    p=dict(prediction_contract=dict(PREDICTION_CONTRACT),format=FORMAT,feature_space=SPACE,geometry_definition=GEOMETRY,distance_scale=1.,max_future_depth=2,
        feature_contract=dict(token_count=3,feature_dim=8,owner_dim=8,text_dim=8),stage1_sha256='a'*64,
        assets={'world_model':{'sha256':'b'*64}},context_contract='observed_only',behavior='stage1_frozen',split='train')
    row=dict(step=0,ghost_ids=['a','b','c'],base_logits=torch.tensor([.3,.2,.1]),topk_base_indices=torch.tensor([0,1]),
        owner_embeddings=torch.randn(2,8),candidate_q0_geometry=torch.randn(2,3),future_tokens=torch.randn(2,2,3,8).half(),
        future_geometry=torch.randn(2,2,5),candidate_present_mask=torch.ones(2,dtype=torch.bool),
        future_valid_mask=torch.ones(2,2,dtype=torch.bool),future_token_mask=torch.ones(2,2,3,dtype=torch.bool),
        future_terminal_mask=torch.zeros(2,2,dtype=torch.bool),future_queried_mask=torch.ones(2,2,dtype=torch.bool),
        target_positions=torch.zeros(2,2,3),target_yaws=torch.zeros(2,2),horizons=torch.ones(2,2),
        request_ids=[['a1','a2'],['b1','b2']],termination_reasons=[['',''],['','']],q0_metadata=[{'source_step':0},{'source_step':0}],
        teacher_base_index=1,teacher_rank_in_topk=1,teacher_valid=True,teacher_stop=False,base_stop=False,no_vp_left=False,
        text_tokens=torch.randn(4,8))
    config=ProgressiveHeadConfig(input_dim=8,hidden_dim=8,num_queries=2,num_attention_heads=2,ffn_dim=16,fusion_layers=1,dropout=0.)
    return p,row,config


def test_training_save_resume_and_causal_replay(tmp_path):
    torch.set_num_threads(1)
    p,row,c=fixture(); batch=collate_progressive([row],p); head=ProgressiveE24Head(c)
    opt=torch.optim.AdamW(head.parameters(),lr=.01)
    before={k:v.clone() for k,v in head.state_dict().items()}
    for _ in range(2):
        opt.zero_grad();loss,states=progressive_loss(head,batch); assert torch.isfinite(loss);loss.backward();opt.step()
    assert any(not torch.equal(before[k],v) for k,v in head.state_dict().items())
    assert all(not batch[k].requires_grad for k in ('owner_embeddings','future_tokens','text_tokens'))
    assert any(v.grad is not None and v.grad.abs().sum()>0 for v in head.parameters())
    path=tmp_path/'state.pt';save_checkpoint(path,head,p,optimizer=opt,step=2,cursor={'offset':2})
    other=ProgressiveE24Head(c); other_opt=torch.optim.AdamW(other.parameters(),lr=.01)
    other,obj=load_checkpoint(path,expected_provenance=p,head=other,optimizer=other_opt,resume=True)
    head.eval();other.eval()
    states=forward_progressive(head,batch); restored=forward_progressive(other,batch)
    assert obj['global_step']==2
    assert torch.equal(states[-1].scores,restored[-1].scores)
    changed=copy.deepcopy(batch);changed['future_tokens'][:,:,1]*=100;changed['future_geometry'][:,:,1]*=100
    assert torch.equal(states[1].scores,forward_progressive(head,changed)[1].scores)
    bad=copy.deepcopy(p);bad['assets']['world_model']['sha256']='c'*64
    with pytest.raises(ValueError,match='source'): load_checkpoint(path,expected_provenance=bad)
    with pytest.raises(ValueError,match='budget'): load_checkpoint(path,expected_provenance=p,head=ProgressiveE24Head(replace(c,budget_fractions=(.2,.8))))
    with pytest.raises(FileExistsError): save_checkpoint(path,head,p)


def test_dataset_sha_and_schema(tmp_path):
    p,row,c=fixture();writer=ProgressiveEpisodeWriter(tmp_path,p)
    writer.append('episode','scene',row['text_tokens'],row);writer.complete('episode',{})
    data=ProgressiveDataset(tmp_path);assert len(data)==1
    assert collate_progressive([data[0]],p)['future_tokens'].shape==(1,2,2,3,8)
    path=next(tmp_path.glob('*.pt'));path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='SHA'): validate_dataset(tmp_path)
    p['format']='etpr1-stage2-predicted-episode-v1'
    with pytest.raises(ValueError,match='schema'): validate_provenance(p)


def test_terminal_invalid_nan_and_no_future():
    p,row,c=fixture()
    row['future_valid_mask'][:]=False;row['future_token_mask'][:]=False
    row['future_terminal_mask'][:]=True;row['future_queried_mask'][:,1]=False
    row['future_tokens'][:]=float('nan');row['future_geometry'][:]=float('nan')
    row['termination_reasons']=[['cwp_none','cwp_none'],['cwp_none','cwp_none']]
    batch=collate_progressive([row],p);head=ProgressiveE24Head(c)
    loss,states=progressive_loss(head,batch);loss.backward()
    assert torch.isfinite(loss);assert torch.equal(states[-1].cumulative_delta,torch.zeros(1,2))
    row['future_terminal_mask'][0,1]=False
    with pytest.raises(ValueError,match='resumed'): validate_row(row,row['text_tokens'],p)


def test_variable_empty_k_and_teacher_absent():
    p,row,c=fixture(); empty=copy.deepcopy(row)
    for k in ('owner_embeddings','candidate_q0_geometry','future_tokens','future_geometry','candidate_present_mask','future_valid_mask','future_token_mask','future_terminal_mask','future_queried_mask','target_positions','target_yaws','horizons','topk_base_indices'):
        empty[k]=empty[k][:0]
    empty.update(q0_metadata=[],request_ids=[],termination_reasons=[],teacher_rank_in_topk=-1)
    row['teacher_base_index']=2;row['teacher_rank_in_topk']=-1
    head=ProgressiveE24Head(c)
    for rows in ([empty],[row,empty]):
        batch=collate_progressive(rows,p);loss,states=progressive_loss(head,batch)
        assert torch.isfinite(loss);loss.backward()


def test_cli_smoke_train_resume_evaluate_replay(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from dataclasses import asdict
    from pathlib import Path
    p,row,c=fixture();writer=ProgressiveEpisodeWriter(tmp_path/'data',p)
    writer.append('mock','mock',row['text_tokens'],row);writer.complete('mock',{})
    (tmp_path/'model.json').write_text(json.dumps(asdict(c)))
    script=Path(__file__).resolve().parents[1]/'scripts/progressive_stage2.py'
    def run(command,*args):
        result=subprocess.run([sys.executable,str(script),command,'--data',str(tmp_path/'data'),*map(str,args)],
            env=dict(os.environ,OMP_NUM_THREADS='1'),capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        return result.stdout
    run('validate')
    run('train','--model-config',tmp_path/'model.json','--output',tmp_path/'run','--steps',2,'--batch-size',1)
    run('train','--model-config',tmp_path/'model.json','--output',tmp_path/'run','--steps',1,'--batch-size',1,'--resume',tmp_path/'run/state_step_000002.pt')
    checkpoint=tmp_path/'run/head_step_000003.pt'
    assert json.loads(run('evaluate','--checkpoint',checkpoint))['rows']==1
    for mode in ('certified','none'):
        report=json.loads(run('replay','--checkpoint',checkpoint,'--pruning-mode',mode))
        assert report['records'][0]['full_argmax_matches']
        assert report['records'][0]['cache_prefix_matches']


def test_checkpoint_version_and_source_scale_are_strict(tmp_path):
    p,row,c=fixture();head=ProgressiveE24Head(c)
    path=tmp_path/'head.pt';save_checkpoint(path,head,p)
    bad=copy.deepcopy(p);bad['distance_scale']=2.
    with pytest.raises(ValueError,match='source'):load_checkpoint(path,expected_provenance=bad)
    obj=torch.load(path,weights_only=False);obj['format_version']='legacy-v1'
    wrong=tmp_path/'wrong.pt';torch.save(obj,wrong)
    with pytest.raises(ValueError,match='unsupported version'):load_checkpoint(wrong,expected_provenance=p)
