#!/usr/bin/env python3
"""CPU/GPU offline progressive validation, training, evaluation and prefix replay.

No Habitat import, source policy, world model, or teacher is instantiated.
"""
import argparse
import importlib.metadata
import platform
from dataclasses import asdict
import json
from pathlib import Path
import sys
import types
import torch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
package=types.ModuleType('vlnce_baselines'); package.__path__=[str(root/'vlnce_baselines')]
sys.modules['vlnce_baselines']=package
from vlnce_baselines.nwm.active_lookahead.progressive_data import ProgressiveDataset,collate_progressive,validate_dataset
from vlnce_baselines.nwm.active_lookahead.progressive_training import initial_state,forward_progressive,full_scores,full_state_scores,progressive_loss,load_checkpoint,save_checkpoint
from vlnce_baselines.nwm.active_lookahead.progressive_head import ProgressiveHeadConfig,ProgressiveE24Head
from vlnce_baselines.nwm.active_lookahead.progressive_core import certify_decision
from vlnce_baselines.nwm.active_lookahead.stage2_data import sha256
from vlnce_baselines.nwm.active_lookahead.stage2_training import seed_everything


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['validate','train','evaluate','replay'])
    p.add_argument('--data',required=True); p.add_argument('--checkpoint'); p.add_argument('--model-config',help='JSON containing ProgressiveHeadConfig fields')
    p.add_argument('--output'); p.add_argument('--resume'); p.add_argument('--device',default='cpu')
    p.add_argument('--steps',type=int,default=100); p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=2)
    p.add_argument('--prefix-loss-weight',type=float,default=.2)
    p.add_argument('--pruning-mode',choices=['certified','none'],default='certified')
    a=p.parse_args()
    versions=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda)
    for name in ('transformers','habitat-lab','habitat-sim'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]='not installed as a distribution'
    print(json.dumps(dict(runtime_versions=versions,device=a.device)),file=sys.stderr,flush=True)
    if a.command=='validate': print(json.dumps(validate_dataset(a.data),ensure_ascii=False)); return
    data=ProgressiveDataset(a.data); provenance=data.provenance
    sources=[dict(root=str(Path(a.data).resolve()),manifest_sha256=sha256(Path(a.data)/'dataset_manifest.json'),provenance_sha256=data.manifest['provenance_sha256'])]
    if a.batch_size<1 or a.steps<1: p.error('steps and batch-size must be positive')
    seed_everything(a.seed)
    def batch(indices): return {k:v.to(a.device) for k,v in collate_progressive([data[i] for i in indices],provenance).items()}
    if a.command=='train':
        if not a.output: p.error('--output is required for training')
        if provenance['split'] != 'train': raise ValueError('training requires train split')
        config=ProgressiveHeadConfig(**json.loads(Path(a.model_config).read_text())) if a.model_config else ProgressiveHeadConfig()
        if config.max_future_depth != provenance['max_future_depth'] or any(provenance['feature_contract'][k] != config.input_dim for k in ('feature_dim','owner_dim','text_dim')):
            raise ValueError('data/model feature or depth contract differs')
        head=ProgressiveE24Head(config).to(a.device); optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr)
        loss_config=dict(final_loss_weight=1.,prefix_loss_weight=a.prefix_loss_weight,increment_regularization=.001)
        start=offset=0
        if a.resume:
            head,state=load_checkpoint(a.resume,expected_provenance=provenance,head=head,optimizer=optimizer,resume=True,device=a.device,expected_loss_config=loss_config,expected_data_sources=sources)
            start=state['global_step']; cursor=state['cursor']
            if cursor['batch_size'] != a.batch_size or cursor['seed'] != a.seed: raise ValueError('resume sampler config differs')
            offset=cursor['offset']
        head.train()
        for step in range(start,start+a.steps):
            indices=[(offset+i)%len(data) for i in range(a.batch_size)]; offset+=len(indices)
            optimizer.zero_grad(set_to_none=True); loss,_=progressive_loss(head,batch(indices),**loss_config)
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.); optimizer.step()
            print(json.dumps(dict(step=step+1,loss=float(loss.detach()))),flush=True)
        output=Path(a.output); end=start+a.steps
        save_checkpoint(output/f'state_step_{end:06d}.pt',head,provenance,optimizer=optimizer,step=end,cursor=dict(offset=offset,batch_size=a.batch_size,seed=a.seed),loss_config=loss_config,data_sources=sources)
        save_checkpoint(output/f'head_step_{end:06d}.pt',head,provenance,step=end,loss_config=loss_config,data_sources=sources)
        return
    if not a.checkpoint: p.error('--checkpoint required')
    head,_=load_checkpoint(a.checkpoint,expected_provenance=provenance,device=a.device); head.eval()
    records=[]; correct=eligible=0
    with torch.no_grad():
        for i in range(len(data)):
            b=batch([i]); states=forward_progressive(head,b); final=full_state_scores(b,states[-1])
            if a.command=='evaluate':
                target=int(b['teacher_base_index'][0]); usable=bool(b['teacher_valid'][0] and not b['teacher_stop'][0] and not b['base_stop'][0] and not b['no_vp_left'][0])
                eligible+=usable; correct+=int(usable and int(final.argmax(1)[0])==target)
                continue
            # Replay reads only the current cache prefix. Deep terminal/valid masks
            # cannot reduce budgets until that depth is actually consumed.
            state=initial_state(head,b); cert=None; reason='max_depth'; deltas=[]; winners=[]
            for depth in range(head.config.max_future_depth+1):
                scores=full_state_scores(b,state)[0]
                remaining=torch.zeros_like(scores)
                alive=b['candidate_present_mask'][0].clone()
                if depth: alive &= ~b['future_terminal_mask'][0,:,depth-1]
                ix=b['topk_base_indices'][0,alive]
                remaining[ix]=sum(head.budgets[depth:])
                cert=certify_decision(scores,remaining,b['ghost_valid_mask'][0]); winners.append(cert.winner_id)
                if bool(b['base_stop'][0] or b['no_vp_left'][0]): reason='original_stop'; break
                if a.pruning_mode=='certified' and cert.certified: reason='certified'; break
                if depth==head.config.max_future_depth: break
                if not alive.any(): reason='natural_termination'; break
                state=head.step(state,b['future_tokens'][:,:,depth],b['future_geometry'][:,:,depth],b['future_valid_mask'][:,:,depth],b['future_token_mask'][:,:,depth]); deltas.append(state.delta[0].tolist())
            same_prefix=torch.allclose(state.scores,states[depth].scores,atol=1e-6,rtol=1e-6)
            same_action=not b['ghost_valid_mask'].any() or int(scores.argmax())==int(final.argmax(1)[0])
            if not same_prefix or not same_action: raise AssertionError('causal replay or certified/full argmax mismatch')
            records.append(dict(row=i,executed_depth=depth,stop_reason=reason,per_depth_delta=deltas,cumulative_delta=state.cumulative_delta[0].tolist(),base_winner=winners[0],prefix_winner=winners,final_winner=cert.winner_id,remaining_budget=remaining.tolist(),certificate_margin=cert.certificate_margin,numerical_guard=cert.numerical_guard,cache_prefix_matches=True,full_argmax_matches=True))
    report=dict(command=a.command,rows=len(data),eligible=eligible,correct=correct,records=records,cache_only=True)
    if a.output:
        path=Path(a.output)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+'\n')
    else: print(json.dumps(report))

if __name__=='__main__': main()
