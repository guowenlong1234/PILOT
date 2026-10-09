"""Causal offline supervision and strict progressive checkpoint contracts."""
from dataclasses import asdict, replace
import math
from pathlib import Path
import torch
from .progressive_data import FORMAT, GEOMETRY, validate_provenance
from .stage2_training import LOSS_CONFIG, atomic_torch_save, rng_state, restore_rng
from .offline_objective import offline_decision_aware_loss

MODEL_TYPE='progressive_e24_v1'
HEAD_FORMAT='progressive-e24-head-v1'
TRAINING_FORMAT='progressive-e24-training-v1'


def initial_state(head,batch):
    base=batch['base_logits'].float().detach()
    masked=base.masked_fill(~batch['ghost_valid_mask'],-torch.inf)
    # Empty move rows (STOP) need a finite context but never submit a residual.
    safe=torch.where(batch['ghost_valid_mask'].any(1,keepdim=True),masked,torch.zeros_like(masked))
    logp=torch.log_softmax(safe,dim=1)
    ix=batch['topk_base_indices'].clamp_min(0)
    present=batch['candidate_present_mask']
    return head.initialize_state(batch['owner_embeddings'].detach(),batch['text_tokens'].detach(),
        base.gather(1,ix).masked_fill(~present,0),present,
        base_log_probs=logp.gather(1,ix).masked_fill(~present,0),
        candidate_q0_geometry=batch['candidate_q0_geometry'].detach(),text_token_mask=batch['text_token_mask'])


def forward_progressive(head,batch):
    state=initial_state(head,batch); states=[state]
    for d in range(head.config.max_future_depth):
        state=head.step(state,batch['future_tokens'][:,:,d].detach(),batch['future_geometry'][:,:,d].detach(),
            batch['future_valid_mask'][:,:,d],batch['future_token_mask'][:,:,d])
        states.append(state)
    return states


def full_scores(batch,cumulative_delta):
    result=batch['base_logits'].float().clone()
    rows,slots=batch['candidate_present_mask'].nonzero(as_tuple=True)
    indices=batch['topk_base_indices'][rows,slots]
    result[rows,indices]=result[rows,indices]+cumulative_delta[rows,slots].float()
    return result.masked_fill(~batch['ghost_valid_mask'],-torch.inf)


def full_state_scores(batch,state):
    """Scatter exact sequential FP32 scores, without reconstructing a residual."""
    result=batch['base_logits'].float().clone()
    rows,slots=batch['candidate_present_mask'].nonzero(as_tuple=True)
    result[rows,batch['topk_base_indices'][rows,slots]]=state.scores[rows,slots]
    return result.masked_fill(~batch['ghost_valid_mask'],-torch.inf)


def progressive_loss(head,batch,*,final_loss_weight=1.,prefix_loss_weight=.2,increment_regularization=.001):
    if any(not math.isfinite(v) or v < 0 for v in (final_loss_weight,prefix_loss_weight,increment_regularization)):
        raise ValueError('invalid progressive loss weights')
    states=forward_progressive(head,batch); losses=[]
    config=replace(LOSS_CONFIG,regularization_weight=0.,absent_noop_weight=0.,residual_bound=max(head.config.total_residual_bound,torch.finfo(torch.float32).eps),
                   decision_row_policy='all_move_with_future')
    for d,state in enumerate(states[1:]):
        available=batch['future_valid_mask'][:,:,:d+1].any(-1) & batch['candidate_present_mask']
        result=offline_decision_aware_loss(state.cumulative_delta,batch['teacher_rank_in_topk'],
            topk_valid_mask=available,config=config,**{k:batch[k] for k in ('teacher_valid','teacher_stop','no_vp_left','base_stop','base_logits','ghost_valid_mask','topk_base_indices','teacher_base_index')})
        losses.append(result.loss)
    # Penalize each submitted increment exactly once, not cumulative deltas at every prefix.
    increments=torch.stack([state.delta for state in states[1:]],-1)
    valid=batch['future_valid_mask'] & batch['candidate_present_mask'][:,:,None]
    regularizer=increments[valid].square().mean() if valid.any() else increments.sum()*0
    prefix=torch.stack(losses[:-1]).mean() if len(losses)>1 else losses[-1]*0
    loss=final_loss_weight*losses[-1]+prefix_loss_weight*prefix+increment_regularization*regularizer
    # All-empty batches still support backward, without unfreezing any source model.
    loss=loss+next(head.parameters()).sum()*0
    return loss,states


def source_contract(provenance):
    validate_provenance(provenance)
    return {k:provenance[k] for k in ('prediction_contract','stage1_sha256','assets','context_contract','feature_space','feature_contract','geometry_definition','distance_scale','max_future_depth')}


def save_checkpoint(path,head,provenance,*,optimizer=None,step=0,cursor=None,loss_config=None,data_sources=None):
    contract=source_contract(provenance)
    if head.config.max_future_depth != contract['max_future_depth'] or any(contract['feature_contract'][k] != head.config.input_dim for k in ('feature_dim','owner_dim','text_dim')):
        raise ValueError('model/data feature or depth differs')
    payload=dict(model_type=MODEL_TYPE,format_version=TRAINING_FORMAT if optimizer is not None else HEAD_FORMAT,
        model_config=asdict(head.config),geometry_definition=GEOMETRY,data_format=FORMAT,source_contract=contract,
        training_provenance=provenance,training_data_sources=data_sources or [],loss_config=loss_config or {},global_step=step,
        head_state_dict={k:v.detach().cpu() for k,v in head.state_dict().items()})
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(),scheduler=None,cursor=cursor,rng=rng_state())
    path=Path(path)
    if path.exists(): raise FileExistsError('refusing to overwrite checkpoint: '+str(path))
    atomic_torch_save(path,payload)
    return payload


def load_checkpoint(path,*,expected_provenance,device='cpu',head=None,optimizer=None,resume=False,expected_loss_config=None,expected_data_sources=None):
    from .progressive_head import ProgressiveHeadConfig,ProgressiveE24Head
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if obj.get('model_type') != MODEL_TYPE or obj.get('format_version') not in (HEAD_FORMAT,TRAINING_FORMAT) or obj.get('data_format') != FORMAT or obj.get('geometry_definition') != GEOMETRY:
        raise ValueError('not a progressive checkpoint or unsupported version')
    if obj['source_contract'] != source_contract(expected_provenance):
        raise ValueError('checkpoint source assets/features/geometry/depth mismatch')
    config=ProgressiveHeadConfig(**obj['model_config'])
    if config.max_future_depth != obj['source_contract']['max_future_depth'] or any(obj['source_contract']['feature_contract'][k] != config.input_dim for k in ('feature_dim','owner_dim','text_dim')):
        raise ValueError('checkpoint internal feature/depth contract mismatch')
    if head is not None and asdict(head.config) != asdict(config): raise ValueError('checkpoint model/budget mismatch')
    if head is None: head=ProgressiveE24Head(config)
    if resume:
        if obj['format_version'] != TRAINING_FORMAT or optimizer is None: raise ValueError('training state/optimizer required for resume')
        if obj['training_provenance'] != expected_provenance or obj['loss_config'] != (expected_loss_config or {}) or obj['training_data_sources'] != (expected_data_sources or []):
            raise ValueError('resume data/loss/provenance mismatch')
        if obj['scheduler'] is not None: raise ValueError('unsupported scheduler state')
    head.load_state_dict(obj['head_state_dict'],strict=True); head.to(device)
    if resume:
        optimizer.load_state_dict(obj['optimizer']); restore_rng(obj['rng'])
    return head,obj
