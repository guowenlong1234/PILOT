"""Verify real E24 forward/loss compatibility without training or backward."""
import argparse,json
from pathlib import Path
import torch
from vlnce_baselines.nwm.active_lookahead.stage2_data import Stage2Dataset,collate_stage2,atomic_json
from vlnce_baselines.nwm.active_lookahead.residual_head import InterleavedCrossModalTopKFutureLogitResidualHead
from vlnce_baselines.nwm.active_lookahead.offline_objective import offline_decision_aware_loss,OfflineDecisionLossConfig

def main():
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+');p.add_argument('--device',default='cpu');p.add_argument('--report',required=True);a=p.parse_args()
    d=Stage2Dataset(a.roots); rows=[]
    for i in range(len(d)):
        r=d[i]
        if r['future_valid_mask'].any() and r['teacher_valid'] and not r['base_stop'] and not r['teacher_stop']:
            rows.append(r)
        if len(rows)==2:break
    if not rows:raise RuntimeError('no usable predicted future rows')
    b={k:v.to(a.device) for k,v in collate_stage2(rows).items()}
    head=InterleavedCrossModalTopKFutureLogitResidualHead().to(a.device).eval()
    with torch.no_grad():
        probabilities=torch.log_softmax(b['base_logits'].masked_fill(~b['ghost_valid_mask'],-torch.inf),dim=1)
        selected=probabilities.gather(1,b['topk_base_indices'].clamp_min(0)).masked_fill(~b['topk_valid_mask'],0)
        delta=head.forward_topk_from_log_probs(b['owner_embeddings'],b['text_tokens'],b['future_tokens'],selected,b['topk_valid_mask'],text_token_mask=b['text_token_mask'],candidate_geometry=b['candidate_q0_geometry']).delta
        assert torch.equal(delta,torch.zeros_like(delta))
        loss=offline_decision_aware_loss(delta,b['teacher_rank_in_topk'],**{k:b[k] for k in ('topk_valid_mask','teacher_valid','teacher_stop','no_vp_left','base_stop','base_logits','ghost_valid_mask','topk_base_indices','teacher_base_index')},config=OfflineDecisionLossConfig(regularization_weight=.001,absent_noop_weight=.05,correct_row_weight=2.,wrong_row_weight=1.))
        assert torch.isfinite(loss.loss)
    report=dict(status='ready',dataset_rows=len(d),checked_rows=len(rows),future_shape=list(b['future_tokens'].shape),parameters=sum(p.numel() for p in head.parameters()),zero_initialized_delta=True,loss=float(loss.loss),backward_called=False,optimizer_created=False,training_started=False)
    atomic_json(a.report,report);print(json.dumps(report))
if __name__=='__main__':main()
