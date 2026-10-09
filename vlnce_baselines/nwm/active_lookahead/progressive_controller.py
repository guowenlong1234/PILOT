"""Whole-row causal stopping using exactly the FP32 scores used by navigation."""
from dataclasses import asdict
import math
import time
import torch
from .progressive_core import certify_decision


def initialize_row(head, row, device):
    indices=row['topk_base_indices'].to(device)
    base=row['base_logits'].to(device).float()
    lp=torch.log_softmax(base,0) if base.numel() else base
    present=row['candidate_present_mask'].to(device).bool()
    selected=indices.clamp_min(0)
    selected_base=base[selected] if base.numel() else base.new_zeros(len(indices))
    selected_lp=lp[selected] if base.numel() else base.new_zeros(len(indices))
    return head.initialize_state(row['owner_embeddings'][None].to(device).float(),
        row['text_tokens'][None].to(device).float(),selected_base[None],present[None],
        candidate_q0_geometry=row['candidate_q0_geometry'][None].to(device).float(),
        base_log_probs=selected_lp[None])


class ProgressiveController:
    def __init__(self, head, pruning_mode='certified'):
        if pruning_mode not in ('certified','none'):
            raise ValueError('pruning_mode must be certified or none')
        if head.training:
            raise ValueError('online controller requires eval mode')
        self.head=head
        self.pruning_mode=pruning_mode

    @torch.no_grad()
    def run(self, predictor, decision):
        device=next(self.head.parameters()).device
        def synchronize():
            if device.type=='cuda': torch.cuda.synchronize(device)
        synchronize(); started=time.perf_counter()
        states=[initialize_row(self.head,row,device) for row in decision.rows]
        scores=[row['base_logits'].to(device).float().clone() for row in decision.rows]
        terminal=[~row['candidate_present_mask'].to(device).bool() for row in decision.rows]
        records=[]
        active=[]
        for i,row in enumerate(decision.rows):
            records.append(dict(executed_depth=0,stop_reason=None,per_depth_delta=[],cumulative_delta=[],
                base_winner=int(scores[i].argmax()) if scores[i].numel() else None,
                prefix_winner=[],final_winner=None,remaining_budget=[],certificate_margin=None,
                numerical_guard=None,certificates=[],per_depth_world_model_queries=[],
                per_depth_valid_futures=[],per_depth_terminal_reasons=[],prediction_seconds=0.,scoring_seconds=0.))
            if row['base_stop']:
                records[i]['stop_reason']='forced_stop' if row.get('forced_stop') else 'base_stop'
            elif not scores[i].numel():
                records[i]['stop_reason']='no_candidates'
            elif not self._check(i,0,decision,states,scores,terminal,records):
                active.append(i)
        for depth in range(1,self.head.config.max_future_depth+1):
            if not active: break
            synchronize(); before=time.perf_counter()
            layers=predictor.predict_next_depth(decision,active_rows=active)
            synchronize(); predicted=time.perf_counter()
            next_active=[]
            for i in active:
                row, layer, record=decision.rows[i],layers[i],records[i]
                current=time.perf_counter()
                if (terminal[i] & layer['future_valid_mask'].to(device)).any():
                    raise ValueError('terminated progressive branch cannot submit new evidence')
                states[i]=self.head.step(states[i],layer['future_tokens'][None].to(device).float(),
                    layer['future_geometry'][None].to(device).float(),
                    layer['future_valid_mask'][None].to(device),layer['future_token_mask'][None].to(device))
                present=row['candidate_present_mask'].to(device).bool()
                indices=row['topk_base_indices'].to(device)[present]
                scores[i][indices]=states[i].scores[0,present]
                terminal[i]=terminal[i] | layer['terminal_mask'].to(device)
                synchronize()
                record['scoring_seconds']+=time.perf_counter()-current
                record['prediction_seconds']+=predicted-before
                record['executed_depth']=depth
                record['per_depth_delta'].append(states[i].delta[0].cpu().tolist())
                record['cumulative_delta'].append(states[i].cumulative_delta[0].cpu().tolist())
                record['prefix_winner'].append(int(scores[i].argmax()))
                record['per_depth_world_model_queries'].append(layer['world_model_queries'])
                record['per_depth_valid_futures'].append(int(layer['future_valid_mask'].sum()))
                record['per_depth_terminal_reasons'].append(layer.get('terminal_reason',[]))
                stopped=self._check(i,depth,decision,states,scores,terminal,records)
                if depth==self.head.config.max_future_depth:
                    record['stop_reason']='max_depth'
                elif terminal[i].all():
                    record['stop_reason']='natural_terminal'
                elif not stopped:
                    next_active.append(i)
            active=next_active
        synchronize(); elapsed=time.perf_counter()-started
        for i,record in enumerate(records):
            record['final_winner']=int(scores[i].argmax()) if scores[i].numel() else None
            record['current_scores']=scores[i].cpu().tolist()
            record['total_stage2_seconds']=elapsed
            if record['stop_reason'] is None: record['stop_reason']='max_depth'
        return dict(scores=scores,states=states,diagnostics=records)

    def _check(self,i,depth,decision,states,scores,terminal,records):
        row=decision.rows[i]
        remaining=torch.zeros_like(scores[i],dtype=torch.float64)
        present=row['candidate_present_mask'].to(scores[i].device).bool()
        indices=row['topk_base_indices'].to(scores[i].device)[present]
        budget=math.fsum(self.head.budgets[depth:])
        remaining[indices]=(~terminal[i][present]).double()*budget
        proof=certify_decision(scores[i],remaining)
        record=records[i]
        record['remaining_budget'].append(remaining.cpu().tolist())
        record['certificates'].append(dict(depth=depth,**asdict(proof)))
        record['certificate_margin']=proof.certificate_margin
        record['numerical_guard']=proof.numerical_guard
        if self.pruning_mode=='certified' and proof.certified:
            record['stop_reason']='certified'
            return True
        return False
