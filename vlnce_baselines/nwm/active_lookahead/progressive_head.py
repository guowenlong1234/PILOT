"""Causal E24 fusion shared over physical successor depths."""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from .progressive_core import validate_budget
from .residual_head import _InterleavedCrossModalBlock


@dataclass(frozen=True)
class ProgressiveHeadConfig:
    input_dim: int = 768
    hidden_dim: int = 768
    num_queries: int = 8
    num_attention_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = .1
    fusion_layers: int = 3
    max_future_depth: int = 2
    total_residual_bound: float = 1.
    budget_fractions: tuple = (.5, .5)
    share_depth_weights: bool = True
    use_depth_embedding: bool = True
    use_geometry_embedding: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'budget_fractions', tuple(self.budget_fractions))
        validate_budget(self.max_future_depth, self.total_residual_bound, self.budget_fractions)
        if self.max_future_depth not in (1,2,3):
            raise ValueError('supported physical depth is 1..3')
        if not self.share_depth_weights:
            raise ValueError('progressive v1 requires shared depth weights')
        if min(self.input_dim,self.hidden_dim,self.num_queries,self.num_attention_heads,self.ffn_dim,self.fusion_layers)<=0 or self.hidden_dim % self.num_attention_heads:
            raise ValueError('invalid model dimensions')


@dataclass
class ProgressiveState:
    depth: int
    memory: torch.Tensor
    base_logits: torch.Tensor
    base_log_probs: torch.Tensor
    scores: torch.Tensor
    cumulative_delta: torch.Tensor
    delta: torch.Tensor
    candidate_present_mask: torch.Tensor
    text_tokens: torch.Tensor
    text_token_mask: Optional[torch.Tensor]


@dataclass
class ProgressiveOutput:
    states: list
    per_depth_delta: torch.Tensor
    cumulative_delta: torch.Tensor
    scores: torch.Tensor
    memory: torch.Tensor


class ProgressiveE24Head(nn.Module):
    model_type = 'progressive_e24_v1'
    format_version = 1

    def __init__(self, config: ProgressiveHeadConfig = None):
        super().__init__()
        self.config = config or ProgressiveHeadConfig()
        c=self.config
        self.budgets=validate_budget(c.max_future_depth,c.total_residual_bound,c.budget_fractions)
        self.owner_norm=nn.LayerNorm(c.input_dim)
        self.owner_proj=nn.Linear(c.input_dim,c.hidden_dim)
        self.base_context_proj=nn.Linear(1,c.hidden_dim,bias=False)
        self.q0_geometry_proj=nn.Linear(3,c.hidden_dim,bias=False)
        self.text_proj=nn.Linear(c.input_dim,c.hidden_dim)
        self.future_proj=nn.Linear(c.input_dim,c.hidden_dim)
        self.depth_embedding=nn.Embedding(c.max_future_depth+1,c.hidden_dim)
        self.geometry_mlp=nn.Sequential(nn.Linear(5,c.hidden_dim),nn.GELU(),nn.Linear(c.hidden_dim,c.hidden_dim))
        self.memory_proj=nn.Linear(c.hidden_dim,c.hidden_dim)
        self.learned_queries=nn.Parameter(torch.empty(c.num_queries,c.hidden_dim))
        nn.init.normal_(self.learned_queries,std=.02)
        self.fusion_blocks=nn.ModuleList([_InterleavedCrossModalBlock(hidden_dim=c.hidden_dim,num_attention_heads=c.num_attention_heads,ffn_dim=c.ffn_dim,dropout=c.dropout) for _ in range(c.fusion_layers)])
        self.candidate_layers=nn.ModuleList([nn.TransformerEncoderLayer(c.hidden_dim,c.num_attention_heads,c.ffn_dim,c.dropout,activation='gelu',batch_first=True) for _ in range(c.fusion_layers)])
        self.feedback_proj=nn.ModuleList([nn.Linear(c.hidden_dim,c.hidden_dim) for _ in range(c.fusion_layers-1)])
        self.feedback_norm=nn.ModuleList([nn.LayerNorm(c.hidden_dim) for _ in range(c.fusion_layers-1)])
        self.score_mlp=nn.Sequential(nn.Linear(c.hidden_dim*2+3,c.hidden_dim),nn.GELU(),nn.Linear(c.hidden_dim,1))
        nn.init.zeros_(self.score_mlp[-1].weight)
        nn.init.zeros_(self.score_mlp[-1].bias)

    def initialize_state(self, owner_embeddings, text_tokens, base_logits, candidate_present_mask, *, candidate_q0_geometry, base_log_probs=None, text_token_mask=None):
        b,k,c=owner_embeddings.shape
        present=candidate_present_mask.detach().bool()
        if c != self.config.input_dim or present.shape!=(b,k) or base_logits.shape!=(b,k) or candidate_q0_geometry.shape!=(b,k,3):
            raise ValueError('invalid progressive initialization shapes')
        if text_tokens.ndim not in (3,4) or text_tokens.shape[0]!=b or (text_tokens.ndim==4 and text_tokens.shape[1]!=k):
            raise ValueError('text must be [B,L,C] or [B,K,L,C]')
        base=base_logits.detach().float().masked_fill(~present,0)
        if base_log_probs is not None and base_log_probs.shape != (b,k):
            raise ValueError('base log probabilities must match candidates')
        lp=base if base_log_probs is None else base_log_probs.detach().float().masked_fill(~present,0)
        owner=owner_embeddings.detach()[present]
        geom=candidate_q0_geometry.detach()[present]
        if any(not torch.isfinite(x).all() for x in (owner,geom,base[present],lp[present])):
            raise ValueError('present candidate inputs must be finite')
        zero=next(self.parameters()).flatten()[0]*0
        memory=self.owner_proj.weight.new_zeros(b,k,self.config.hidden_dim)+zero
        dtype=self.owner_proj.weight.dtype
        memory[present]=self.owner_proj(self.owner_norm(owner.to(dtype)))+self.q0_geometry_proj(geom.to(dtype))+self.base_context_proj(lp[present].unsqueeze(-1).to(dtype))
        delta=base.new_zeros(b,k)+zero.float()
        return ProgressiveState(0,memory,base,lp,base+delta,delta,delta,present,text_tokens.detach(),None if text_token_mask is None else text_token_mask.detach().bool())

    def step(self, state, future_tokens, future_geometry, future_valid_mask, future_token_mask=None):
        d=state.depth+1
        if d>self.config.max_future_depth:
            raise ValueError('maximum physical depth exceeded')
        b,k=state.scores.shape
        if future_tokens.ndim!=4 or future_tokens.shape[:2]!=(b,k) or future_tokens.shape[-1]!=self.config.input_dim or future_geometry.shape!=(b,k,5) or future_valid_mask.shape!=(b,k):
            raise ValueError('invalid single-depth shapes')
        valid=future_valid_mask.detach().bool()
        if (valid & ~state.candidate_present_mask).any():
            raise ValueError('valid future requires present candidate')
        zero=next(self.parameters()).flatten()[0]*0
        delta=state.scores.new_zeros(b,k)+zero.float()
        memory=state.memory
        rows,slots=valid.nonzero(as_tuple=True)
        if rows.numel():
            tokens=future_tokens.detach()[rows,slots]
            geometry=future_geometry.detach()[rows,slots]
            mask=torch.ones(tokens.shape[:2],dtype=torch.bool,device=tokens.device) if future_token_mask is None else future_token_mask.detach().bool()[rows,slots]
            if tokens.shape[1]<2 or not mask[:,0].all() or not mask[:,1:].any(-1).all():
                raise ValueError('valid future requires CLS and at least one patch')
            if not torch.isfinite(tokens[mask]).all() or not torch.isfinite(geometry).all():
                raise ValueError('valid future values must be finite')
            dtype=self.future_proj.weight.dtype
            clean=tokens.masked_fill(~mask[...,None],0).to(dtype)
            features=self.future_proj(clean)
            depth=self.depth_embedding.weight[d] if self.config.use_depth_embedding else torch.zeros_like(self.depth_embedding.weight[d])
            features=features+depth
            if self.config.use_geometry_embedding:
                features=features+self.geometry_mlp(geometry.to(dtype))[:,None]
            text=state.text_tokens[rows,slots] if state.text_tokens.ndim==4 else state.text_tokens[rows]
            tm=state.text_token_mask
            tm=torch.ones(text.shape[:2],device=text.device,dtype=torch.bool) if tm is None else (tm[rows,slots] if tm.ndim==3 else tm[rows])
            if not tm.any(-1).all() or not torch.isfinite(text[tm]).all():
                raise ValueError('instruction requires finite valid tokens')
            text_h=self.text_proj(text.masked_fill(~tm[...,None],0).to(dtype))
            queries=self.learned_queries[None]+self.memory_proj(memory[rows,slots])[:,None]
            active_rows=valid.any(-1).nonzero(as_tuple=True)[0]
            for layer,(fusion,comparison) in enumerate(zip(self.fusion_blocks,self.candidate_layers)):
                queries=fusion(queries,text_h,features[:,1:],features[:,0],text_padding_mask=~tm,patch_padding_mask=~mask[:,1:])
                comparison_input=memory.clone()
                comparison_input[rows,slots]=queries.mean(1)
                compared=comparison(comparison_input[active_rows],src_key_padding_mask=~state.candidate_present_mask[active_rows])
                dense=memory.clone()
                dense[active_rows]=compared
                if layer+1<len(self.fusion_blocks):
                    queries=self.feedback_norm[layer](queries+self.feedback_proj[layer](dense[rows,slots])[:,None])
            memory=memory.clone()
            memory[rows,slots]=dense[rows,slots]
            context=torch.stack((state.base_log_probs[rows,slots],state.cumulative_delta[rows,slots],state.scores[rows,slots]),-1).to(dtype)
            score_input=torch.cat((memory[rows,slots],depth.expand(len(rows),-1),context),-1)
            raw=self.score_mlp(score_input).squeeze(-1).float()
            if not torch.isfinite(raw).all():
                raise FloatingPointError('nonfinite progressive score increment')
            delta[rows,slots]=self.budgets[d-1]*torch.tanh(raw)
        cumulative=state.cumulative_delta.float()+delta
        # This exact sequential FP32 sum is also the controller's score source.
        scores=state.scores.float()+delta
        return ProgressiveState(d,memory,state.base_logits,state.base_log_probs,scores,cumulative,delta,state.candidate_present_mask,state.text_tokens,state.text_token_mask)

    def forward(self, owner_embeddings,text_tokens,base_logits,future_tokens,future_geometry,future_valid_mask,candidate_present_mask,*,candidate_q0_geometry,base_log_probs=None,text_token_mask=None,future_token_mask=None):
        if future_tokens.ndim!=5 or future_tokens.shape[2]!=self.config.max_future_depth:
            raise ValueError('full forward requires configured depth [B,K,D,T,C]')
        state=self.initialize_state(owner_embeddings,text_tokens,base_logits,candidate_present_mask,candidate_q0_geometry=candidate_q0_geometry,base_log_probs=base_log_probs,text_token_mask=text_token_mask)
        states=[state]
        for depth in range(self.config.max_future_depth):
            state=self.step(state,future_tokens[:,:,depth],future_geometry[:,:,depth],future_valid_mask[:,:,depth],None if future_token_mask is None else future_token_mask[:,:,depth])
            states.append(state)
        return ProgressiveOutput(states,torch.stack([s.delta for s in states[1:]],-1),state.cumulative_delta,state.scores,state.memory)
