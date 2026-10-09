"""Strict progressive deployment adapter; legacy Stage2Online stays untouched."""
import json
from pathlib import Path
import torch
from .stage2_collect import Stage2Collector
from .stage2_data import sha256, atomic_json
from .base_freeze import capture_base_tensor_manifest
from .progressive_contract import PREDICTION_CONTRACT
from .progressive_core import validate_budget
from .progressive_prediction import ProgressivePrediction, FUTURE_TOKEN_COUNT, FUTURE_FEATURE_DIM
from .progressive_controller import ProgressiveController
from .progressive_training import load_checkpoint
from .progressive_data import FORMAT, SPACE, GEOMETRY, validate_provenance


def live_provenance(trainer, depth, distance_scale):
    """Compute actual deployment identity, never trust a training asset path."""
    cfg=trainer.config
    assets={}
    for name,path in dict(world_model=cfg.MODEL.RAENWM.checkpoint_path,
        statistics=cfg.MODEL.RAENWM.stat_path,world_config=cfg.MODEL.RAENWM.config_path,
        cwp=cfg.MODEL.ACTIVE_LOOKAHEAD.dino_cwp_checkpoint_path).items():
        if not path: raise ValueError('missing progressive prediction asset: '+name)
        assets[name]=dict(sha256=sha256(path))
    pc=dict(PREDICTION_CONTRACT)
    mode=cfg.MODEL.STAGE2_ONLINE if cfg.MODEL.STAGE2_ONLINE.enabled else cfg.MODEL.STAGE2_COLLECT
    pc['seed']=int(mode.seed)
    for name in ('context_size','num_steps','final_only_euler','max_horizon','condition_source_pose','panorama_context_mode'):
        pc[name]=getattr(cfg.MODEL.RAENWM,name)
    p=dict(prediction_contract=pc,format=FORMAT,feature_space=SPACE,geometry_definition=GEOMETRY,
        distance_scale=float(distance_scale),max_future_depth=int(depth),
        feature_contract=dict(token_count=FUTURE_TOKEN_COUNT,feature_dim=FUTURE_FEATURE_DIM,
                              owner_dim=FUTURE_FEATURE_DIM,text_dim=FUTURE_FEATURE_DIM),
        stage1_sha256=sha256(cfg.EVAL.CKPT_PATH_DIR),assets=assets,
        context_contract='stage2_panorama_q0_snapshot_v1',behavior='stage1_argmax',split=cfg.EVAL.SPLIT)
    validate_provenance(p)
    return p


def validate_progressive_config(cfg, online=None):
    validate_budget(cfg.max_future_depth,cfg.total_residual_bound,cfg.budget_fractions,
                    deployment_gain=1. if online is None else online.gain)
    if cfg.max_future_depth not in (1,2,3): raise ValueError('supported physical depth is 1..3')
    if cfg.pruning_mode not in ('none','certified'): raise ValueError('invalid pruning_mode')
    if not cfg.share_depth_weights: raise ValueError('depth weights must be shared')
    if online is not None and online.margin_threshold != -1:
        raise ValueError('progressive requires margin_threshold=-1')


class ProgressiveOnline(Stage2Collector):
    def __init__(self,trainer):
        cfg=trainer.config.MODEL.STAGE2_ONLINE
        pcfg=trainer.config.MODEL.PROGRESSIVE
        validate_progressive_config(pcfg,cfg)
        if sha256(cfg.head)!=cfg.head_sha256: raise ValueError('progressive head SHA mismatch')
        provenance=live_provenance(trainer,pcfg.max_future_depth,pcfg.distance_scale)
        # Strict source checks occur before creating the frozen prediction runtime.
        with torch.random.fork_rng():
            head,checkpoint=load_checkpoint(cfg.head,expected_provenance=provenance,device=trainer.device)
        for key in ('max_future_depth','total_residual_bound','share_depth_weights',
                    'use_depth_embedding','use_geometry_embedding'):
            if getattr(head.config,key)!=getattr(pcfg,key): raise ValueError('checkpoint/config mismatch: '+key)
        if tuple(head.config.budget_fractions)!=tuple(pcfg.budget_fractions):
            raise ValueError('checkpoint/config budget mismatch')
        if checkpoint['training_provenance']['split']!='train':
            raise ValueError('progressive head requires train split provenance')
        super().__init__(trainer,prediction_only=True)
        self.head=head.eval().requires_grad_(False)
        self.modules['head']=head
        self.before=capture_base_tensor_manifest(self.modules)
        self.versions['head']=[(t,t._version) for t in list(head.parameters())+list(head.buffers())]
        self.predictor=ProgressivePrediction(self,pcfg.max_future_depth,pcfg.distance_scale)
        self.controller=ProgressiveController(head,pcfg.pruning_mode)
        self.episode_diagnostics={}
        Path(cfg.output).mkdir(parents=True,exist_ok=True)
        self.trace_file=(Path(cfg.output)/'decisions.jsonl').open('x')
        self.action_scores=None

    @torch.no_grad()
    def score_step(self,nav_inputs,nav_outs,text,text_mask,no_vp_left,step):
        try:
            decision=self.predictor.initialize_decision(nav_inputs,nav_outs,text,text_mask,no_vp_left,step)
            result=self.controller.run(self.predictor,decision)
        except Exception as exc:
            self.trace_file.write(json.dumps(dict(step=step,stop_reason='exception_failure',error=str(exc)))+'\n')
            self.trace_file.flush()
            raise
        self.action_scores=nav_outs['global_logits'].detach().float().clone()
        actions=nav_outs['global_logits'].detach().argmax(-1).clone()
        for i,(row,scores,record) in enumerate(zip(decision.rows,result['scores'],result['diagnostics'])):
            if not row['base_stop']:
                self.action_scores[i,row['global_indices']]=scores
                actions[i]=row['global_indices'][int(scores.argmax())]
            record.update(step=step,episode=str(self.trainer.envs.current_episodes()[i].episode_id),
                scene=str(self.trainer.envs.current_episodes()[i].scene_id),
                ghost_ids=row['ghost_ids'],topk_base_indices=row['topk_base_indices'].tolist(),
                executed_action=0 if row['forced_stop'] else int(actions[i]),
                score_semantics='executed_prefix_logits',schema='progressive-decision-v1')
            self.trace_file.write(json.dumps(record)+'\n')
            self.counts['rows']+=1
            self.counts['depth_sum']+=record['executed_depth']
        self.trace_file.flush()
        decision.closed=True
        self.actions=actions
        # Returned delta is diagnostic only; trainer uses exact certified FP32 scores/actions.
        dense=torch.zeros_like(self.action_scores)
        for i,row in enumerate(decision.rows):
            if not row['base_stop']:
                ix=row['global_indices']
                dense[i,ix]=self.action_scores[i,ix]-nav_outs['global_logits'][i,ix].float()
        return dense

    def complete_episode(self,episode,metrics):
        for name,tensors in self.versions.items():
            if any(t._version!=version for t,version in tensors): raise RuntimeError('frozen tensor mutated: '+name)
        self.counts['episodes']+=1
        self.episode_diagnostics[episode]=metrics

    def finish(self):
        super().finish()
        self.trace_file.close()
        atomic_json(Path(self.cfg.output)/'episode_diagnostics.json',self.episode_diagnostics)
        atomic_json(Path(self.cfg.output)/'online_summary.json',dict(self.counts))
