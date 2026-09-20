"""Deploy the predicted-future offline head without any teacher/goal inputs."""
import json
import time
from pathlib import Path
import torch
from .stage2_collect import Stage2Collector
from .stage2_data import collate_stage2, load, sha256, atomic_json
from .stage2_training import MODEL_CONFIG
from .residual_head import InterleavedCrossModalTopKFutureLogitResidualHead
from .base_freeze import capture_base_tensor_manifest

BASE_SHA = '4c729c84bf4338452da4d459fc82734dcbb5f72ac6a2b574ee8f20e1080bc2fe'
BEST_SHA = '598986525cb3ac4696743b0480ac6733b918e3d4407d647c4cddb0a498ae286e'


def score_rows(head, rows, device, gain):
    """Identical fp16-cache roundtrip and BF16 inference to the offline evaluator."""
    batch = {k:v.to(device) for k,v in collate_stage2(rows, include_targets=False).items()}
    logits = batch['base_logits'].masked_fill(~batch['ghost_valid_mask'], -torch.inf)
    logits[~batch['ghost_valid_mask'].any(1)] = 0
    selected = torch.log_softmax(logits,1).gather(1,batch['topk_base_indices'].clamp_min(0))
    selected = selected.masked_fill(~batch['topk_valid_mask'],0)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                        enabled=torch.device(device).type=='cuda'):
        delta = head.forward_topk_from_log_probs(batch['owner_embeddings'],batch['text_tokens'],
            batch['future_tokens'],selected,batch['topk_valid_mask'],
            text_token_mask=batch['text_token_mask'],candidate_geometry=batch['candidate_q0_geometry']).delta
    delta = (gain*delta.float()).clamp(-1,1).masked_fill(~batch['topk_valid_mask'],0)
    if not torch.isfinite(delta).all():
        raise FloatingPointError('nonfinite online E24 residual')
    return delta


class Stage2Online(Stage2Collector):
    def __init__(self, trainer):
        cfg=trainer.config.MODEL.STAGE2_ONLINE
        if cfg.gain not in (0.,1.5):
            raise ValueError('online comparison is locked to zero or selected gain1.5')
        if cfg.seed != 20260916:
            raise ValueError('online future-noise seed differs from collection')
        if cfg.head_sha256 != BEST_SHA or sha256(cfg.head) != BEST_SHA:
            raise ValueError('selected 4750 head SHA differs')
        if sha256(trainer.config.EVAL.CKPT_PATH_DIR) != BASE_SHA:
            raise ValueError('online stage1 base differs from offline collection')
        checkpoint=load(cfg.head)
        if checkpoint['model_config'] != MODEL_CONFIG or checkpoint['global_step'] != 4750:
            raise ValueError('unexpected selected head configuration')
        for part in checkpoint['contract']['dataset']:
            p=part['provenance']
            if p['stage1_sha256']!=BASE_SHA or p['feature_space']!='raw_cls+normalized_patch_fp16' or p['context_contract']!='stage2_panorama_q0_snapshot_v1':
                raise ValueError('offline head feature/base contract differs')
            for path,digest in p['assets'].items():
                if sha256(path)!=digest: raise ValueError('online prediction asset differs: '+path)
        super().__init__(trainer,prediction_only=True)
        with torch.random.fork_rng():
            self.head=InterleavedCrossModalTopKFutureLogitResidualHead(**checkpoint['model_config']).to(trainer.device).eval()
            self.head.load_state_dict(checkpoint['future_head_state_dict'],strict=True)
        self.head.requires_grad_(False)
        self.modules['head']=self.head
        self.before=capture_base_tensor_manifest(self.modules)
        self.versions['head']=[(t,t._version) for t in list(self.head.parameters())+list(self.head.buffers())]
        self.episode_diagnostics={}
        self.head_stream=torch.cuda.Stream(device=trainer.device)
        self.started=time.perf_counter(); self.online_seconds=0.
        Path(cfg.output).mkdir(parents=True,exist_ok=True)
        self.trace_file=(Path(cfg.output)/'decisions.jsonl').open('x') if cfg.trace else None

    @torch.no_grad()
    def score_step(self,nav_inputs,nav_outs,text,text_mask,no_vp_left,step):
        start=time.perf_counter()
        rows=self.predict_step(nav_inputs,nav_outs,text,text_mask,no_vp_left,step)
        for i,row in enumerate(rows):
            row['text_tokens']=text[i][text_mask[i].bool()].detach().cpu().half()
        cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state(self.trainer.device)
        backend=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                 torch.is_autocast_enabled(),torch.get_autocast_gpu_dtype())
        current=torch.cuda.current_stream(self.trainer.device)
        self.head_stream.wait_stream(current)
        with torch.cuda.stream(self.head_stream):
            delta=score_rows(self.head,rows,self.trainer.device,float(self.cfg.gain))
        current.wait_stream(self.head_stream)
        delta.record_stream(current)
        if not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(cuda_rng,torch.cuda.get_rng_state(self.trainer.device)):
            raise RuntimeError('E24 head consumed global RNG')
        if backend!=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32,
                    torch.is_autocast_enabled(),torch.get_autocast_gpu_dtype()):
            raise RuntimeError('E24 head changed global precision state')
        dense=torch.zeros_like(nav_outs['global_logits'])
        episodes=self.trainer.envs.current_episodes()
        for i,row in enumerate(rows):
            valid=row['future_valid_mask']; k=len(valid)
            if not row['base_stop']:
                indices=[row['global_indices'][int(row['topk_base_indices'][j])] for j in range(k)]
                dense[i,indices]=delta[i,:k].to(dense.dtype)
            scores=nav_outs['global_logits'][i].detach()
            base=int(scores.argmax()); ghosts=row['global_indices']
            chosen=base if row['base_stop'] else ghosts[int((scores[ghosts]+dense[i,ghosts]).argmax())]
            self.counts['rows']+=1; self.counts['action_flips']+=int(chosen!=base)
            self.counts['future_valid_slots']+=int(valid.sum())
            episode=str(episodes[i].episode_id)
            diagnostic=self.episode_diagnostics.setdefault(episode,dict(scene=str(episodes[i].scene_id),
                decisions=0,action_flips=0,future_valid_slots=0,invalid_reasons={}))
            diagnostic['decisions']+=1; diagnostic['action_flips']+=int(chosen!=base)
            diagnostic['future_valid_slots']+=int(valid.sum())
            for reason in row['invalid_reason']:
                diagnostic['invalid_reasons'][reason]=diagnostic['invalid_reasons'].get(reason,0)+1
            if row['base_stop'] and bool(dense[i].any()): raise RuntimeError('STOP residual must be zero')
            if self.trace_file:
                self.trace_file.write(json.dumps(dict(episode=episode,
                    step=step,base_action=base,action=chosen,forced_stop=row['forced_stop'],
                    logits=scores[:len(nav_inputs['gmap_vp_ids'][i])].cpu().tolist(),
                    delta=dense[i,:len(nav_inputs['gmap_vp_ids'][i])].cpu().tolist()))+'\n')
                self.trace_file.flush()
        self.online_seconds+=time.perf_counter()-start
        return dense

    def complete_episode(self,episode,metrics):
        for name,tensors in self.versions.items():
            if any(t._version!=version for t,version in tensors):
                raise RuntimeError('frozen online tensor mutated: '+name)
        self.episode_diagnostics[episode]['completed']=True
        self.counts['episodes']+=1
        atomic_json(Path(self.cfg.output)/'progress.json',dict(self.counts))

    def finish(self):
        super().finish()
        atomic_json(Path(self.cfg.output)/'episode_diagnostics.json',self.episode_diagnostics)
        if self.trace_file: self.trace_file.close()
        atomic_json(Path(self.cfg.output)/'online_summary.json',dict(counts=dict(self.counts),
            elapsed_seconds=time.perf_counter()-self.started,online_seconds=self.online_seconds,
            head_sha256=BEST_SHA,gain=self.cfg.gain,residual_bound=1.,teacher_calls=0))
