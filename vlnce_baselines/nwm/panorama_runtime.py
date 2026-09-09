"""Reusable frozen NWM interface for observed full-panorama contexts.

The caller supplies actual observed cube faces. There is no simulator callback,
target RGB argument, navigation-policy mutation, or automatic history replay.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .etp_adapter import RaeGhostInputRecord, RaeNwmInputBatch
from .panorama_context import FORMAT, MODES, local_xy, make_context_plan, observed_view
from .raenwm_core.models import pack_cls_patch
from .runtime import build_native_cls_prediction
from .types import NwmCondition, NwmPrediction


@dataclass
class ObservedPanoramaFrame:
    frame_id: str
    segment_id: str
    position: np.ndarray
    body_yaw: float
    cube_rgb: np.ndarray
    native_world_rgb12: object = None
    front_rgb: object = None
    _cache: OrderedDict = field(default_factory=OrderedDict,repr=False)

    def __post_init__(self):
        self.position=np.asarray(self.position,dtype=np.float32).copy()
        self.cube_rgb=np.asarray(self.cube_rgb).copy()
        if self.position.shape!=(3,) or not np.isfinite(self.position).all() or not np.isfinite(self.body_yaw):
            raise ValueError('invalid observed pose')
        if self.cube_rgb.dtype!=np.uint8 or self.cube_rgb.ndim!=4 or self.cube_rgb.shape[0]!=6 or self.cube_rgb.shape[-1]!=3 or self.cube_rgb.shape[1]!=self.cube_rgb.shape[2]:
            raise ValueError('observed cube must be [6,H,H,3] uint8')
        self.position.flags.writeable=False;self.cube_rgb.flags.writeable=False
        if self.native_world_rgb12 is not None:
            bank=np.asarray(self.native_world_rgb12).copy()
            if bank.shape!=(12,224,224,3) or bank.dtype!=np.uint8:raise ValueError('invalid native direction bank')
            bank.flags.writeable=False;self.native_world_rgb12=bank
        if self.front_rgb is not None:
            front=np.asarray(self.front_rgb).copy()
            if front.shape!=(224,224,3) or front.dtype!=np.uint8:raise ValueError('invalid recorded front view')
            front.flags.writeable=False;self.front_rgb=front


class PanoramaHistory:
    """Own at most four observed time points; physical discontinuities reset."""
    def __init__(self):self.frames=deque(maxlen=4)

    def clear(self):self.frames.clear()

    def append(self,frame):
        if self.frames:
            last=self.frames[-1]
            if frame.segment_id!=last.segment_id:
                self.clear()
            elif frame.frame_id==last.frame_id:
                raise ValueError('one timestamp cannot be duplicated to fill history')
            elif np.linalg.norm((frame.position-last.position)[[0,2]])>.51:
                raise ValueError('position jump requires a new segment id')
        self.frames.append(frame)


@dataclass(frozen=True)
class PanoramaTarget:
    env_index: int
    query_id: str
    position: tuple
    yaw: float


class PanoramaPredictionRuntime:
    def __init__(self,*,encoder,normalizer,predictor,mode,
                 device='cuda:0',encode_batch_size=16,prediction_batch_size=8,
                 cached_views_per_frame=12,require_native_views=True):
        if mode not in MODES:raise ValueError('unknown panorama context mode')
        if min(encode_batch_size,prediction_batch_size,cached_views_per_frame)<1:raise ValueError('positive batch/cache sizes required')
        self.encoder=encoder;self.normalizer=normalizer;self.predictor=predictor
        self.mode=mode;self.device=torch.device(device)
        self.encode_batch_size=encode_batch_size;self.prediction_batch_size=prediction_batch_size
        self.cached_views_per_frame=cached_views_per_frame
        self.require_native_views=bool(require_native_views)
        self.cache_identity=object()
        self.last_diagnostics={}

    @property
    def context_metadata(self):
        return {'format':FORMAT,'mode':self.mode,'context_size':4,
                'cube_order':'yaw0_90_180_270_pitch+90_-90','source':'selected_last_camera',
                'target':'absolute_pose_unchanged','cache_dtype':'float16',
                'require_native_views':self.require_native_views,
                'view_extraction':'native_world_30deg_else_cube_bilinear'}

    @torch.no_grad()
    def predict(self,targets:Sequence[PanoramaTarget],histories,*,initial_noise=None,generator=None):
        plans=[];pending=OrderedDict();features={};skipped={};seen=set()
        for target in targets:
            key=(target.env_index,target.query_id)
            if key in seen:raise ValueError('duplicate target identity')
            seen.add(key)
            frames=list(histories[target.env_index].frames)
            if len(frames)!=4:
                skipped['context_not_ready']=skipped.get('context_not_ready',0)+1;continue
            if self.require_native_views and any(f.native_world_rgb12 is None for f in frames):
                raise ValueError('validated context requires the recorded native world direction bank')
            if self.mode=='front' and any(f.front_rgb is None for f in frames):
                raise ValueError('baseline mode requires the four directly recorded front images')
            plan=make_context_plan([f.position for f in frames],[f.body_yaw for f in frames],
                target.position,target.yaw,self.mode,[f.segment_id for f in frames])
            keys=[]
            for i in plan.order:
                frame=frames[i];yaw=plan.view_yaws[i]
                cachekey=(self.cache_identity,self.mode=='front',round(float(yaw%(2*np.pi)),8))
                batchkey=(id(frame),cachekey);keys.append(batchkey)
                if cachekey in frame._cache:
                    features[batchkey]=frame._cache[cachekey];frame._cache.move_to_end(cachekey)
                else:pending[batchkey]=(frame,cachekey,yaw)
            plans.append((target,plan,keys))
        pending_items=list(pending.items())
        for start in range(0,len(pending_items),self.encode_batch_size):
            part=pending_items[start:start+self.encode_batch_size]
            rgb=np.stack([frame.front_rgb if self.mode=='front' else
                observed_view(frame.cube_rgb,yaw,frame.native_world_rgb12) for _,(frame,_,yaw) in part])
            cls,patch=self.encoder.forward_raw_cls_and_patch_latents({'rgb':rgb})
            tokens=pack_cls_patch(self.normalizer.normalize_cls(cls),self.normalizer.normalize_patch(patch)).detach().half().cpu()
            for token,(batchkey,(frame,cachekey,_)) in zip(tokens,part):
                features[batchkey]=token;frame._cache[cachekey]=token
                while len(frame._cache)>self.cached_views_per_frame:frame._cache.popitem(last=False)
        self.last_diagnostics={'queries':len(targets),'predicted':len(plans),'encoded_views':len(pending),
            'reverse_queries':sum(p.source_index==0 for _,p,_ in plans),'skipped':skipped}
        if not plans:return NwmPrediction(None,meta={'empty':True,'records':[],'skipped':skipped})
        if initial_noise is not None and tuple(initial_noise.shape)!=(len(plans),257,768):
            raise ValueError('initial_noise must match valid query rows in input order')
        predictions=[];records=[];provenance=[]
        for start in range(0,len(plans),self.prediction_batch_size):
            part=plans[start:start+self.prediction_batch_size];part_records=[]
            for target,plan,_ in part:
                xy=local_xy(plan.source_position,plan.target_position,plan.source_yaw)
                part_records.append(RaeGhostInputRecord(target.env_index,target.query_id,
                    float(xy[0]),float(xy[1]),float(np.linalg.norm(xy)),plan.rel_t*128,
                    NwmCondition(*plan.delta,plan.rel_t)))
                provenance.append({'env_index':target.env_index,'query_id':target.query_id,
                    'source_position':plan.source_position,'source_yaw':plan.source_yaw,
                    'source_index':plan.source_index,'target_position':plan.target_position,
                    'target_yaw':plan.target_yaw,'fallback':plan.fallback})
            batch=RaeNwmInputBatch(context=torch.empty(0),
                context_latent=torch.stack([torch.stack([features[k] for k in keys]) for _,_,keys in part]).float().to(self.device),
                curr_delta=torch.tensor([p.delta for _,p,_ in part],device=self.device)[:,None],
                rel_t=torch.tensor([p.rel_t for _,p,_ in part],device=self.device),
                condition_tensor=torch.tensor([(*p.delta,p.rel_t) for _,p,_ in part],device=self.device),
                records=part_records,skipped={})
            result=self.predictor.predict_time_from_etp_batch(batch,generator=generator,
                initial_noise=None if initial_noise is None else initial_noise[start:start+len(part)])
            predictions.append(result.pred_latent);records.extend(part_records)
        return build_native_cls_prediction(torch.cat(predictions),normalizer=self.normalizer,
            meta={'records':records,'sources':provenance,'skipped':skipped,'context':self.context_metadata})
