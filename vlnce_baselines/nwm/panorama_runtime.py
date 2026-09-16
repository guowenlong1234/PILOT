"""Reusable frozen NWM interface for observed full-panorama contexts.

The caller supplies observed cube faces or a provider restricted to recorded
history IDs. Target RGB is never an input to the prediction interface.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Sequence
import time

import numpy as np
import torch

from .etp_adapter import RaeGhostInputRecord, RaeNwmInputBatch
from .panorama_context import (
    FORMAT, MODES, local_xy, make_context_plan, observed_view,
    _perspective_sampling_map, wrap,
)
from .raenwm_core.models import pack_cls_patch, make_latent_noise
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
    render_token: object = None
    _cache: OrderedDict = field(default_factory=OrderedDict,repr=False)

    def __post_init__(self):
        self.position=np.asarray(self.position,dtype=np.float32).copy()
        if self.position.shape!=(3,) or not np.isfinite(self.position).all() or not np.isfinite(self.body_yaw):
            raise ValueError('invalid observed pose')
        if self.cube_rgb is None:
            if self.render_token is None:raise ValueError('pose-only history requires an observed render token')
        else:
            self.cube_rgb=np.asarray(self.cube_rgb).copy()
            if self.cube_rgb.dtype!=np.uint8 or self.cube_rgb.ndim!=4 or self.cube_rgb.shape[0]!=6 or self.cube_rgb.shape[-1]!=3 or self.cube_rgb.shape[1]!=self.cube_rgb.shape[2]:
                raise ValueError('observed cube must be [6,H,H,3] uint8')
            self.cube_rgb.flags.writeable=False
        self.position.flags.writeable=False
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
                 cached_views_per_frame=12,require_native_views=True,
                 observation_source='cube',visual_precision='float32',render_observed=None):
        if mode not in MODES:raise ValueError('unknown panorama context mode')
        if min(encode_batch_size,prediction_batch_size,cached_views_per_frame)<1:raise ValueError('positive batch/cache sizes required')
        self.encoder=encoder;self.normalizer=normalizer;self.predictor=predictor
        self.mode=mode;self.device=torch.device(device)
        self.encode_batch_size=encode_batch_size;self.prediction_batch_size=prediction_batch_size
        self.cached_views_per_frame=cached_views_per_frame
        self.require_native_views=bool(require_native_views)
        if observation_source not in ('cube','direct') or visual_precision not in ('float32','fp16','bf16'):
            raise ValueError('invalid observed image source or visual precision')
        if observation_source == 'direct' and mode == 'front':
            raise ValueError('direct observed rendering requires target-aligned context')
        forced_dtype=getattr(encoder,'compute_dtype',None)
        requested_dtype={'float32':torch.float32,'fp16':torch.float16,'bf16':torch.bfloat16}[visual_precision]
        if forced_dtype is not None and forced_dtype != requested_dtype:
            raise ValueError('context precision conflicts with the RGB encoder; use ambient RGB precision')
        self.observation_source=observation_source;self.visual_precision=visual_precision
        self.render_observed=render_observed
        self.cache_identity=object()
        self.last_diagnostics={}
        self._projection_cache = OrderedDict()

    def _observed_rgb_batch(self, views):
        """Apply the same double-precision pixel interpolation in one GPU batch."""
        if (self.mode == 'front' or self.device.type != 'cuda'
                or len({frame.cube_rgb.shape for frame, _ in views}) > 1):
            return np.stack([frame.front_rgb if self.mode == 'front' else
                observed_view(frame.cube_rgb, yaw, frame.native_world_rgb12)
                for frame, yaw in views])
        result = torch.empty((len(views),224,224,3),dtype=torch.uint8,device=self.device)
        projected=[];maps=[];indices=[];native=[];native_indices=[]
        for index,(frame,yaw) in enumerate(views):
            sector=int(np.floor(float(yaw)/(np.pi/6)+.5))%12
            if frame.native_world_rgb12 is not None and abs(float(wrap(yaw-sector*np.pi/6)))<1e-6:
                native.append(frame.native_world_rgb12[sector]);native_indices.append(index)
                continue
            key=(frame.cube_rgb.shape[1],float(yaw),224,90.)
            if key not in self._projection_cache:
                self._projection_cache[key]=tuple(torch.tensor(a,device=self.device)
                    for a in _perspective_sampling_map(*key))
                while len(self._projection_cache)>16:self._projection_cache.popitem(last=False)
            self._projection_cache.move_to_end(key)
            maps.append(self._projection_cache[key]);projected.append(frame.cube_rgb);indices.append(index)
        if native:
            result[native_indices]=torch.as_tensor(np.stack(native),device=self.device)
        if projected:
            cubes=torch.as_tensor(np.stack(projected),device=self.device)
            face,y0,x0,y1,x1,wx,wy=[torch.stack([m[j] for m in maps]) for j in range(7)]
            batch=torch.arange(len(projected),device=self.device)[:,None,None]
            # Match NumPy operation order and float64 weights exactly; there
            # is no reduced-precision image sampling or grid_sample rounding.
            pixels=((1-wx)*(1-wy)*cubes[batch,face,y0,x0]
                    +wx*(1-wy)*cubes[batch,face,y0,x1]
                    +(1-wx)*wy*cubes[batch,face,y1,x0]
                    +wx*wy*cubes[batch,face,y1,x1])
            result[indices]=pixels.round().clamp(0,255).to(torch.uint8)
        return result

    @property
    def context_metadata(self):
        return {'format':FORMAT,'mode':self.mode,'context_size':4,
                'cube_order':'yaw0_90_180_270_pitch+90_-90','source':'selected_last_camera',
                'target':'absolute_pose_unchanged','cache_dtype':'float16',
                'require_native_views':self.require_native_views,
                'view_extraction':('direct_observed_pose' if self.observation_source=='direct' else 'native_world_30deg_else_cube_bilinear'),
                'visual_precision':self.visual_precision,'noise_batch_size':8}

    @torch.no_grad()
    def predict(self,targets:Sequence[PanoramaTarget],histories,*,initial_noise=None,generator=None):
        plans=[];pending=OrderedDict();features={};skipped={};seen=set();frame_envs={}
        for target in targets:
            key=(target.env_index,target.query_id)
            if key in seen:raise ValueError('duplicate target identity')
            seen.add(key)
            frames=list(histories[target.env_index].frames)
            if len(frames)!=4:
                skipped['context_not_ready']=skipped.get('context_not_ready',0)+1;continue
            if self.observation_source=='cube' and self.require_native_views and any(f.native_world_rgb12 is None for f in frames):
                raise ValueError('validated context requires the recorded native world direction bank')
            if self.observation_source=='direct' and any(f.render_token is None for f in frames):
                raise ValueError('direct context requires worker-issued observed frame IDs')
            if self.mode=='front' and any(f.front_rgb is None for f in frames):
                raise ValueError('baseline mode requires the four directly recorded front images')
            plan=make_context_plan([f.position for f in frames],[f.body_yaw for f in frames],
                target.position,target.yaw,self.mode,[f.segment_id for f in frames])
            keys=[]
            for i in plan.order:
                frame=frames[i];yaw=plan.view_yaws[i]
                owner=frame_envs.setdefault(id(frame),target.env_index)
                if owner!=target.env_index:raise ValueError('an observed frame cannot belong to multiple environments')
                cachekey=(self.cache_identity,self.mode=='front',round(float(yaw%(2*np.pi)),8))
                batchkey=(id(frame),cachekey);keys.append(batchkey)
                if cachekey in frame._cache:
                    features[batchkey]=frame._cache[cachekey];frame._cache.move_to_end(cachekey)
                else:pending[batchkey]=(frame,cachekey,yaw)
            plans.append((target,plan,keys))
        pending_items=list(pending.items())
        direct_images=None
        direct_seconds=0.
        if pending_items and self.observation_source=='direct':
            if self.render_observed is None:raise RuntimeError('observed view provider is not configured')
            requests=[dict(env_index=frame_envs[id(frame)],frame_id=frame.render_token,yaw=float(yaw))
                      for _,(frame,_,yaw) in pending_items]
            started=time.perf_counter()
            direct_images=np.asarray(self.render_observed(requests))
            direct_seconds=time.perf_counter()-started
            if direct_images.shape!=(len(requests),224,224,3) or direct_images.dtype!=np.uint8:
                raise ValueError('direct provider must return ordered uint8 RGB rows')
        for start in range(0,len(pending_items),self.encode_batch_size):
            part=pending_items[start:start+self.encode_batch_size]
            rgb=(direct_images[start:start+len(part)] if direct_images is not None else
                 self._observed_rgb_batch([(frame,yaw) for _,(frame,_,yaw) in part]))
            # Match the verified FP32 visual representation inside SFT's outer
            # autocast too; this raw-only call never touches the navigation MLP.
            with torch.autocast(device_type=self.device.type,enabled=self.visual_precision!='float32',
                                dtype=torch.bfloat16 if self.visual_precision=='bf16' else torch.float16):
                cls,patch=self.encoder.forward_raw_cls_and_patch_latents({'rgb':rgb})
            # The bounded per-frame cache lives on the inference device. Keep
            # the same float16 quantization without a synchronous GPU->CPU->GPU
            # round trip for every freshly encoded view.
            tokens=pack_cls_patch(self.normalizer.normalize_cls(cls),self.normalizer.normalize_patch(patch)).detach().half()
            for token,(batchkey,(frame,cachekey,_)) in zip(tokens,part):
                # Own one view's storage, rather than pinning an entire encode
                # batch after the other views have been evicted.
                token = token.clone()
                features[batchkey]=token;frame._cache[cachekey]=token
                while len(frame._cache)>self.cached_views_per_frame:frame._cache.popitem(last=False)
        self.last_diagnostics={'queries':len(targets),'predicted':len(plans),'encoded_views':len(pending),
            'reverse_queries':sum(p.source_index==0 for _,p,_ in plans),'skipped':skipped,
            'direct_render_seconds':direct_seconds,'direct_rendered_views':0 if direct_images is None else len(direct_images)}
        if not plans:return NwmPrediction(None,meta={'empty':True,'records':[],'skipped':skipped})
        if initial_noise is not None and tuple(initial_noise.shape)!=(len(plans),257,768):
            raise ValueError('initial_noise must match valid query rows in input order')
        if initial_noise is None:
            # Preserve the established eight-row random stream independently
            # of execution batching; larger batches do not reassign noise.
            initial_noise=torch.cat([make_latent_noise(min(8,len(plans)-i),768,16,self.device,
                dtype=torch.float32,predict_cls_token=True,generator=generator)
                for i in range(0,len(plans),8)])
        predictions=[];records=[];provenance=[];snapshots=[]
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
            if getattr(self, 'capture_stage2_snapshots', False):
                for j, (target, plan, keys) in enumerate(part):
                    snapshots.append(dict(
                        env_index=target.env_index, query_id=target.query_id,
                        context_latents=batch.context_latent[j].detach().cpu().half().contiguous(),
                        source_position=np.asarray(plan.source_position).copy(),
                        source_yaw=float(plan.source_yaw), target_yaw=float(plan.target_yaw),
                        target_position=np.asarray(plan.target_position).copy(),
                        horizon=float(plan.rel_t)*128,
                        order=list(plan.order), view_yaws=list(plan.view_yaws),
                        context_metadata=self.context_metadata,
                    ))
            result=self.predictor.predict_time_from_etp_batch(batch,generator=generator,
                initial_noise=None if initial_noise is None else initial_noise[start:start+len(part)])
            predictions.append(result.pred_latent);records.extend(part_records)
        return build_native_cls_prediction(torch.cat(predictions),normalizer=self.normalizer,
            meta={'records':records,'sources':provenance,'skipped':skipped,'context':self.context_metadata,'stage2_snapshots':snapshots})
