from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlnce_baselines.common.environments import VLNCEDaggerEnv
from vlnce_baselines.nwm.panorama_runtime import (
    ObservedPanoramaFrame,PanoramaHistory,PanoramaTarget,PanoramaPredictionRuntime,
)


def test_worker_only_renders_registered_history_and_reset_invalidates_tokens():
    env=object.__new__(VLNCEDaggerEnv)
    env.raenwm_observation_source='direct';env.raenwm_panorama_mode='world_exact_select'
    env._raenwm_direct_history={};env._raenwm_direct_serial=0
    env._low_level_context_enabled=lambda:True
    events=[dict(type='frame',position=[0.,0.,-i*.25]) for i in range(5)]
    env.raenwm_context_events=SimpleNamespace(pop_payload=lambda:dict(events=events,stats={}))
    rendered=[]
    def render(requests):
        rendered.extend(requests)
        return [dict(rgb=np.zeros((224,224,3),np.uint8)) for _ in requests]
    env._render_nwm_rgb_views=render
    env.pop_raenwm_context_events()
    assert not rendered and len(env._raenwm_direct_history)==4
    with pytest.raises(ValueError):env.render_raenwm_observed_directions([dict(frame_id='1',yaw=0)])
    env.render_raenwm_observed_directions([dict(frame_id='5',yaw=.47,position=[999,0,999])])
    np.testing.assert_array_equal(rendered[0]['position'],[0,0,-1])
    events[:]=[dict(type='reset')];env.pop_raenwm_context_events()
    with pytest.raises(ValueError):env.render_raenwm_observed_directions([dict(frame_id='5',yaw=.47)])


class Encoder:
    def forward_raw_cls_and_patch_latents(self,obs):
        n=len(obs['rgb']);return torch.ones(n,768),torch.ones(n,768,16,16)


class Normalizer:
    def normalize_cls(self,x):return x
    def normalize_patch(self,x):return x
    def denormalize_cls(self,x):return x


class Predictor:
    def __init__(self):self.noises=[]
    def predict_time_from_etp_batch(self,batch,**kw):
        self.noises.append(kw['initial_noise'].clone())
        return SimpleNamespace(pred_latent=batch.context_latent[:,0])


def setup(batch=8):
    history=PanoramaHistory()
    for i in range(4):
        history.append(ObservedPanoramaFrame(str(i),'s',[0,0,-i*.25],0,None,render_token=str(i)))
    calls=[]
    def render(rows):
        calls.append(rows)
        return np.zeros((len(rows),224,224,3),np.uint8)
    predictor=Predictor()
    runtime=PanoramaPredictionRuntime(encoder=Encoder(),normalizer=Normalizer(),predictor=predictor,
        mode='world_exact_select',device='cpu',observation_source='direct',render_observed=render,
        prediction_batch_size=batch)
    return runtime,history,predictor,calls


def test_direct_requests_are_deduplicated_and_features_cached():
    runtime,history,predictor,calls=setup()
    targets=[PanoramaTarget(0,str(i),(0,0,-2-i*.1),.47) for i in range(2)]
    runtime.predict(targets,{0:history})
    assert len(calls)==1 and len(calls[0])==4
    assert all(set(r)=={'env_index','frame_id','yaw'} and r['yaw']==.47 for r in calls[0])
    runtime.predict(targets,{0:history})
    assert len(calls)==1 and runtime.last_diagnostics['encoded_views']==0


def test_noise_stream_is_independent_of_execution_batch_size():
    tapes=[];states=[]
    for batch in [2,8,16]:
        runtime,history,predictor,calls=setup(batch)
        targets=[PanoramaTarget(0,str(i),(0,0,-2-i*.1),.47) for i in range(11)]
        g=torch.Generator().manual_seed(73)
        runtime.predict(targets,{0:history},generator=g)
        tapes.append(torch.cat(predictor.noises));states.append(g.get_state())
    assert all(torch.equal(tapes[0],v) for v in tapes)
    assert all(torch.equal(states[0],v) for v in states)


def test_fast_preset_records_source_precision_and_batches():
    from vlnce_baselines.config.default import get_config
    from vlnce_baselines.nwm.low_level_context import context_metadata_from_config
    cfg=get_config('run_r2r/iter_train_rae_dino_ghost_concat.yaml,configs/nwm/direct_context_fast.yaml')
    meta=context_metadata_from_config(cfg.MODEL.RAENWM)
    assert not cfg.IL.freeze_navigation_backbone and cfg.NUM_ENVIRONMENTS==4
    assert meta['panorama_observation_source']=='direct'
    assert meta['panorama_visual_precision']=='fp16'
    assert meta['panorama_encode_batch_size']==meta['panorama_prediction_batch_size']==64
    assert meta['panorama_noise_batch_size']==8
