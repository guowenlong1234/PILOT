from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlnce_baselines.nwm.panorama_runtime import (
    ObservedPanoramaFrame,PanoramaHistory,PanoramaPredictionRuntime,PanoramaTarget)


def frame(i,segment='a'):
    return ObservedPanoramaFrame(str(i),segment,np.array([0.,0.,-i*.25]),0.,np.zeros((6,8,8,3),np.uint8))


def test_segment_reset_and_owned_rgb():
    h=PanoramaHistory();a=frame(0);h.append(a)
    with pytest.raises(ValueError,match='duplicated'):h.append(frame(0))
    h.append(frame(4,'b'));assert len(h.frames)==1
    assert not h.frames[0].cube_rgb.flags.writeable


class Encoder:
    def __init__(self):self.frames=0
    def forward_raw_cls_and_patch_latents(self,obs):
        n=len(obs['rgb']);self.frames+=n
        return torch.ones(n,768),torch.ones(n,768,16,16)


class Normalizer:
    def normalize_cls(self,x):return x
    def normalize_patch(self,x):return x
    def denormalize_cls(self,x):return x


class Predictor:
    def __init__(self):self.batches=[]
    def predict_time_from_etp_batch(self,batch,**kwargs):
        self.batches.append(batch)
        return SimpleNamespace(pred_latent=batch.context_latent[:,0])


def test_prediction_keeps_target_pose_changes_anchor_and_reuses_shared_views():
    h=PanoramaHistory()
    for i in range(4):h.append(frame(i))
    e,p=Encoder(),Predictor();runtime=PanoramaPredictionRuntime(encoder=e,normalizer=Normalizer(),predictor=p,mode='world_exact_select',device='cpu',require_native_views=False)
    targets=[PanoramaTarget(0,'a',(0.,0.,.5),np.pi),PanoramaTarget(0,'b',(0.,0.,1.),np.pi)]
    before=[f.position.copy() for f in h.frames]
    result=runtime.predict(targets,{0:h})
    assert e.frames==4 and len(p.batches)==1
    assert result.meta['sources'][0]['source_index']==0
    assert result.meta['sources'][0]['target_position']==targets[0].position
    assert [r.query_id for r in result.meta['records']]==['a','b']
    assert p.batches[0].curr_delta[0,0,0]>0
    assert result.meta['records'][0].condition.rel_t==pytest.approx(.5/.24975892673356762/128)
    runtime.predict(targets,{0:h});assert e.frames==4
    for a,b in zip(before,h.frames):np.testing.assert_array_equal(a,b.position)


def test_missing_history_does_not_duplicate_spatial_views_as_time():
    h=PanoramaHistory();h.append(frame(0));e,p=Encoder(),Predictor()
    runtime=PanoramaPredictionRuntime(encoder=e,normalizer=Normalizer(),predictor=p,mode='world30',device='cpu')
    result=runtime.predict([PanoramaTarget(0,'a',(0,0,-2),0)],{0:h})
    assert result.meta['empty'] and e.frames==0 and not p.batches


def test_validated_preset_rejects_missing_native_observations():
    h=PanoramaHistory()
    for i in range(4):h.append(frame(i))
    runtime=PanoramaPredictionRuntime(encoder=Encoder(),normalizer=Normalizer(),predictor=Predictor(),mode='world_exact_select',device='cpu')
    with pytest.raises(ValueError,match='native world direction bank'):
        runtime.predict([PanoramaTarget(0,'a',(0,0,-2),0)],{0:h})
