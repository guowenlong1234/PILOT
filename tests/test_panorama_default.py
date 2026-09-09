from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch

from vlnce_baselines.config.default import get_config
from vlnce_baselines.nwm.low_level_context import context_metadata_from_config,panorama_mode_from_config
from vlnce_baselines.nwm.runtime import NwmPredictionRuntime,NwmQuery
from vlnce_baselines.nwm.etp_adapter import NwmEtpAdapter
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer


def test_current_training_config_defaults_to_winner():
    cfg=get_config('run_r2r/iter_train_rae_dino_ghost_concat.yaml')
    assert panorama_mode_from_config(cfg.MODEL.RAENWM)=='world_exact_select'
    assert context_metadata_from_config(cfg.MODEL.RAENWM)['panorama_format']=='nwm_observed_panorama_virtual_context_v2'


def test_legacy_e24_and_explicit_front_remain_reproducible():
    cfg=get_config('run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml')
    assert panorama_mode_from_config(cfg.MODEL.RAENWM)=='front'
    cfg=get_config('run_r2r/iter_train_rae_dino_ghost_concat.yaml',['MODEL.RAENWM.panorama_context_mode','front'])
    assert 'panorama_context_mode' not in context_metadata_from_config(cfg.MODEL.RAENWM)


def test_weights_only_migration_allowed_but_resume_rejected():
    cfg=get_config('run_r2r/iter_train_rae_dino_ghost_concat.yaml')
    trainer=object.__new__(RLTrainer);trainer.config=cfg
    meta=trainer._raenwm_context_metadata();old={k:v for k,v in meta.items() if not k.startswith('panorama_')}
    assert trainer._validate_raenwm_context_checkpoint_metadata({'raenwm_context_metadata':old},allow_missing=False)==old
    cfg.defrost();cfg.IL.is_requeue=True;cfg.freeze()
    with pytest.raises(ValueError,match='metadata mismatch'):
        trainer._validate_raenwm_context_checkpoint_metadata({'raenwm_context_metadata':old},allow_missing=False)
    assert trainer._validate_raenwm_context_checkpoint_metadata({'raenwm_context_metadata':meta},allow_missing=False)==meta


def test_prediction_routes_to_panorama_with_original_absolute_target():
    runtime=object.__new__(NwmPredictionRuntime);runtime.adapter=NwmEtpAdapter()
    runtime.panorama_mode='world_exact_select';runtime.panorama_histories=['history'];runtime.generator=None
    runtime._source_pose_prediction_calls=0
    captured=[]
    class Predictor:
        last_diagnostics={}
        def predict(self,targets,histories,**kw):
            captured.extend(targets);return 'prediction'
    runtime.panorama_predictor=Predictor()
    q=NwmQuery(0,'g1',np.zeros(3),0.,np.array([1.,0.,0.]))
    assert runtime.predict([q])=='prediction'
    assert captured[0].position==(1.,0.,0.)
    assert captured[0].yaw==pytest.approx(-np.pi/2)


def test_pause_and_reset_keep_environment_histories_aligned():
    runtime=object.__new__(NwmPredictionRuntime);runtime.adapter=NwmEtpAdapter()
    runtime.panorama_mode='world_exact_select';runtime.reset(3)
    remaining=runtime.panorama_histories[2]
    runtime.pause_at(1)
    assert len(runtime.adapter.buffers)==len(runtime.panorama_histories)==2
    assert runtime.panorama_histories[1] is remaining
    runtime.reset(1);assert len(runtime.panorama_histories)==1
