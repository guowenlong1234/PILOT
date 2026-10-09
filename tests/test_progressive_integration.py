"""Configuration/entrypoint contracts; no real assets or navigation launched."""
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_full_certified_configs_differ_only_in_stopping():
    full=yaml.safe_load((ROOT/'configs/progressive/full.yaml').read_text())
    cert=yaml.safe_load((ROOT/'configs/progressive/certified.yaml').read_text())
    full['MODEL']['PROGRESSIVE']['pruning_mode']='certified'
    assert full==cert
    legacy=yaml.safe_load((ROOT/'configs/progressive/legacy.yaml').read_text())
    assert legacy['MODEL']['PROGRESSIVE']['enabled'] is False


@pytest.mark.parametrize('action',['collect','online'])
def test_real_job_builder_progressive_contract(monkeypatch,tmp_path,action):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location('progressive_job_test',ROOT/'scripts/stage2_e24_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    split='train' if action=='collect' else 'val_unseen'
    dataset=tmp_path/f'data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/{split}/{split}.json.gz'
    dataset.parent.mkdir(parents=True)
    with gzip.open(dataset,'wt') as stream:json.dump({'episodes':[{'episode_id':1,'scene_id':'mock'}]},stream)
    # All asset operations and subprocess Git identity are explicitly mocked.
    monkeypatch.setattr(module,'ROOT',tmp_path)
    contract=tmp_path/'vlnce_baselines/nwm/active_lookahead/progressive_contract.py'
    contract.parent.mkdir(parents=True);contract.write_text((ROOT/'vlnce_baselines/nwm/active_lookahead/progressive_contract.py').read_text())
    monkeypatch.setattr(module,'sha',lambda path:module.BASE_SHAS[9200])
    monkeypatch.setattr(module.subprocess,'check_output',lambda *args,**kwargs:'mock-commit\n')
    monkeypatch.setattr(module,'runtime',lambda machine,args,gpu:['mock-runtime',*args])
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['job',action,'--machine','eval','--output',str(out),
        '--checkpoint','mock-base','--base-step','9200','--split',split,'--head','mock-head',
        '--gain','1','--margin-threshold','-1','--progressive','certified','--dry-run'])
    assert module.main()==0
    launch=json.loads((out/'launch.json').read_text())
    command=launch['command']
    assert command[command.index('MODEL.PROGRESSIVE.enabled')+1]=='True'
    assert launch['provenance']['format']=='etpr1-progressive-episode-v1'
    assert launch['provenance']['max_future_depth']==2
    assert set(launch['provenance']['assets'])=={'world_model','statistics','cwp','world_config'}


def test_progressive_rejects_gain_margin_and_budget():
    from types import SimpleNamespace as NS
    from vlnce_baselines.nwm.active_lookahead.progressive_online import validate_progressive_config
    cfg=NS(max_future_depth=2,total_residual_bound=1.,budget_fractions=[.5,.5],
           share_depth_weights=True,pruning_mode='certified')
    validate_progressive_config(cfg,NS(gain=1.,margin_threshold=-1.))
    for online in [NS(gain=1.5,margin_threshold=-1.),NS(gain=1.,margin_threshold=1.)]:
        with pytest.raises(ValueError): validate_progressive_config(cfg,online)
    cfg.budget_fractions=[.5]
    with pytest.raises(ValueError): validate_progressive_config(cfg)
