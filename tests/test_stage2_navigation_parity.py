import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_stage2_navigation_comparison import validate_parity


def run(path,logits,action=1,success=1):
    (path/'results').mkdir(parents=True)
    p=dict(stage1_sha256='fixed',assets={},commit='same',split='val_unseen',seed=100,environments=8,compile=True,episode_ids=['1'])
    (path/'provenance.json').write_text(json.dumps(p))
    (path/'launch.json').write_text(json.dumps(dict(command=['run.py','SEED','100','RESULTS_DIR',str(path/'results')])))
    (path/'results/stats_ep_test.json').write_text(json.dumps({'1':{'success':success}}))
    (path/'run.log').write_text('prefix STAGE2_BASE_TRACE '+json.dumps(dict(episode='1',step=0,action=action,logits=logits))+'\n')
    return path


def test_baseline_measured_tolerance_keeps_actions_strict(tmp_path):
    base=run(tmp_path/'base',[0.,1.]);repeat=run(tmp_path/'repeat',[0.,1.125]);zero=run(tmp_path/'zero',[0.,1.0625])
    r=validate_parity(base,zero,repeat)
    assert r['logits_tolerance']==.125 and r['logits_max_abs_error']==.0625
    assert r['exact_actions_metrics'] and not r['exact_logits_actions_metrics']
    with pytest.raises(ValueError,match='exceeds'):validate_parity(base,zero)
    bad=run(tmp_path/'bad',[0.,1.25])
    with pytest.raises(ValueError,match='exceeds'):validate_parity(base,bad,repeat)
    action=run(tmp_path/'action',[0.,1.0625],action=0)
    with pytest.raises(ValueError,match='action changed'):validate_parity(base,action,repeat)
    metrics=run(tmp_path/'metrics',[0.,1.0625],success=0)
    with pytest.raises(ValueError,match='metrics differ'):validate_parity(base,metrics,repeat)


def test_repeated_baseline_must_match_configuration(tmp_path):
    base=run(tmp_path/'base',[0.,1.]);repeat=run(tmp_path/'repeat',[0.,1.125]);zero=run(tmp_path/'zero',[0.,1.0625])
    p=json.loads((repeat/'provenance.json').read_text());p['seed']=9
    (repeat/'provenance.json').write_text(json.dumps(p))
    with pytest.raises(ValueError,match='configuration differs'):validate_parity(base,zero,repeat)


def test_calibration_envelope_uses_only_baseline_pairs(tmp_path):
    base=run(tmp_path/'base',[0.,1.]);low=run(tmp_path/'low',[0.,.9375]);high=run(tmp_path/'high',[0.,1.0625])
    zero=run(tmp_path/'zero',[0.,1.125])
    r=validate_parity(base,zero,[low,high])
    assert r['logits_tolerance']==.125 and r['logits_max_abs_error']==.125
    assert r['baseline_repeats']==[str(low),str(high)]
    outlier=run(tmp_path/'outlier',[0.,1.25])
    with pytest.raises(ValueError,match='exceeds'):validate_parity(base,outlier,[low,high])
