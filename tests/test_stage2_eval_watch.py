"""Host orchestration contracts; no SSH, GPU or data copy during tests."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('stage2_eval_watch_test_module', SCRIPTS/'watch_stage2_offline_eval.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def report(step=250, net=4, harms=2, ce=.7):
    return dict(status='complete', dataset=[dict(manifest_sha256=watch.DEV_MANIFEST_SHA)],
                results=[dict(global_step=step, gain=1., primary_rows=10,
                              net_fixes=net, harms=harms, classification_loss=ce)])


def watcher(tmp_path):
    return watch.Watcher(SimpleNamespace(train_dir=str(tmp_path),
        eval_run_dir='data/logs/test', max_hours=1, save_every=250, poll_seconds=.01))


def test_report_rejects_incomplete_wrong_gain_step_manifest_and_nan():
    assert watch.validate_report(report(), 250)['net_corrections'] == 4
    for mutate in (lambda r:r.update(status='running'),
                   lambda r:r['results'][0].update(gain=.5),
                   lambda r:r['results'][0].update(global_step=500),
                   lambda r:r['dataset'][0].update(manifest_sha256='wrong'),
                   lambda r:r['results'][0].update(classification_loss=float('nan'))):
        data = report()
        mutate(data)
        with pytest.raises(ValueError):
            watch.validate_report(data, 250)


def test_ambiguous_prior_worker_is_never_relaunched(tmp_path):
    w = watcher(tmp_path)
    head = tmp_path/'head_step_000250.pt'
    head.write_bytes(b'published')
    watch.save(w.out/'step_000250.json', dict(status='evaluating', head_sha256=watch.sha(head), job_dir='job'))
    w.remote = lambda *a, **kw: pytest.fail('Must not launch a remote command')
    with pytest.raises(RuntimeError, match='automatic relaunch is disabled'):
        w.evaluate(head, 250)


def test_completed_training_rejects_missing_periodic_heads(tmp_path):
    w = watcher(tmp_path)
    watch.save(tmp_path/'status.json', dict(status='completed', step=500))
    (tmp_path/'head_step_000500.pt').write_bytes(b'head')
    w.identity = lambda: None
    w.evaluate = lambda *args: None
    with pytest.raises(RuntimeError, match='Missing periodic'):
        w.watch()


def test_curve_retains_all_points_and_selection_ties(tmp_path):
    w = watcher(tmp_path)
    for step,net,harms,ce in ((250,3,1,.8),(500,3,1,.7),(750,3,1,.7),(1000,2,0,.1)):
        path = w.out/f'metrics_step_{step:06d}.json'
        watch.save(path,report(step,net,harms,ce))
        watch.save(w.out/f'step_{step:06d}.json', dict(status='completed',step=step,
                   head='head',head_sha256='sha',report_file=path.name))
    w.curve()
    selection=json.loads((w.out/'selection.json').read_text())
    assert [p['step'] for p in selection['selected']] == [500,750]
    assert selection['status'] == 'provisional'
    assert len(json.loads((w.out/'curve.json').read_text())['points']) == 4
