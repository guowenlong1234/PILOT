"""Reject incomplete or misleading evaluation success before advancing a queue."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'rxr_queue', Path(__file__).parents[1] / 'scripts/run_rxr_checkpoint_queue.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


@pytest.fixture
def result(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset = Path('data/datasets/RxR_VLNCE_v0_enc_xlmr/val_unseen/val_unseen_guide.json.gz')
    dataset.parent.mkdir(parents=True)
    with gzip.open(dataset, 'wt') as stream:
        json.dump({'episodes': [{'episode_id': i} for i in range(11006)]}, stream)
    output = Path('output')
    directory = output / 'results/test/eval_results'
    directory.mkdir(parents=True)
    episodes = directory / 'stats_ep_ckpt_30000_val_unseen_r0_w1.json'
    summary = directory / 'stats_ckpt_30000_val_unseen.json'
    episodes.write_text(json.dumps({str(i): {'success': 1.0} for i in range(11006)}))
    summary.write_text(json.dumps({'success': 1.0}))
    return output, episodes, summary


def test_complete_result(result):
    output, _, _ = result
    queue.validate(output, -1)
    assert json.loads((output / 'validation.json').read_text())['episodes'] == 11006


def test_missing_route(result):
    output, episodes, _ = result
    rows = json.loads(episodes.read_text())
    del rows['42']
    episodes.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='Incomplete coverage'):
        queue.validate(output, -1)


def test_duplicate_route(result):
    output, episodes, _ = result
    episodes.write_text('{"1":{"success":1},"1":{"success":1}}')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        queue.validate(output, -1)


@pytest.mark.parametrize('metric', [0.5, float('nan')])
def test_invalid_summary(result, metric):
    output, _, summary = result
    summary.write_text(json.dumps({'success': metric}))
    with pytest.raises(ValueError):
        queue.validate(output, -1)
