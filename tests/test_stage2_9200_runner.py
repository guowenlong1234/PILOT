"""CPU-only contract tests for the evaluation-host 9200 orchestrator."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_runner():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            'run_stage2_9200_ablation_test', SCRIPTS/'run_stage2_9200_ablation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_resume_capacity_counts_only_unfinished_collections():
    runner = load_runner()
    estimates = {'train': 100, 'dev': 20}
    assert runner.required_capacity(estimates, ['train','dev']) == 150 + 40*runner.GIB
    assert runner.required_capacity(estimates, ['dev']) == 25 + 40*runner.GIB


def test_full_and_none_training_commands_have_the_same_budget(tmp_path):
    runner = load_runner()
    full = runner.training_command('/data/train',tmp_path/'full','full')
    none = runner.training_command('/data/train',tmp_path/'none','none')
    def normalize(command):
        command = list(command)
        command[command.index('--output')+1] = '<output>'
        command[command.index('--future-mode')+1] = '<mode>'
        return command
    assert normalize(full) == normalize(none)
    for command in (full,none):
        assert command[command.index('--max-steps')+1] == '6000'
        assert command[command.index('--save-every')+1] == '250'
        assert command[command.index('--seed')+1] == '2'


def test_top_level_failure_replaces_running_status(tmp_path):
    runner = load_runner()
    runner.save(tmp_path/'pipeline.json', {'status':'running'})
    error = RuntimeError('capacity gate failed')
    try:
        raise error
    except RuntimeError as caught:
        runner.record_pipeline_failure(['--output',str(tmp_path)],caught)
    status = json.loads((tmp_path/'pipeline.json').read_text())
    assert status['status'] == 'failed'
    assert status['error_type'] == 'RuntimeError'
    assert status['error'] == 'capacity gate failed'


def test_evaluation_index_is_separate_hashed_record(tmp_path):
    runner = load_runner()
    report = tmp_path/'metrics_000250.json'
    checkpoint = tmp_path/'head_step_000250.pt'
    index = tmp_path/'step_000250.json'
    report.write_text(json.dumps({'status':'complete','results':[{'global_step':250}]}))
    checkpoint.write_bytes(b'head')
    payload = runner.publish_evaluation_index(index,report,checkpoint,250)
    assert json.loads(index.read_text()) == payload
    assert payload['report_file'] == report.name
    assert payload['report_sha256'] == runner.sha(report)
    assert payload['head_sha256'] == runner.sha(checkpoint)

    report.write_text(json.dumps({'status':'failed','results':[]}))
    with pytest.raises(ValueError,match='incomplete evaluator report'):
        runner.publish_evaluation_index(index,report,checkpoint,250)


def test_host_orchestrators_do_not_import_torch():
    for name in ('run_stage2_9200_ablation.py','stage2_e24_job.py',
                 'run_stage2_offline_preflight.py','run_stage2_collection.py',
                 'stage2_offline_worker.py'):
        source = (SCRIPTS/name).read_text()
        assert 'import torch' not in source
