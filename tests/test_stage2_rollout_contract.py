"""Prevent silent mixing or deployment of different future-depth distributions."""
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead.stage2_data import (
    FORMAT, SPACE, Stage2Dataset, atomic_json, sha256,
)
from vlnce_baselines.nwm.active_lookahead.stage2_training import EpisodeBlockSampler
from vlnce_baselines.nwm.active_lookahead.stage2_rollout_contract import (
    rollout_contract, deployment_rollout_contract, validate_rollout_row,
)


def test_legacy_contract_is_exactly_one_step():
    assert rollout_contract({}) == rollout_contract({'lookahead_horizon_steps': 1})
    metadata = deployment_rollout_contract([{'provenance': {}}])
    assert metadata['head_training_lookahead_horizon_steps'] == 1
    assert metadata['rollout_depth_transfer'] is False


@pytest.mark.parametrize('depth', [0, 4, True, 1.5, '2'])
def test_invalid_depth_is_rejected(depth):
    with pytest.raises(ValueError, match='1, 2 or 3'):
        rollout_contract({'lookahead_horizon_steps': depth})


@pytest.mark.parametrize('depth', [2, 3])
def test_depth_transfer_requires_explicit_opt_in_and_records_no_retraining(depth):
    dataset = [{'provenance': {}}]
    with pytest.raises(ValueError, match='explicit allow_rollout_depth_transfer'):
        deployment_rollout_contract(dataset, depth)
    metadata = deployment_rollout_contract(dataset, depth, True)
    assert metadata['rollout_depth_transfer'] is True
    assert metadata['head_training_lookahead_horizon_steps'] == 1
    assert metadata['lookahead_horizon_steps'] == depth
    assert metadata['head_training_depth_matches_inference'] is False


def test_head_cannot_mix_rollout_depths_even_when_transfer_is_allowed():
    with pytest.raises(ValueError, match='mixes incompatible'):
        deployment_rollout_contract([{'provenance': {}},
            {'provenance': {'lookahead_horizon_steps': 2}}], 3, True)


def _manifest(root, episode, depth):
    root.mkdir()
    provenance = {'split': 'train'}
    if depth != 1:
        provenance['lookahead_horizon_steps'] = depth
    atomic_json(root/'provenance.json', provenance)
    shard = root/(episode+'.pt')
    shard.write_bytes(b'shard integrity fixture; root contract rejects before loading')
    atomic_json(root/'dataset_manifest.json', dict(status='validated', format=FORMAT,
        feature_space=SPACE, provenance_sha256=sha256(root/'provenance.json'),
        trainable_rows=1, entries=[dict(episode_id=episode, rows=1, trainable_rows=1,
        file=shard.name, bytes=shard.stat().st_size, sha256=sha256(shard))]))
    return root


@pytest.mark.parametrize('reader', [Stage2Dataset, EpisodeBlockSampler])
def test_dataset_readers_refuse_mixed_depth_roots(tmp_path, reader):
    roots = [_manifest(tmp_path/'one', 'a', 1), _manifest(tmp_path/'two', 'b', 2)]
    with pytest.raises(ValueError, match='incompatible'):
        reader(roots)


def test_row_depth_and_fallback_consistency():
    row = dict(rollout_requested_depth=3, future_valid_mask=torch.tensor([True, True, False]),
        realized_depths=torch.tensor([3, 1, 0]), full_horizon_mask=torch.tensor([True, False, False]),
        fallback_mask=torch.tensor([False, True, False]), rollout_metadata=[[], [], []],
        rollout_failure_reason=['', 'cwp_none', 'q0_invalid'], endpoint_metadata=[{}, {}, None],
        endpoint_conditions=torch.zeros(3, 4))
    validate_rollout_row(row, 3)
    with pytest.raises(ValueError, match='differs from provenance'):
        validate_rollout_row(row, 2)
    row['fallback_mask'][1] = False
    with pytest.raises(ValueError, match='fallback mask'):
        validate_rollout_row(row, 3)


def test_legacy_row_cannot_claim_multistep_provenance():
    validate_rollout_row({}, 1)
    with pytest.raises(ValueError, match='differs from provenance'):
        validate_rollout_row({}, 2)
