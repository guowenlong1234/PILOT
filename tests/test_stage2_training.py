"""Offline training cursor, RNG and provenance rejection checks."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from vlnce_baselines.nwm.active_lookahead.stage2_data import FORMAT, SPACE, atomic_json, sha256
from vlnce_baselines.nwm.active_lookahead.stage2_training import (
    EpisodeBlockSampler, seed_everything, save_training_checkpoint,
    restore_training_checkpoint,
)


def make_data(root):
    root.mkdir()
    provenance = dict(split='train', episode_ids=['a', 'b', 'c'])
    atomic_json(root/'provenance.json', provenance)
    entries = []
    for episode_id in provenance['episode_ids']:
        rows = [dict(id=f'{episode_id}{i}', future_valid_mask=torch.tensor([True]),
            teacher_valid=True, teacher_stop=False, base_stop=i == 4, no_vp_left=False) for i in range(5)]
        name = episode_id+'.pt'
        torch.save(dict(episode_id=episode_id, rows=rows, text_tokens=torch.ones(1,2)), root/name)
        entries.append(dict(episode_id=episode_id, rows=5, trainable_rows=4, file=name,
            bytes=(root/name).stat().st_size, sha256=sha256(root/name)))
    atomic_json(root/'dataset_manifest.json', dict(status='validated', format=FORMAT, feature_space=SPACE,
        provenance_sha256=sha256(root/'provenance.json'), entries=entries, trainable_rows=12))
    return root


def test_sampler_complete_epochs_and_mid_block_resume(tmp_path):
    root = make_data(tmp_path/'data')
    source = EpisodeBlockSampler([root], block_episodes=2)
    first = source.next_rows(5, 2)
    state = source.state_dict()
    resumed = EpisodeBlockSampler([root], block_episodes=2)
    resumed.load_state_dict(state)
    rest = []
    while True:
        a, b = source.next_rows(5, 2), resumed.next_rows(5, 2)
        assert (None if a is None else [r['id'] for r in a]) == (None if b is None else [r['id'] for r in b])
        if a is None:
            break
        rest.extend(a)
    ids = [r['id'] for r in first+rest]
    expected = {f'{ep}{i}' for ep in 'abc' for i in range(4)}
    assert set(ids[:12]) == set(ids[12:]) == expected
    assert len(ids) == 24


def test_resume_restores_optimizer_sampler_and_all_rng(tmp_path):
    root = make_data(tmp_path/'data')
    seed_everything(2)
    head = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.Dropout(.2), torch.nn.Linear(3,1))
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01)
    sampler = EpisodeBlockSampler([root], block_episodes=2)
    contract = {'data': sampler.provenance, 'config': {'max_steps':20}}

    def update(model, opt, data):
        rows = data.next_rows(3,10)
        x = torch.randn(len(rows),2) * random.random() + np.random.rand()
        loss = model(x).square().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.detach().clone(), [r['id'] for r in rows]

    update(head, optimizer, sampler)
    checkpoint = save_training_checkpoint(tmp_path/'run', head, optimizer, None, sampler, 1, contract)
    expected = update(head, optimizer, sampler)
    parameters = copy.deepcopy(head.state_dict())
    # Arbitrary RNG use and newly initialized objects emulate a new process.
    seed_everything(99)
    replacement = torch.nn.Sequential(torch.nn.Linear(2,3), torch.nn.Dropout(.2), torch.nn.Linear(3,1))
    replacement_opt = torch.optim.AdamW(replacement.parameters(), lr=.01)
    replacement_data = EpisodeBlockSampler([root], block_episodes=2)
    assert restore_training_checkpoint(checkpoint, replacement, replacement_opt, None,
        replacement_data, contract) == 1
    actual = update(replacement, replacement_opt, replacement_data)
    assert torch.equal(actual[0], expected[0]) and actual[1] == expected[1]
    assert all(torch.equal(parameters[k], replacement.state_dict()[k]) for k in parameters)
    with pytest.raises(ValueError, match='contract differs'):
        restore_training_checkpoint(checkpoint, replacement, replacement_opt, None,
            replacement_data, dict(contract, config={'max_steps':21}))
    with pytest.raises(ValueError, match='cannot resume'):
        restore_training_checkpoint(tmp_path/'run'/'head_step_000001.pt', replacement,
            replacement_opt, None, replacement_data, contract)


def test_sampler_rejects_overlap_wrong_split_and_changed_shards(tmp_path):
    root = make_data(tmp_path/'data')
    with pytest.raises(ValueError, match='duplicate'):
        EpisodeBlockSampler([root, root])
    with pytest.raises(ValueError, match='split'):
        EpisodeBlockSampler([root], expected_split='val_unseen')
    with (root/'a.pt').open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='size changed'):
        EpisodeBlockSampler([root])
