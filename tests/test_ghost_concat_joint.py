from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vlnce_baselines.ss_trainer_ETP_R1 import (
    RLTrainer, _ghost_concat_optimizer_groups, _validate_joint_optimizer_state,
)


def _trainer():
    t = object.__new__(RLTrainer)
    t.device = torch.device('cpu')
    t.raenwm_rgb_fusion_adapter = None
    t.config = SimpleNamespace(
        GPU_NUMBERS=2, NUM_ENVIRONMENTS=4,
        IL=SimpleNamespace(freeze_navigation_backbone=False, lr=2e-6, rgb_fusion_lr=1e-5,
                           batch_size=4, gradient_accumulation_steps=1, warmup_iters=0,
                           min_lr_ratio=1.0, iters=2000, sample_ratio=.75,
                           sample_ratio_iteration_offset=14200, decay_interval=3000, is_requeue=False),
        MODEL=SimpleNamespace(
            RGB_ENCODER=SimpleNamespace(type='rae_dinov2', output_size=768),
            ACTIVE_LOOKAHEAD=SimpleNamespace(enabled=False),
            RAENWM=SimpleNamespace(enabled=True, rgb_fusion_enabled=True, rgb_fusion_trainable=True,
                                  rgb_fusion_type='ghost_concat', ghost_concat_hidden_dim=1536,
                                  predict_cls_token=True, rgb_fusion_alpha=1.0,
                                  rgb_fusion_align_navigation_cls=False, condition_source_pose='context_last'),
        ),
    )
    return t


def test_joint_factory_keeps_policy_trainable_and_records_full_resume_contract():
    t = _trainer()
    t.policy = torch.nn.Linear(4, 4)
    t._initialize_raenwm_rgb_fusion_adapter()
    t._configure_navigation_backbone_training()
    assert all(p.requires_grad for p in t.policy.parameters())
    assert t._ghost_concat_joint_enabled()
    checkpoint = {'raenwm_rgb_fusion_adapter_state_dict': {},
                  'rgb_fusion_navigation_contract': t._rgb_fusion_navigation_contract()}
    assert checkpoint['rgb_fusion_navigation_contract']['optimization']['batch_per_rank'] == 4
    t.config.IL.is_requeue = True
    t._validate_rgb_fusion_navigation_contract(checkpoint)
    for field, value in [('lr', 1e-5), ('rgb_fusion_lr', 2e-5), ('batch_size', 8),
                         ('gradient_accumulation_steps', 2), ('iters', 10000)]:
        original = getattr(t.config.IL, field)
        setattr(t.config.IL, field, value)
        with pytest.raises(ValueError, match='training contract'):
            t._validate_rgb_fusion_navigation_contract(checkpoint)
        setattr(t.config.IL, field, original)


def test_joint_groups_exclude_frozen_parameters_and_restore_rates():
    policy = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    policy[0].requires_grad_(False)
    fusion = torch.nn.Linear(4, 4)
    groups = _ghost_concat_optimizer_groups(policy.named_parameters(), fusion.named_parameters(), 2e-6, 1e-5)
    all_params = [p for g in groups for p in g['params']]
    assert len(set(map(id, all_params))) == len(all_params)
    assert set(map(id, all_params)) == set(map(id, [*policy[1].parameters(), *fusion.parameters()]))
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    policy(fusion(torch.ones(1, 4))).sum().backward()
    optimizer.step(); scheduler.step()
    saved = deepcopy(optimizer.state_dict())
    _validate_joint_optimizer_state(optimizer, saved)
    assert [g['lr'] for g in optimizer.param_groups] == [2e-6, 2e-6, 1e-5, 1e-5]
    bad = deepcopy(saved); bad['param_groups'][2]['initial_lr'] = 2e-6
    with pytest.raises(ValueError, match='learning rate'):
        _validate_joint_optimizer_state(optimizer, bad)
    bad = deepcopy(saved); bad['param_groups'].reverse()
    with pytest.raises(ValueError, match='identity'):
        _validate_joint_optimizer_state(optimizer, bad)


@pytest.mark.parametrize('lr', [0, float('nan'), float('inf')])
def test_joint_invalid_lr_is_rejected(lr):
    p = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match='learning rates'):
        _ghost_concat_optimizer_groups(p.named_parameters(), [], lr, 1e-5)
