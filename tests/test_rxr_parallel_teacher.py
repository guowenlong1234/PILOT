"""Compare real serial/parallel teacher methods without importing Habitat."""
import ast
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch


def methods():
    source = Path('vlnce_baselines/ss_trainer_ETP_R1.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RLTrainer')
    scope = {'torch': torch, 'random': random}
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            '_teacher_action_new', '_parallel_rxr_teacher_action',
        }:
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<trainer>', 'exec'), scope)
    return type('Trainer', (), {k:v for k,v in scope.items() if k.startswith('_teacher') or k.startswith('_parallel')})


class Envs:
    num_envs = 4
    def __init__(self):
        self.teacher_calls = []
        self.requests = 0
    def current_episodes(self):
        return [SimpleNamespace(episode_id=i) for i in range(4)]
    def call_at(self, i, name, args):
        if name == 'current_dist_to_goal':
            return [0.5, 5., 5., 5.][i]
        positions = args['ghost_vp_pos']
        if not positions:
            return None
        self.teacher_calls.append((i, positions, args['ref_path']))
        return min(positions, key=lambda p:p[1])[0]
    def call(self, names, args):
        self.requests += 1
        return [self.call_at(i, n, a) for i,(n,a) in enumerate(zip(names,args))]


@pytest.mark.parametrize('cached', [True, False])
@pytest.mark.parametrize('seed', [0, 17, 2026])
def test_parallel_preserves_actions_rng_and_skipped_teacher_state(cached, seed):
    trainer = methods()()
    trainer.device = 'cpu'
    trainer.config = SimpleNamespace(MODEL=SimpleNamespace(task_type='rxr'),
        IL=SimpleNamespace(expert_policy='ndtw', parallel_rxr_teacher=False))
    trainer.gmaps = [SimpleNamespace(ghost_real_pos={'a':[1., 8.], 'b':[4., 9.]}) for _ in range(4)]
    trainer.gt_data = {str(i):{'locations':[i]} for i in range(4)}
    ids = [['stop','a','b'] for _ in range(4)]
    distances = [0.5,5.,5.,5.] if cached else None
    results = []
    for parallel in [False, True]:
        trainer.envs = Envs()
        trainer.config.IL.parallel_rxr_teacher = parallel
        random.seed(seed)
        action = trainer._teacher_action_new(ids, [False,True,False,False], True, distances)
        results.append((action.tolist(), random.getstate(), trainer.envs.teacher_calls))
    assert results[0] == results[1]
    assert results[1][0][:2] == [0,-100]
    assert [i for i,_,_ in results[1][2]] == [2,3]
    assert trainer.envs.requests == (1 if cached else 2)
