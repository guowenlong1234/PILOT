"""Explicit collector resize never silently bypasses a normal exact restore."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def fixture(allow=False, current=12):
    tree=ast.parse(Path('vlnce_baselines/ss_trainer_ETP_R1.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RLTrainer')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_restore_episode_iterator_state')
    scope={'logger':Mock()}
    exec(compile(ast.Module(body=[method],type_ignores=[]),'<trainer>','exec'),scope)
    obj=SimpleNamespace(world_size=2,local_rank=0,
        config=SimpleNamespace(IL=SimpleNamespace(allow_env_count_change_on_resume=allow)),
        envs=SimpleNamespace(num_envs=current,call=Mock()))
    state={'format_version':1,'world_size':2,'ranks':[
        {'rank':i,'num_envs':6,'environments':[{'episode':j} for j in range(6)]} for i in range(2)]}
    return scope['_restore_episode_iterator_state'],obj,state


def test_resize_is_rejected_by_default():
    restore,obj,state=fixture()
    with pytest.raises(ValueError,match='changing the number of environments'):restore(obj,state)
    obj.envs.call.assert_not_called()


def test_explicit_resize_leaves_source_state_intact():
    restore,obj,state=fixture(allow=True)
    before=copy.deepcopy(state)
    restore(obj,state)
    obj.envs.call.assert_not_called()
    assert state==before


def test_same_count_still_restores_exact_queues_when_opted_in():
    restore,obj,state=fixture(allow=True,current=6)
    restore(obj,state)
    obj.envs.call.assert_called_once_with(['set_episode_iterator_state']*6,
        [{'state':x} for x in state['ranks'][0]['environments']])


def test_resize_does_not_allow_changed_world_size():
    restore,obj,state=fixture(allow=True)
    state['world_size']=1
    with pytest.raises(ValueError,match='training ranks'):restore(obj,state)


def test_resize_does_not_hide_malformed_saved_queues():
    restore,obj,state=fixture(allow=True)
    state['ranks'][0]['environments']=[]
    with pytest.raises(ValueError,match='Malformed'):restore(obj,state)
