import ast
import inspect
import torch
from vlnce_baselines.nwm.active_lookahead.stage2_collect import Stage2Collector
from vlnce_baselines.nwm.active_lookahead.stage2_data import collate_stage2
from vlnce_baselines.nwm.active_lookahead.stage2_online import score_rows


def test_live_collate_requires_no_teacher_fields():
    row=dict(ghost_ids=['a'],topk_base_indices=torch.tensor([0]),owner_embeddings=torch.zeros(1,768),
        future_tokens=torch.zeros(1,257,768),candidate_q0_geometry=torch.zeros(1,3),q1_conditions=torch.zeros(1,4),
        text_tokens=torch.zeros(2,768),base_logits=torch.ones(1),future_valid_mask=torch.tensor([False]))
    batch=collate_stage2([row],include_targets=False)
    assert not any(k.startswith('teacher') for k in batch)
    assert batch['topk_base_indices'][0].tolist()==[0,-1,-1,-1,-1]


def test_prediction_only_path_has_no_teacher_or_writer_calls():
    import textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(Stage2Collector.predict_step)))
    attrs={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
    assert '_teacher_action_new' not in attrs
    assert 'writer' not in attrs


def test_base_transfer_requires_explicit_mode_and_exact_weight():
    import pytest
    from vlnce_baselines.nwm.active_lookahead.stage2_online import BASE_SHA, TRANSFER_9200_SHA, validate_deployment_base
    assert validate_deployment_base(BASE_SHA,'same')['head_retrained'] is False
    with pytest.raises(ValueError):validate_deployment_base(TRANSFER_9200_SHA,'same')
    with pytest.raises(ValueError):validate_deployment_base(BASE_SHA,'6400_to_9200')
    with pytest.raises(ValueError):validate_deployment_base('other','6400_to_9200')
    metadata=validate_deployment_base(TRANSFER_9200_SHA,'6400_to_9200')
    assert metadata['head_training_base_sha256']==BASE_SHA
    assert metadata['deployment_base_sha256']==TRANSFER_9200_SHA
