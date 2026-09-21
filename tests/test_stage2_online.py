import ast
import inspect
from types import SimpleNamespace
import pytest
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
    from vlnce_baselines.nwm.active_lookahead.stage2_online import BASE_SHA, TRANSFER_9200_SHA, validate_deployment_base
    assert validate_deployment_base(BASE_SHA,'same')['head_retrained'] is False
    with pytest.raises(ValueError):validate_deployment_base(TRANSFER_9200_SHA,'same')
    with pytest.raises(ValueError):validate_deployment_base(BASE_SHA,'6400_to_9200')
    with pytest.raises(ValueError):validate_deployment_base('other','6400_to_9200')
    metadata=validate_deployment_base(TRANSFER_9200_SHA,'6400_to_9200')
    assert metadata['head_training_base_sha256']==BASE_SHA
    assert metadata['deployment_base_sha256']==TRANSFER_9200_SHA
    native=validate_deployment_base(TRANSFER_9200_SHA,'native_9200')
    assert native['head_training_base_sha256']==TRANSFER_9200_SHA
    assert native['head_retrained'] is True


def _native_checkpoint(asset, future_mode='full'):
    from vlnce_baselines.nwm.active_lookahead.stage2_data import sha256
    from vlnce_baselines.nwm.active_lookahead.stage2_online import TRANSFER_9200_SHA
    from vlnce_baselines.nwm.active_lookahead.stage2_training import MODEL_CONFIG
    model=dict(MODEL_CONFIG,future_mode=future_mode)
    provenance=dict(stage1_sha256=TRANSFER_9200_SHA,split='train',
        feature_space='raw_cls+normalized_patch_fp16',
        context_contract='stage2_panorama_q0_snapshot_v1',assets={str(asset):sha256(asset)})
    return dict(model_config=model,contract=dict(model_config=model,
        train_config=dict(future_mode=future_mode),dataset=[dict(provenance=provenance)]))


def test_native_9200_accepts_explicit_head_sha_and_reports_ablation(tmp_path):
    from vlnce_baselines.nwm.active_lookahead.stage2_data import sha256
    from vlnce_baselines.nwm.active_lookahead.stage2_online import validate_deployment_base, validate_head_checkpoint, TRANSFER_9200_SHA
    head=tmp_path/'head.pt'; head.write_bytes(b'native head')
    asset=tmp_path/'asset.pt'; asset.write_bytes(b'prediction asset')
    cfg=SimpleNamespace(head=str(head),head_sha256=sha256(head))
    metadata=validate_head_checkpoint(_native_checkpoint(asset,'none'),cfg,
        validate_deployment_base(TRANSFER_9200_SHA,'native_9200'))
    assert metadata==dict(head_sha256=sha256(head),future_mode='none')


@pytest.mark.parametrize('mutation,error',[
    (lambda c: c['model_config'].pop('future_mode'),'explicit future_mode'),
    (lambda c: c['contract']['dataset'][0]['provenance'].update(split='val_unseen'),'train split'),
    (lambda c: c['contract']['dataset'][0]['provenance'].update(stage1_sha256='wrong'),'feature/base'),
    (lambda c: c['model_config'].update(num_layers=2),'baseline model structure'),
])
def test_native_9200_rejects_non_native_head_contract(tmp_path,mutation,error):
    from vlnce_baselines.nwm.active_lookahead.stage2_data import sha256
    from vlnce_baselines.nwm.active_lookahead.stage2_online import validate_deployment_base, validate_head_checkpoint, TRANSFER_9200_SHA
    head=tmp_path/'head.pt'; head.write_bytes(b'native head')
    asset=tmp_path/'asset.pt'; asset.write_bytes(b'prediction asset')
    checkpoint=_native_checkpoint(asset); mutation(checkpoint)
    with pytest.raises(ValueError,match=error):
        validate_head_checkpoint(checkpoint,SimpleNamespace(head=str(head),head_sha256=sha256(head)),
            validate_deployment_base(TRANSFER_9200_SHA,'native_9200'))


def test_native_and_legacy_gain_contracts_are_distinct():
    from vlnce_baselines.nwm.active_lookahead.stage2_online import validate_online_gain
    for gain in (0.,1.):validate_online_gain(gain,'native_9200')
    with pytest.raises(ValueError):validate_online_gain(1.5,'native_9200')
    for gain in (0.,1.5):validate_online_gain(gain,'same')
    with pytest.raises(ValueError):validate_online_gain(1.,'same')
