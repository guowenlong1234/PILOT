"""Versioned rollout semantics shared by collection, training and deployment."""
import torch


def rollout_contract(provenance=None):
    source = provenance or {}
    depth = source.get('lookahead_horizon_steps', 1)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth not in (1, 2, 3):
        raise ValueError('lookahead_horizon_steps must be 1, 2 or 3')
    aggregation = source.get('rollout_aggregation', 'endpoint')
    failure = source.get('rollout_failure_policy', 'deepest_valid')
    if aggregation != 'endpoint' or failure != 'deepest_valid':
        raise ValueError('unsupported rollout aggregation/failure policy')
    return dict(lookahead_horizon_steps=depth, rollout_aggregation=aggregation,
                rollout_failure_policy=failure)


def validate_rollout_row(row, expected_depth=None):
    depth = row.get('rollout_requested_depth', 1)
    rollout_contract(dict(lookahead_horizon_steps=depth))
    if expected_depth is not None and depth != expected_depth:
        raise ValueError('row rollout depth differs from provenance')
    if 'rollout_requested_depth' not in row:
        return  # Legacy one-step shards remain readable without rewriting them.
    k = len(row['future_valid_mask'])
    for key, dtype in (('realized_depths', torch.long), ('full_horizon_mask', torch.bool),
                       ('fallback_mask', torch.bool)):
        value = row.get(key)
        if not torch.is_tensor(value) or value.shape != (k,) or value.dtype != dtype:
            raise ValueError('invalid rollout row field: ' + key)
    realized = row['realized_depths']
    valid = row['future_valid_mask']
    if ((realized < 0) | (realized > depth)).any() or not torch.equal(realized > 0, valid):
        raise ValueError('realized rollout depths disagree with future validity')
    if not torch.equal(row['full_horizon_mask'], realized == depth):
        raise ValueError('full horizon mask disagrees with realized depths')
    if not torch.equal(row['fallback_mask'], (realized > 0) & (realized < depth)):
        raise ValueError('fallback mask disagrees with realized depths')
    for key in ('rollout_metadata', 'rollout_failure_reason', 'endpoint_metadata'):
        if len(row.get(key, [])) != k:
            raise ValueError('invalid rollout metadata: ' + key)
    conditions = row.get('endpoint_conditions')
    if (not torch.is_tensor(conditions) or conditions.shape != (k, 4)
            or not torch.isfinite(conditions).all()):
        raise ValueError('invalid endpoint conditions')


def deployment_rollout_contract(dataset, requested_depth=1, allow_transfer=False):
    contracts = [rollout_contract(part.get('provenance', {})) for part in dataset]
    if not contracts or any(c != contracts[0] for c in contracts):
        raise ValueError('head dataset mixes incompatible rollout contracts')
    requested = rollout_contract(dict(lookahead_horizon_steps=requested_depth))
    trained_depth = contracts[0]['lookahead_horizon_steps']
    transfer = trained_depth != requested_depth
    if transfer and not allow_transfer:
        raise ValueError('head training rollout depth differs from inference; explicit allow_rollout_depth_transfer required')
    return dict(**requested, head_training_lookahead_horizon_steps=trained_depth,
                rollout_depth_transfer=transfer, allow_rollout_depth_transfer=bool(allow_transfer),
                head_training_depth_matches_inference=not transfer)
