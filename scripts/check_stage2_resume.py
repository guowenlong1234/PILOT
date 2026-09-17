#!/usr/bin/env python3
"""Compare complete E24 training states after continuous and resumed updates."""
import argparse
import json

import numpy as np
import torch

from vlnce_baselines.nwm.active_lookahead.stage2_data import atomic_json, load


def compare(left, right, path='state'):
    if type(left) is not type(right):
        raise AssertionError(f'{path}: types differ')
    if torch.is_tensor(left):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise AssertionError(f'{path}: tensors differ')
    elif isinstance(left, np.ndarray):
        if left.dtype != right.dtype or not np.array_equal(left, right):
            raise AssertionError(f'{path}: arrays differ')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{path}: keys differ')
        for key in left:
            compare(left[key], right[key], f'{path}.{key}')
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f'{path}: lengths differ')
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f'{path}[{index}]')
    elif left != right:
        raise AssertionError(f'{path}: values differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('continuous')
    parser.add_argument('resumed')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    a, b = load(args.continuous), load(args.resumed)
    experiments = [a['contract']['experiment'], b['contract']['experiment']]
    # These are deliberately separate runs. All computational/data contracts
    # still have to agree; restore itself checks the exact experiment identity.
    a['contract'] = {k:v for k,v in a['contract'].items() if k != 'experiment'}
    b['contract'] = {k:v for k,v in b['contract'].items() if k != 'experiment'}
    fields = ('format_version', 'global_step', 'model_config', 'loss_config',
              'contract', 'future_head_state_dict', 'optimizer', 'scheduler',
              'scaler', 'sampler', 'rng')
    for field in fields:
        compare(a[field], b[field], field)
    report = dict(status='passed', bitwise_equal=True, global_step=a['global_step'],
                  checked_fields=list(fields), continuous=args.continuous, resumed=args.resumed,
                  experiments=experiments, excluded_noncomputational_fields=['contract.experiment'],
                  model_tensors=len(a['future_head_state_dict']))
    atomic_json(args.output, report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
