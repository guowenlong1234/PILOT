#!/usr/bin/env python3
"""Audit a real training invocation without changing its model/optimizer command.

Pass --report-dir followed by regular run.py arguments. Optional
--stop-after-save deliberately exits 75 after a complete checkpoint on the
initial run; repeat the same command with IL.is_requeue True to test recovery.
This tool is for isolated short validation jobs only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer


def digest(parameters):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in parameters}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--stop-after-save', type=int)
    args, remaining = parser.parse_known_args()
    rank = int(os.environ.get('LOCAL_RANK', 0))
    root = Path(args.report_dir)
    root.mkdir(parents=True, exist_ok=True)
    report = dict(rank=rank, ok=False, intervals=[], saved=[], planned_stop=False)
    trainers = []
    world_before = {}
    old_runtime = RLTrainer._initialize_raenwm_runtime
    old_interval = RLTrainer._train_interval
    old_save = RLTrainer.save_checkpoint

    def initialize(self, *a, **kw):
        result = old_runtime(self, *a, **kw)
        if not world_before:
            model = self.raenwm_runtime.predictor.bundle.model
            assert all(not p.requires_grad for p in model.parameters())
            world_before.update(digest(model.named_parameters()))
        return result

    def interval(self, *a, **kw):
        if not trainers:
            trainers.append(self)
        assert self._ghost_concat_memory_mode() == 'persistent_node_state'
        assert all(p.requires_grad for group in self.optimizer.param_groups for p in group['params'])
        frozen = digest((n, p) for n, p in self.policy.named_parameters() if not p.requires_grad)
        waypoint = digest(self.waypoint_predictor.named_parameters())
        cls = digest((n, p) for n, p in self.policy.named_parameters() if 'cls_residual_mlp' in n)
        fusion = digest(self.raenwm_rgb_fusion_adapter.named_parameters())
        result = old_interval(self, *a, **kw)
        audit = dict(
            frozen_visual_unchanged=frozen == digest((n, p) for n, p in self.policy.named_parameters() if not p.requires_grad),
            frozen_waypoint_unchanged=waypoint == digest(self.waypoint_predictor.named_parameters()),
            frozen_world_unchanged=world_before == digest(self.raenwm_runtime.predictor.bundle.model.named_parameters()),
            cls_mapping_updated=cls != digest((n, p) for n, p in self.policy.named_parameters() if 'cls_residual_mlp' in n),
            fusion_updated=fusion != digest(self.raenwm_rgb_fusion_adapter.named_parameters()),
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        )
        assert all(audit[k] for k in ('frozen_visual_unchanged', 'frozen_waypoint_unchanged', 'frozen_world_unchanged')), audit
        report['intervals'].append(audit)
        return result

    def save(self, iteration, *a, **kw):
        result = old_save(self, iteration, *a, **kw)
        report['saved'].append(dict(iteration=iteration, optimizer_states=len(self.optimizer.state),
                                   scheduler=self.scheduler.state_dict()))
        if args.stop_after_save == iteration and not self.config.IL.is_requeue:
            report['planned_stop'] = True
            raise SystemExit(75)
        return result

    RLTrainer._initialize_raenwm_runtime = initialize
    RLTrainer._train_interval = interval
    RLTrainer.save_checkpoint = save
    try:
        sys.argv = ['run.py', *remaining]
        runpy.run_path('run.py', run_name='__main__')
        assert report['intervals']
        assert any(row['fusion_updated'] for row in report['intervals'])
        assert any(row['cls_mapping_updated'] for row in report['intervals'])
        report['ok'] = True
    finally:
        (root / f'rank{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        for trainer in trainers:
            if getattr(trainer, 'envs', None) is not None:
                trainer.envs.close()
        RLTrainer._initialize_raenwm_runtime = old_runtime
        RLTrainer._train_interval = old_interval
        RLTrainer.save_checkpoint = old_save


if __name__ == '__main__':
    main()
