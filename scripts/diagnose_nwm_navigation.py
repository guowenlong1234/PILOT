#!/usr/bin/env python3
"""Capture real navigation NWM queries and exact-pose oracle RGB, without training.

Run through rgb_only_optimization_runtime.sh on the chosen runtime host.
Oracle renders are diagnostic outputs only and never enter the policy.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import runpy
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--max-samples', type=int, default=36)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'capture.json').exists():
        raise RuntimeError('Use a fresh diagnostic output directory')

    from rgb_only_optimization import common
    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer
    from vlnce_baselines.nwm.raenwm_core.models import pack_cls_patch
    from vlnce_baselines.models.graph_utils import heading_from_quaternion
    import habitat
    import habitat_sim
    import transformers

    report = {'samples': [], 'versions': {
        'python': platform.python_version(), 'torch': str(torch.__version__),
        'cuda': torch.version.cuda, 'transformers': transformers.__version__,
        'habitat': getattr(habitat, '__version__', 'unknown'),
        'habitat_sim': getattr(habitat_sim, '__version__', 'unknown'),
        'python_executable': sys.executable}, 'checkpoint': args.checkpoint}
    original = RLTrainer._run_raenwm_rgb_fusion_prediction
    calls = [0]
    per_episode = {}

    def capture(self, *call_args, **kwargs):
        prediction = original(self, *call_args, **kwargs)
        calls[0] += 1
        if prediction is None or prediction.pred_tokens is None:
            return prediction
        if len(report['samples']) >= args.max_samples:
            return prediction
        runtime = self.raenwm_runtime
        batch = runtime.last_batch
        cur_pos, cur_ori, previews = call_args[1:4]
        queries = self._build_raenwm_preview_queries(cur_pos, cur_ori, previews)
        lookup = {(q.env_index, q.query_id): q for q in queries}
        episodes = self.envs.current_episodes()
        # First three ready decision points per episode, up to three candidates
        # each, in runtime order (not selected by prediction quality).
        ep = str(episodes[0].episode_id)
        if per_episode.get(ep, 0) >= 3:
            return prediction
        per_episode[ep] = per_episode.get(ep, 0) + 1
        selected = list(range(min(3, len(batch.records), args.max_samples-len(report['samples']))))
        with torch.no_grad(), torch.random.fork_rng(devices=[self.device]):
            for row in selected:
                rec = batch.records[row]
                query = lookup[(rec.env_index, rec.query_id)]
                frames = runtime.adapter.buffers[rec.env_index].get_context()
                yaw = frames[-1].yaw + rec.condition.dtheta
                poses = [(f.position, f.yaw) for f in frames]
                poses.append((query.target_position, yaw))
                images = []
                for position, heading in poses:
                    rotation = [0., math.sin(heading/2), 0., math.cos(heading/2)]
                    error = (heading_from_quaternion(rotation)-heading+math.pi)%(2*math.pi)-math.pi
                    assert abs(error) < 1e-5, error
                    obs = self.envs.call_at(rec.env_index, 'get_observation_at',
                        {'source_position': np.asarray(position).tolist(),
                         'source_rotation': rotation, 'keep_agent_at_new_pose': False})
                    rgb = obs['rgb']
                    images.append(np.asarray(rgb).copy())
                encoder = self.raenwm_low_level_synchronizer.encoder
                raw_cls, raw_patch = encoder.forward_raw_cls_and_patch_latents({'rgb': np.stack(images)})
                truth = pack_cls_patch(runtime.normalizer.normalize_cls(raw_cls),
                                       runtime.normalizer.normalize_patch(raw_patch))
                source_error = (truth[:4] - batch.context_latent[row]).abs()
                target_raw_cls = raw_cls[-1].float()
                predicted_cls = prediction.pred_cls_raw[row].float()
                cosine = torch.nn.functional.cosine_similarity
                meta = {'index': len(report['samples']), 'episode': ep,
                    'scene': str(episodes[rec.env_index].scene_id), 'call': calls[0],
                    'query': rec.query_id, 'condition': rec.condition.as_dict(),
                    'source_position': frames[-1].position.tolist(), 'source_yaw': frames[-1].yaw,
                    'target_position': query.target_position.tolist(), 'target_yaw': yaw,
                    'context_positions': [f.position.tolist() for f in frames],
                    'context_yaws': [f.yaw for f in frames],
                    'distance_m': rec.distance_m,
                    'context_rerender_max_abs': source_error.max().item(),
                    'context_rerender_mean_abs': source_error.mean().item(),
                    'cls_pred_target_cos': cosine(predicted_cls, target_raw_cls, dim=0).item(),
                    'cls_source_target_cos': cosine(raw_cls[-2].float(), target_raw_cls, dim=0).item(),
                    'patch_pred_target_cos': cosine(prediction.pred_tokens[row, 1:].float(), truth[-1, 1:].float(), dim=-1).mean().item(),
                    'patch_source_target_cos': cosine(truth[-2, 1:].float(), truth[-1, 1:].float(), dim=-1).mean().item()}
                payload = {'meta': meta, 'rgb': np.stack(images),
                    'context': batch.context_latent[row:row+1].detach().float().cpu(),
                    'curr_delta': batch.curr_delta[row:row+1].detach().cpu(),
                    'rel_t': batch.rel_t[row:row+1].detach().cpu(),
                    'pred_tokens': prediction.pred_tokens[row].detach().float().cpu(),
                    'truth_tokens': truth.detach().float().cpu(),
                    'normalizer_mean': runtime.normalizer.mean.cpu(),
                    'normalizer_var': runtime.normalizer.var.cpu()}
                torch.save(payload, root / ('sample_%03d.pt' % meta['index']))
                report['samples'].append(meta)
                print('NWM_CAPTURE ' + json.dumps(meta), flush=True)
        report['runtime_config'] = str(self.config)
        (root / 'capture.json').write_text(json.dumps(report, indent=2))
        return prediction

    RLTrainer._run_raenwm_rgb_fusion_prediction = capture
    opts = common('0', 1)
    opts.update({'IL.freeze_navigation_backbone': False, 'IL.lr': 2e-6,
        'IL.rgb_fusion_lr': 1e-5, 'EVAL.CKPT_PATH_DIR': str(Path(args.checkpoint).resolve()),
        'EVAL.EPISODE_COUNT': args.episodes, 'EVAL.SAVE_RESULTS': True,
        'EVAL.USE_CKPT_CONFIG': False, 'EVAL.fast_eval': False,
        'CHECKPOINT_FOLDER': str(root / 'checkpoints')+'/',
        'TENSORBOARD_DIR': str(root / 'tensorboard')+'/',
        'RESULTS_DIR': str(root / 'results')+'/'})
    sys.argv = ['run.py', '--exp_name', 'nwm_call_diagnostic', '--run-type', 'eval',
        '--exp-config', 'run_r2r/iter_train_rae_dino_ghost_concat.yaml']
    for key, value in opts.items():
        sys.argv += [key, str(value)]
    report['command'] = list(sys.argv)
    try:
        runpy.run_path('run.py', run_name='__main__')
        report['completed'] = True
    finally:
        RLTrainer._run_raenwm_rgb_fusion_prediction = original
        (root / 'capture.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
