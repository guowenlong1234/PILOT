"""Offline, paired decision metrics. This does not measure navigation success."""
from collections import defaultdict
import numpy as np
import torch

CALIBRATION_GAINS = (0., .5, .75, 1., 1.25, 1.5)


def decision_records(batch, delta, gain=1.):
    """Include every row; primary denominator never requires a valid future."""
    if gain not in CALIBRATION_GAINS:
        raise ValueError('gain outside preregistered calibration grid')
    b = {k: v.detach().cpu() for k, v in batch.items() if torch.is_tensor(v)}
    delta = delta.detach().float().cpu()
    if not torch.isfinite(delta).all():
        raise ValueError('nonfinite residual')
    records = []
    for i in range(len(delta)):
        n = int(b['ghost_valid_mask'][i].sum())
        base = b['base_logits'][i, :n].float()
        corrected = base.clone()
        valid = b['topk_valid_mask'][i]
        applied = (gain * delta[i]).clamp(-1., 1.).masked_fill(~valid, 0.)
        if b['base_stop'][i] or b['no_vp_left'][i]:
            applied.zero_()
        indices = b['topk_base_indices'][i]
        for slot in valid.nonzero().flatten().tolist():
            index = int(indices[slot])
            if not 0 <= index < n:
                raise ValueError('invalid candidate mapping')
            corrected[index] += applied[slot]
        base_stop = bool(b['base_stop'][i]) or bool(b['no_vp_left'][i]) or n == 0
        base_action = -1 if base_stop else int(base.argmax())
        action = -1 if base_stop else int(corrected.argmax())
        teacher = int(b['teacher_base_index'][i])
        primary = (bool(b['teacher_valid'][i]) and not bool(b['teacher_stop'][i])
                   and not base_stop and 0 <= teacher < n)
        rank = int(b['teacher_rank_in_topk'][i])
        teacher_future = rank >= 0 and bool(valid[rank])
        base_ok = primary and base_action == teacher
        ok = primary and action == teacher
        ce = float(-torch.log_softmax(corrected, 0)[teacher]) if primary else 0.
        def entropy(logits):
            if not n: return 0.
            lp = torch.log_softmax(logits, 0)
            return float(-(lp.exp() * lp).sum())
        margin = float(base.topk(2).values.diff().abs()[0]) if n > 1 else float('inf')
        records.append(dict(primary=primary, base_correct=int(base_ok), correct=int(ok),
            fixes=int(primary and not base_ok and ok), harms=int(primary and base_ok and not ok),
            flips=int(primary and action != base_action), action_changed=int(action != base_action),
            stop_consistent=int((action == -1) == base_stop), base_stop=int(base_stop),
            teacher_in_top5=rank >= 0, teacher_future=teacher_future,
            future_count=int(valid.sum()), ce=ce, base_entropy=entropy(base), entropy=entropy(corrected),
            residual_abs_sum=float(applied[valid].abs().sum()), residual_slots=int(valid.sum()),
            residual_saturated=int((applied[valid].abs() >= .99).sum()),
            margin_group='0-.25' if margin < .25 else '.25-.5' if margin < .5 else '.5-1' if margin < 1 else '1+',
            episode_id=str(batch['episode_id'][i]), scene_id=str(batch['scene_id'][i])))
    return records


def summarize(records):
    selected = [r for r in records if r['primary']]
    n = len(selected)
    result = dict(rows=len(records), primary_rows=n)
    for key in ('base_correct', 'correct', 'fixes', 'harms', 'flips'):
        result[key] = sum(r[key] for r in selected)
    result['net_fixes'] = result['fixes'] - result['harms']
    for out, key in (('base_accuracy','base_correct'), ('accuracy','correct'), ('flip_rate','flips'), ('net_improvement','net_fixes')):
        result[out] = result[key] / n if n else None
    result['classification_loss'] = sum(r['ce'] for r in selected) / n if n else None
    result['stop_consistent_rows'] = sum(r['stop_consistent'] for r in records)
    result['base_stop_rows'] = sum(r['base_stop'] for r in records)
    result['all_action_changes'] = sum(r['action_changed'] for r in records)
    slots = sum(r['residual_slots'] for r in records)
    result['residual_mean_abs'] = sum(r['residual_abs_sum'] for r in records) / slots if slots else 0.
    result['residual_saturation_rate'] = sum(r['residual_saturated'] for r in records) / slots if slots else 0.
    for key in ('base_entropy','entropy'):
        result[key] = sum(r[key] for r in selected) / n if n else None
    return result


def paired_episode_bootstrap(records, samples=2000, seed=2):
    grouped = defaultdict(lambda: [0, 0])
    for row in records:
        item = grouped[(row['scene_id'], row['episode_id'])]
        item[0] += row['fixes'] - row['harms']
        item[1] += int(row['primary'])
    values = np.asarray(list(grouped.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        chosen = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        if chosen[1]: draws.append(chosen[0] / chosen[1])
    return dict(unit='paired_episode', estimand='decision_weighted_accuracy_difference',
                samples=samples, seed=seed, episodes=len(values),
                ci95=np.quantile(draws, [.025,.975]).tolist() if draws else None)


def build_report(records, bootstrap_samples=2000):
    result = summarize(records)
    result['strata'] = {}
    for key in ('teacher_in_top5','teacher_future','future_count','margin_group'):
        groups = defaultdict(list)
        for row in records: groups[str(row[key])].append(row)
        result['strata'][key] = {name:summarize(rows) for name,rows in sorted(groups.items())}
    for key, name in (('scene_id','scenes'),('episode_id','episodes')):
        groups = defaultdict(list)
        for row in records: groups[row[key]].append(row)
        result[name] = {identity:summarize(rows) for identity,rows in sorted(groups.items())}
    result['bootstrap'] = paired_episode_bootstrap(records, bootstrap_samples)
    return result


def selection_key(result):
    """Plan step 9: g=1, net fixes, harms, CE, earliest step."""
    if result['gain'] != 1.: raise ValueError('checkpoint selection requires gain=1')
    return (-result['net_fixes'], result['harms'],
            result['classification_loss'] if result['classification_loss'] is not None else float('inf'),
            result['global_step'])
