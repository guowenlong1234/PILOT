#!/usr/bin/env python3
"""Audit the paired Stage-2 future-content ablation and select both heads."""
import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path


STAGE1_9200_SHA256 = '87bf7ad691a93abfe2d5030c2c314ef4e630c41d3abbe61fea3b38872686055e'
EXPECTED_STEPS = list(range(250, 6001, 250))


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rank(row):
    return (-row['net_fixes'], row['harms'], row['classification_loss'],
            row['global_step'])


def compact(row, checkpoint=None):
    keys = ('global_step', 'gain', 'checkpoint_sha256', 'fixes', 'harms',
            'net_fixes', 'classification_loss', 'base_accuracy', 'accuracy',
            'net_improvement', 'primary_rows', 'rows', 'bootstrap', 'future_mode',
            'residual_mean_abs', 'residual_saturation_rate', 'all_action_changes')
    result = {key: row[key] for key in keys if key in row}
    result['episode_count'] = len(row['episodes'])
    result['scene_count'] = len(row['scenes'])
    result['checkpoint'] = str(Path(checkpoint or row['checkpoint']).resolve())
    return result


def normalized_contract(contract):
    result = copy.deepcopy(contract)
    result.pop('experiment', None)
    result.get('train_config', {}).pop('future_mode', None)
    result.get('model_config', {}).pop('future_mode', None)
    return result


def dataset_identity(contract):
    datasets = contract.get('dataset', [])
    require(datasets, 'training contract has no dataset provenance')
    identity = []
    for source in datasets:
        provenance = source.get('provenance', {})
        require(provenance.get('stage1_sha256') == STAGE1_9200_SHA256,
                'training data was not collected from the fixed 9200 base')
        require(provenance.get('split') == 'train', 'training dataset split differs')
        identity.append({key: source.get(key) for key in
                         ('manifest_sha256', 'provenance_sha256')})
    return identity


def load_training(root, mode):
    contract = read(root / 'contract.json')
    status = read(root / 'status.json')
    audit = read(root / 'parameter_audit.json')
    require(status.get('status') == 'completed' and status.get('step') == 6000,
            f'{mode}: training did not complete exactly 6000 steps')
    require(contract.get('model_config', {}).get('future_mode') == mode,
            f'{mode}: model future_mode differs')
    require(contract.get('train_config', {}).get('future_mode') == mode,
            f'{mode}: train future_mode differs')
    require(audit.get('optimizer_exactly_e24') is True and audit.get('final_step') == 6000,
            f'{mode}: parameter audit incomplete')
    records = [json.loads(line) for line in (root / 'train.jsonl').read_text().splitlines()
               if line.strip()]
    require([row['step'] for row in records] == list(range(20, 6001, 20)),
            f'{mode}: expected one training log every 20 steps')
    for row in records:
        require(all(math.isfinite(float(row[key])) for key in
                    ('loss', 'gradient_norm', 'delta_abs')),
                f'{mode}: nonfinite training metric at step {row["step"]}')
    return dict(root=str(root.resolve()), contract=contract, status=status,
                audit=audit, dataset=dataset_identity(contract),
                sample_cursors={str(row['step']): row['sampler'] for row in records},
                training_log=records)


def resolve_report(eval_root, record):
    path = Path(record['report_file'])
    return path if path.is_absolute() else eval_root / path


def validate_result(row, mode, step):
    require(row['global_step'] == step and row['gain'] == 1.,
            f'{mode}: evaluation must be gain=1 at step {step}')
    require(row.get('future_mode', mode) == mode,
            f'{mode}: evaluation future_mode differs at step {step}')
    require(row['net_fixes'] == row['fixes'] - row['harms'],
            f'{mode}: inconsistent fixes/harms at step {step}')
    require(row['stop_consistent_rows'] == row['rows'],
            f'{mode}: STOP changed at step {step}')
    require(all(math.isfinite(float(row[key])) for key in
                ('classification_loss', 'accuracy', 'base_accuracy', 'net_improvement')),
            f'{mode}: nonfinite evaluation metric at step {step}')
    require(len(row['scenes']) > 1 and len(row['episodes']) > 1,
            f'{mode}: incomplete development coverage at step {step}')


def load_evaluations(eval_root, train_root, mode):
    step_files = sorted(eval_root.glob('step_*.json'))
    require(len(step_files) == 24, f'{mode}: expected exactly 24 evaluation records')
    rows = []
    dev_identity = None
    for path in step_files:
        record = read(path)
        step = int(record['step'])
        require(record.get('status') in ('complete', 'completed') and
                record.get('exit_code', 0) == 0,
                f'{mode}: incomplete evaluation record {path.name}')
        report_path = resolve_report(eval_root, record)
        require(report_path.is_file(), f'{mode}: missing evaluator report for step {step}')
        require(sha256(report_path) == record['report_sha256'],
                f'{mode}: evaluator report SHA differs at step {step}')
        checkpoint = train_root / f'head_step_{step:06d}.pt'
        require(checkpoint.is_file(), f'{mode}: missing training head at step {step}')
        actual_head_sha = sha256(checkpoint)
        require(actual_head_sha == record['head_sha256'],
                f'{mode}: actual head SHA differs at step {step}')
        report = read(report_path)
        require(report.get('status') == 'complete' and len(report['results']) == 1,
                f'{mode}: evaluator report must contain one complete result')
        identity = report.get('dataset')
        require(identity, f'{mode}: evaluator report lacks development identity')
        dev_identity = identity if dev_identity is None else dev_identity
        require(identity == dev_identity, f'{mode}: development data changed between steps')
        row = report['results'][0]
        validate_result(row, mode, step)
        require(row['checkpoint_sha256'] == actual_head_sha,
                f'{mode}: result checkpoint SHA differs at step {step}')
        row = dict(row, future_mode=mode, actual_checkpoint=str(checkpoint.resolve()))
        rows.append(row)
    rows.sort(key=lambda row: row['global_step'])
    require([row['global_step'] for row in rows] == EXPECTED_STEPS,
            f'{mode}: evaluation steps are not the fixed 250-step grid')
    return rows, dev_identity


def paired_scene_bootstrap(full, none, samples=20000, seed=20260921):
    require(set(full['scenes']) == set(none['scenes']),
            'selected full/none scene sets differ')
    units = []
    for scene in sorted(full['scenes']):
        a, b = full['scenes'][scene], none['scenes'][scene]
        require(a['primary_rows'] == b['primary_rows'],
                f'full/none primary rows differ in scene {scene}')
        units.append((a['correct'] - b['correct'], a['primary_rows']))
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [units[rng.randrange(len(units))] for _ in units]
        numerator, denominator = map(sum, zip(*chosen))
        if denominator:
            draws.append(numerator / denominator)
    draws.sort()
    low = draws[int(.025 * (len(draws) - 1))]
    high = draws[int(.975 * (len(draws) - 1))]
    observed = sum(item[0] for item in units) / sum(item[1] for item in units)
    return dict(unit='scene', estimand='full_minus_none_decision_accuracy',
                scenes=len(units), samples=samples, seed=seed,
                observed=observed, ci95=[low, high])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True,
                        help='experiment root containing full/{train,eval} and none/{train,eval}')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    runs, raw, dev = {}, {}, {}
    for mode in ('full', 'none'):
        train_root, eval_root = root / mode / 'train', root / mode / 'eval'
        runs[mode] = load_training(train_root, mode)
        raw[mode], dev[mode] = load_evaluations(eval_root, train_root, mode)

    require(runs['full']['audit']['initial_sha256'] == runs['none']['audit']['initial_sha256'],
            'full/none seeded initial weights differ')
    require(runs['full']['sample_cursors'] == runs['none']['sample_cursors'],
            'full/none logged sample order differs')
    require(normalized_contract(runs['full']['contract']) ==
            normalized_contract(runs['none']['contract']),
            'training contracts differ beyond future_mode and experiment path')
    require(runs['full']['dataset'] == runs['none']['dataset'],
            'full/none training manifest/provenance SHA differs')
    require(dev['full'] == dev['none'], 'full/none development dataset SHA differs')

    # Frozen-base metrics and denominators must match at every paired point.
    by_mode = {mode: {row['global_step']: row for row in rows}
               for mode, rows in raw.items()}
    for step in EXPECTED_STEPS:
        full, none = by_mode['full'][step], by_mode['none'][step]
        for key in ('rows', 'primary_rows', 'base_correct', 'base_accuracy'):
            require(full[key] == none[key], f'paired frozen metric differs at step {step}: {key}')
        require(set(full['scenes']) == set(none['scenes']),
                f'paired scene coverage differs at step {step}')

    selected_raw = {mode: min(rows, key=rank) for mode, rows in raw.items()}
    selected = {mode: compact(row, row['actual_checkpoint'])
                for mode, row in selected_raw.items()}
    scene_interval = paired_scene_bootstrap(selected_raw['full'], selected_raw['none'])
    curves = []
    for step in EXPECTED_STEPS:
        curves.append(dict(step=step,
            full=compact(by_mode['full'][step], by_mode['full'][step]['actual_checkpoint']),
            none=compact(by_mode['none'][step], by_mode['none'][step]['actual_checkpoint'])))
    result = dict(format_version='stage2-future-ablation-report-v1', status='complete',
        experiment_root=str(root), stage1_sha256=STAGE1_9200_SHA256,
        same_initialization=True, same_training_sample_order=True,
        same_training_contract_except_future_mode_and_experiment=True,
        same_training_data_sha=True, same_development_data_sha=True,
        evaluation_steps=EXPECTED_STEPS, selected=selected, curves=curves,
        final_step={mode: compact(by_mode[mode][6000],
                                  by_mode[mode][6000]['actual_checkpoint'])
                    for mode in ('full', 'none')},
        selected_full_vs_none_scene_bootstrap=scene_interval,
        development=dict(dataset=dev['full'],
            rows=selected_raw['full']['rows'],
            primary_rows=selected_raw['full']['primary_rows'],
            episodes=len(selected_raw['full']['episodes']),
            scenes=len(selected_raw['full']['scenes'])),
        training={mode: {key: runs[mode][key] for key in
                         ('root', 'dataset', 'status', 'audit')}
                  for mode in ('full', 'none')},
        limitations=[
            '两个模型都用同一个开发集进行24次检查点选择，存在多轮选点偏差。',
            '本实验只有一个训练随机种子，不能估计训练随机性带来的波动。',
            '按场景重采样区间只有11个重采样单位，且没有包含选点过程的不确定性。',
            'none仍保留stage1冻结特征融合和真实future有效mask，只移除了future token内容。',
            '离线决策指标不等于完整导航SR/SPL。'])

    out = root / 'final_report'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 9200评分头未来内容消融', '',
        '两组使用相同数据、初始化、样本顺序、监督、掩码和训练预算。none组仅把未来token内容固定为零；真实future有效mask和stage1冻结特征融合仍保留。', '',
        '|模式|选中步数|改对|改坏|净改善|决策准确率|相对基座提升|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in ('full', 'none'):
        row = selected[mode]
        lines.append(f"|{mode}|{row['global_step']}|{row['fixes']}|{row['harms']}|{row['net_fixes']:+d}|{row['accuracy']*100:.4f}%|{row['net_improvement']*100:+.4f}个百分点|")
    interval = scene_interval
    lines += ['', f"选中点full−none决策准确率：{interval['observed']*100:+.4f}个百分点；按场景重采样95%区间：[{interval['ci95'][0]*100:+.4f}, {interval['ci95'][1]*100:+.4f}]个百分点。", '',
              '## 共同训练步曲线', '', '|步数|full净改善|none净改善|full−none准确率百分点|',
              '|---:|---:|---:|---:|']
    for pair in curves:
        delta = (pair['full']['accuracy'] - pair['none']['accuracy']) * 100
        lines.append(f"|{pair['step']}|{pair['full']['net_fixes']:+d}|{pair['none']['net_fixes']:+d}|{delta:+.4f}|")
    lines += ['', '## 解释边界', ''] + ['- ' + item for item in result['limitations']]
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status='complete', summary=str((out/'summary.json').resolve()),
                          report=str((out/'report.md').resolve())), ensure_ascii=False))


if __name__ == '__main__':
    main()
