#!/usr/bin/env python3
"""Audit completed stage2 A/B/C artifacts and write a Chinese comparison report."""
import argparse
import hashlib
import json
import math
from pathlib import Path

DEV_SHA = '25d9666e89e482186b9d6f7436dcf5c9934f82cd9265fa0658c0f1ec632ba93d'
STEPS = set(range(250, 6001, 250))
GAINS = (0., .5, .75, 1., 1.25, 1.5)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rank(row):
    return (-row['net_fixes'], row['harms'], row['classification_loss'],
            row['global_step'], row['gain'])


def validate_report(report):
    require(report['status'] == 'complete', 'incomplete evaluator report')
    require(len(report['dataset']) == 1 and report['dataset'][0]['manifest_sha256'] == DEV_SHA,
            'development manifest differs')
    for row in report['results']:
        require(row['rows'] == 15703 and row['primary_rows'] == 13389, 'development row count differs')
        require(len(row['scenes']) == 11 and len(row['episodes']) == 1839, 'development coverage differs')
        require(row['stop_consistent_rows'] == 15703, 'STOP changed')
        require(row['base_correct'] == 8013, 'frozen baseline decisions differ')
        require(row['strata']['future_count']['0']['all_action_changes'] == 0, 'no-future row changed action')
        require(row['net_fixes'] == row['fixes'] - row['harms'], 'inconsistent net fixes')
        require(all(math.isfinite(row[k]) for k in ('classification_loss', 'accuracy', 'net_improvement')),
                'nonfinite selection metric')
        require(row['bootstrap']['episodes'] == 1839, 'bootstrap coverage differs')


def compact(row):
    keys = ('global_step', 'gain', 'checkpoint', 'checkpoint_sha256', 'fixes', 'harms',
            'net_fixes', 'classification_loss', 'accuracy', 'base_accuracy', 'net_improvement',
            'residual_mean_abs', 'residual_saturation_rate', 'bootstrap')
    return {k: row[k] for k in keys}


def audit_checkpoint(root, contract):
    import torch
    head_path = root / 'head_step_006000.pt'
    state_path = root / 'state_step_006000.pt'
    head = torch.load(head_path, map_location='cpu', weights_only=False)
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    require(head['global_step'] == state['global_step'] == 6000, 'final checkpoint step differs')
    require(head['contract'] == state['contract'] == contract, 'checkpoint contract differs')
    require(state['scheduler'] is None and state['optimizer']['state'], 'missing optimizer/unexpected scheduler')
    tensors = 0
    def finite(value):
        nonlocal tensors
        if torch.is_tensor(value):
            tensors += 1
            require(bool(torch.isfinite(value).all()), 'nonfinite checkpoint tensor')
        elif isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, (list, tuple)):
            for child in value: finite(child)
    finite(state['future_head_state_dict'])
    finite(state['optimizer'])
    require(head['future_head_state_dict'].keys() == state['future_head_state_dict'].keys(), 'head/state keys differ')
    require(all(torch.equal(value, state['future_head_state_dict'][key])
                for key, value in head['future_head_state_dict'].items()), 'head/state weights differ')
    return dict(status='passed', checked_tensors=tensors, state_sha256=sha(state_path),
                head_sha256=sha(head_path), sampler=state['sampler'], torch=torch.__version__)


def load_run(path, variant, verify_checkpoints):
    root = Path(path).resolve()
    if root.name == 'offline_eval': root = root.parent
    out = root / 'offline_eval'
    watcher, status, contract = read(out / 'watcher.json'), read(root / 'status.json'), read(root / 'contract.json')
    require(watcher['status'] == 'completed' and watcher['exit_code'] == 0, f'{variant}: watcher incomplete')
    require(status['status'] == 'completed', f'{variant}: training incomplete')
    if variant != 'baseline':
        require(status['step'] == status['target'] == 6000, f'{variant}: wrong training extent')
        require(watcher['evaluated_heads'] == 24, f'{variant}: expected 24 evaluations')
        require(contract['train_config']['variant'] == variant, 'variant contract mismatch')
    paths = sorted(out.glob('step_*.json'))
    if variant != 'baseline': require(len(paths) == 24, 'unexpected step records')
    training = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    training = [record for record in training if record['step'] <= 6000]
    require([r['step'] for r in training] == list(range(20, 6001, 20)), 'training log has missing/duplicate steps')
    for record in training:
        require(all(math.isfinite(record[k]) for k in ('loss', 'gradient_norm', 'delta_abs')), 'nonfinite training metric')
    parameter_audit = read(root/'parameter_audit.json')
    require(parameter_audit['optimizer_exactly_e24'], 'optimizer includes non-E24 parameters')
    if variant != 'baseline':
        require(parameter_audit['final_step'] == 6000 and parameter_audit['initial_sha256'] != parameter_audit['final_sha256'],
                'missing E24 parameter updates')
    rows = []
    for path in paths:
        record = read(path)
        step = int(record['step'])
        if step > 6000: continue
        require(record['status'] == 'completed' and record['exit_code'] == 0, f'unfinished {path}')
        report_path = out / record['report_file']
        require(sha(report_path) == record['report_sha256'], f'report SHA differs: {path}')
        head_path = root / f'head_step_{step:06d}.pt'
        require(sha(head_path) == record['head_sha256'], f'head SHA differs: {path}')
        report = read(report_path)
        validate_report(report)
        require(len(report['results']) == 1, 'expected single gain1 evaluation')
        row = report['results'][0]
        require(row['global_step'] == step and row['gain'] == 1., 'step/gain differs')
        require(row['checkpoint_sha256'] == record['head_sha256'] and row['checkpoint'] == record['remote_head'],
                'report checkpoint provenance differs')
        rows.append(row)
    require({r['global_step'] for r in rows} == STEPS and len(rows) == 24, 'missing/duplicate first6000 evaluations')
    selected = min(rows, key=rank)
    neighbors = [compact(r) for r in rows if abs(r['global_step']-selected['global_step']) <= 500]
    last = [compact(r) for r in rows if r['global_step'] > 5000]
    scene = sorted((dict(scene=k, **{x: v[x] for x in ('primary_rows','fixes','harms','net_fixes')})
                    for k,v in selected['scenes'].items()), key=lambda r: -r['net_fixes'])
    positive = sum(max(0, s['net_fixes']) for s in scene)
    concentration = dict(positive_scenes=sum(s['net_fixes'] > 0 for s in scene),
                         top_positive_share=max(0, scene[0]['net_fixes']) / positive if positive else None,
                         selected_net_excluding_best_scene=selected['net_fixes']-scene[0]['net_fixes'])
    audit = audit_checkpoint(root, contract) if verify_checkpoints and variant != 'baseline' else dict(status='not_requested')
    return dict(root=str(root), contract=contract, watcher_source_commit=watcher.get('source_commit'),
                status=status, parameter_audit=parameter_audit, sample_cursors={r['step']:r['sampler'] for r in training}, selected=compact(selected), ranked=[compact(r) for r in sorted(rows,key=rank)],
                curve=[compact(r) for r in rows], selected_strata=selected['strata'], neighbors=neighbors, last1000=last, scenes=scene,
                scene_concentration=concentration, checkpoint_audit=audit), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-dir', 'a-report', 'b-dir', 'c-dir', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--verify-checkpoints', action='store_true', help='CPU torch audit of B/C final full states')
    args = parser.parse_args()
    runs, raw = {}, {}
    for variant, path in [('baseline',args.baseline_dir), ('B',args.b_dir), ('C',args.c_dir)]:
        runs[variant], raw[variant] = load_run(path, variant, args.verify_checkpoints)
    base = runs['baseline']['contract']
    # Original training stores model/loss metadata on the head, not its contract.
    import torch
    original_head = torch.load(Path(runs['baseline']['root']) / 'head_step_006000.pt',
                               map_location='cpu', weights_only=False)
    require(original_head['contract'] == base, 'baseline head contract differs')
    base_model, base_loss = original_head['model_config'], original_head['loss_config']
    del original_head
    for variant in ('B', 'C'):
        other = runs[variant]['contract']
        require(runs[variant]['sample_cursors'] == runs['baseline']['sample_cursors'], 'actual logged sample order differs')
        require(runs[variant]['parameter_audit']['initial_sha256'] == runs['baseline']['parameter_audit']['initial_sha256'],
                'seeded initial weights differ')
        require(other['dataset'] == base['dataset'], 'training datasets/order provenance differs')
        for key in ('seed','block_episodes','batch_size','lr','precision','max_epochs'):
            require(other['train_config'][key] == base['train_config'][key], f'training contract differs: {key}')
        for key in ('optimizer', 'scheduler', 'gradient_clip', 'trainable_module', 'initialization'):
            require(other[key] == base[key], f'training contract differs: {key}')
    expected_b_loss = dict(base_loss, decision_row_policy='all_move_with_future')
    expected_c_model = dict(base_model, candidate_context_mode='all_present')
    require(runs['B']['contract']['model_config'] == base_model and
            runs['B']['contract']['loss_config'] == expected_b_loss, 'B changed more than supervision policy')
    require(runs['C']['contract']['model_config'] == expected_c_model and
            runs['C']['contract']['loss_config'] == base_loss, 'C changed more than candidate context')
    require(runs['B']['checkpoint_audit'].get('sampler') == runs['C']['checkpoint_audit'].get('sampler'),
            'B/C final sampling positions differ')
    a = read(args.a_report)
    validate_report(a)
    expected = {(step, gain) for step in (4000,4750) for gain in GAINS}
    require(len(a['results']) == 12 and {(r['global_step'],r['gain']) for r in a['results']} == expected,
            'A must cover exactly 2 checkpoints x 6 preregistered gains')
    for row in a['results']:
        baseline = next(r for r in raw['baseline'] if r['global_step'] == row['global_step'])
        require(row['checkpoint_sha256'] == baseline['checkpoint_sha256'], 'A source head SHA differs')
        if row['gain'] == 0: require(row['all_action_changes'] == 0, 'gain0 changed action')
        if row['gain'] == 1:
            require((row['fixes'],row['harms']) == {4750:(105,81),4000:(120,103)}[row['global_step']], 'A gain1 replay differs')
            for key in ('fixes','harms','net_fixes','base_correct','correct'):
                require(row[key] == baseline[key], 'A baseline replay metric differs')
    ranked_a = sorted(a['results'], key=rank)
    result = dict(status='complete', dev_manifest_sha256=DEV_SHA, rows=15703, primary_rows=13389,
                  episodes=1839, scenes=11, stop_unchanged=True, same_training_data_seed_order_contract=True,
                  A=dict(report=str(Path(args.a_report).resolve()), report_sha256=sha(args.a_report),
                         ranked=[compact(r) for r in ranked_a], selected=compact(ranked_a[0]),
                         selected_scenes=ranked_a[0]['scenes']), runs=runs,
                  limitations=['同一开发集上选择检查点与gain，存在多重尝试的选择偏差。',
                               '配对区间仅比较各模型与冻结基座，不能据此断言A/B/C互相显著优于对方。',
                               '按episode重采样，未把场景内相关性或选点过程纳入区间。',
                               '离线移动决策准确率不是完整导航SR/SPL；本报告不含在线评测。'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines = ['# E24 A、B、C 离线对比', '',
             '固定完整开发集：1839条路线、11个场景、15703行，其中13389行纳入移动决策主指标；STOP全部保持不变。', '',
             'A：原模型4000/4750步各测六个固定倍率；B：扩展决策监督行；C：让无有效未来的现有候选参与上下文。B/C均从头训练6000步，每250步评价，共24点。', '',
             '排序：净改对数降序、改坏数升序、交叉熵升序、步数升序；A最后按倍率升序。B/C固定倍率1。', '',
             '|实验|步数|倍率|改对|改坏|净改善|移动准确率|相对基座提升百分点|配对95%区间百分点|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for label, row in [('原方案≤6000',runs['baseline']['selected']),('A',ranked_a[0]),('B',runs['B']['selected']),('C',runs['C']['selected'])]:
        ci = row['bootstrap']['ci95']
        lines.append(f"|{label}|{row['global_step']}|{row['gain']:g}|{row['fixes']}|{row['harms']}|{row['net_fixes']:+d}|{row['accuracy']*100:.4f}%|{row['net_improvement']*100:+.4f}|{[round(x*100,4) for x in ci]}|")
    lines += ['', '## A 全部固定网格', '', '|步数|倍率|改对|改坏|净改善|分类损失|', '|---:|---:|---:|---:|---:|---:|']
    for r in ranked_a:
        lines.append(f"|{r['global_step']}|{r['gain']:g}|{r['fixes']}|{r['harms']}|{r['net_fixes']:+d}|{r['classification_loss']:.6f}|")
    lines += ['', 'A选中点逐场景表现：', '', '|场景|主指标行数|改对|改坏|净改善|', '|---|---:|---:|---:|---:|']
    a_scenes = sorted(ranked_a[0]['scenes'].items(), key=lambda item: -item[1]['net_fixes'])
    for name, scene in a_scenes:
        lines.append(f"|{name}|{scene['primary_rows']}|{scene['fixes']}|{scene['harms']}|{scene['net_fixes']:+d}|")
    lines += ['', f"A去掉净改善最多场景后总净改善：{ranked_a[0]['net_fixes']-a_scenes[0][1]['net_fixes']:+d}。"]
    for label, run in runs.items():
        lines += ['', '## '+label, '', '选点附近±500步（步数:净改善）：'+', '.join(f"{r['global_step']}:{r['net_fixes']:+d}" for r in run['neighbors']), '',
                  '最后1000步（5000<步数≤6000）：'+', '.join(f"{r['global_step']}:{r['net_fixes']:+d}" for r in run['last1000']), '',
                  '选中点逐场景表现：', '', '|场景|主指标行数|改对|改坏|净改善|', '|---|---:|---:|---:|---:|']
        for s in run['scenes']:
            lines.append(f"|{s['scene']}|{s['primary_rows']}|{s['fixes']}|{s['harms']}|{s['net_fixes']:+d}|")
        c = run['scene_concentration']
        lines += ['', f"正收益场景 {c['positive_scenes']}/11；去掉净改善最多的场景后，总净改善为 {c['selected_net_excluding_best_scene']:+d}。", '',
                  '训练配置、模型/损失差异、源码提交和完整排名保存在 summary.json。最终恢复状态审计：'+run['checkpoint_audit']['status']+'。']
    lines += ['', '## 正确候选是否有未来预测（选中点）', '',
              '|实验|正确候选有未来|行数|改对|改坏|净改善|', '|---|---|---:|---:|---:|---:|']
    for label, run in runs.items():
        for present, group in run['selected_strata']['teacher_future'].items():
            lines.append(f"|{label}|{'有' if present == 'True' else '无'}|{group['primary_rows']}|{group['fixes']}|{group['harms']}|{group['net_fixes']:+d}|")
    lines += ['', '## 解释边界', ''] + ['- '+s for s in result['limitations']]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; summary.json/report.md complete, curves.svg skipped')
    else:
        fig, ax = plt.subplots(figsize=(10,5))
        for label, run in runs.items():
            ax.plot([r['global_step'] for r in run['curve']], [r['net_fixes'] for r in run['curve']], marker='.', label=label)
        ax.axhline(0,color='gray',linewidth=.7)
        ax.set(xlabel='Training step',ylabel='Fixes minus harms (full dev, gain=1)')
        ax.legend(); fig.tight_layout(); fig.savefig(out/'curves.svg'); fig.savefig(out/'curves.png', dpi=150); plt.close(fig)
    print(json.dumps(dict(status='complete',output=str(out.resolve())),ensure_ascii=False))


if __name__ == '__main__':
    main()
