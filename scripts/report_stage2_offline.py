#!/usr/bin/env python3
"""Produce a compact final report only after training and every dev point finish."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    state = read(root/'status.json')
    watch = read(root/'offline_eval/watcher.json')
    selection = read(root/'offline_eval/selection.json')
    audit = read(root/'parameter_audit.json')
    completion = read(root/'completion_audit.json')
    contract = read(root/'contract.json')
    if any(x['status'] != 'completed' for x in (state, watch, selection)):
        raise RuntimeError('training, evaluation and selection must all complete')
    if completion['status'] != 'passed':
        raise RuntimeError('final checkpoint/log audit must pass')
    if not audit['optimizer_exactly_e24'] or audit['initial_sha256'] == audit['final_sha256']:
        raise RuntimeError('parameter audit failed')
    points = read(root/'offline_eval/curve.json')['points']
    metrics = {p['step']: read(root/'offline_eval'/p['report_file'])['results'][0] for p in points}
    expected = set(range(250, state['step']+1, 250)) | {state['step']}
    if set(metrics) != expected:
        raise RuntimeError('missing or unexpected evaluation points')
    for point in points:
        if metrics[point['step']]['checkpoint_sha256'] != point['head_sha256']:
            raise RuntimeError('evaluated head SHA differs from the exported model')
    for m in metrics.values():
        if m['rows'] != 15703 or m['primary_rows'] != 13389 or m['stop_consistent_rows'] != 15703:
            raise RuntimeError('incomplete decision coverage or STOP inconsistency')
    selected = [metrics[p['step']] for p in selection['selected']]
    best, last = selected[0], metrics[state['step']]
    train = [json.loads(line) for line in (root/'train.jsonl').read_text().splitlines()]
    summary = dict(status='completed', source_commit=contract['git_commit'],
        training_steps=state['step'], completed_epochs=state['sampler']['epoch'],
        evaluated_points=len(points), positive_points=sum(m['net_fixes'] > 0 for m in metrics.values()),
        baseline_accuracy=best['base_accuracy'], primary_rows=best['primary_rows'],
        selected=[{k:m[k] for k in ('global_step','checkpoint','checkpoint_sha256','fixes','harms',
            'net_fixes','accuracy','classification_loss','bootstrap')} for m in selected],
        last={k:last[k] for k in ('global_step','fixes','harms','net_fixes','accuracy')},
        parameter_audit=audit, completion_audit=completion, train_config=contract['train_config'],
        limitations=['val_unseen is a reused development set, not independent testing',
                    'checkpoint chosen on this development set; bootstrap interval is not adjusted for selection',
                    'offline fixed-state decisions do not measure navigation SR or SPL'])
    out = root/'final_report'
    out.mkdir(exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    ci = best['bootstrap']['ci95']
    best_path = root / ('head_step_%06d.pt' % best['global_step'])
    lines = ['# E24首轮离线训练结果', '',
        f"训练完成 **{state['step']}** 次更新、**{state['sampler']['epoch']}** 遍有效训练样本；完整开发集评价 **{len(points)}** 个保存点。",
        f"固定一阶段6400基座；E24从零初始化，batch32、AdamW学习率2e-5、seed2、BF16、修正倍率g=1。源码 `{contract['git_commit']}`。", '',
        f"主分母是开发集13389条有效移动决策，基础正确8013条（{best['base_accuracy']*100:.4f}%）。没有future的行保留在分母内。", '',
        '| 模型 | 改对 | 改坏 | 净增加正确 | 决策准确率 |',
        '| --- | ---: | ---: | ---: | ---: |',
        f"| 一阶段基础模型 | 0 | 0 | 0 | {best['base_accuracy']*100:.4f}% |"]
    seen = set()
    for label, m in [('首选',selected[0]),('备选',selected[1]),('最后',last)]:
        if m['global_step'] in seen:
            continue
        seen.add(m['global_step'])
        lines.append(f"| {label}：{m['global_step']}步 | {m['fixes']} | {m['harms']} | {m['net_fixes']:+d} | {m['accuracy']*100:.4f}% |")
    lines += ['', f"首选相对基础模型提升 **{100*best['net_fixes']/best['primary_rows']:+.4f}个百分点**。按整条路线配对重采样的95%区间为 [{ci[0]*100:+.4f}, {ci[1]*100:+.4f}] 个百分点；该区间没有校正从多个检查点中选最优造成的偏差。",
        f"{len(points)}个点中有{summary['positive_points']}个净改善为正。全部保存点均完整覆盖15703条记录，STOP状态全部一致。", '',
        '选点规则提前固定：净改善最多；平分时改坏更少、分类损失更低、步数更早。未做全模型倍率搜索，没有用开发集反向更新权重。', '',
        '短训练与跨进程恢复已通过逐位一致验收。正式训练优化器只包含E24，参数确实变化；一阶段导航、融合、路点与世界模型未加载进入离线训练。', '',
        '**解释边界：** 这是固定访问状态下的离线开发集结果；val_unseen此前已参与一阶段选点，不是独立测试集，也不能换算成整条路线的成功率或SPL。本轮未接回完整导航。', '',
        f"训练机首选模型：`{best_path}`", '',
        f"原始完整曲线：`{root/'offline_eval/curve.json'}`；逐点完整结果：`{root/'offline_eval'}/metrics_step_*.json`。", '',
        '![训练与离线开发曲线](curves.svg)', '']
    (out/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot([x['step'] for x in train], [x['loss'] for x in train], linewidth=.8)
    axes[0].set(xlabel='Training update', ylabel='Training loss (20-update mean)')
    ordered = [metrics[x] for x in sorted(metrics)]
    axes[1].plot([x['global_step'] for x in ordered], [100*x['net_fixes']/x['primary_rows'] for x in ordered],
                 marker='.', linewidth=1, label='Full development set, g=1')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1].scatter([best['global_step']], [100*best['net_fixes']/best['primary_rows']],
                    color='red', label='Selected checkpoint', zorder=3)
    axes[1].set(xlabel='Training update', ylabel='Accuracy change (percentage points)')
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(out/'curves.svg')
    fig.savefig(out/'curves.png', dpi=140)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
