"""Audit native9200 base/full/none navigation and paired scene uncertainty."""
import argparse
import gzip
import json
import math
from pathlib import Path
import numpy as np
from vlnce_baselines.nwm.active_lookahead.stage2_data import atomic_json,sha256


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args()
    exp=Path(a.experiment).resolve();root=exp/'navigation';selected=read(exp/'final_report/summary.json')['selected']
    with gzip.open('data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz','rt') as f:
        scenes={str(ep['episode_id']):str(ep['scene_id']) for ep in json.load(f)['episodes']}
    if len(scenes)!=1839:raise ValueError('unexpected val_unseen coverage')
    keys=['success','spl','ndtw','sdtw','distance_to_goal','path_length']
    runs={};sources={};metrics={};timing={}
    for mode in ('base','full','none'):
        run=root/mode;state=read(run/'status.json')
        if state['status']!='completed' or state['exit_code']!=0:raise ValueError('incomplete navigation')
        files=list((run/'results').rglob('stats_ep_*.json'))
        if len(files)!=1:raise ValueError('ambiguous episode results')
        rows=read(files[0]);prov=read(run/'provenance.json')
        if set(rows)!=set(scenes) or set(prov['episode_ids'])!=set(scenes):raise ValueError('episode coverage mismatch')
        if not all(math.isfinite(float(r[k])) for r in rows.values() for k in keys):raise ValueError('nonfinite metrics')
        if prov['stage1_sha256']!='87bf7ad691a93abfe2d5030c2c314ef4e630c41d3abbe61fea3b38872686055e':raise ValueError('wrong base')
        sources[mode]=dict(path=str(files[0]),sha256=sha256(files[0]),provenance=prov)
        runs[mode]=rows;metrics[mode]={k:sum(v[k] for v in rows.values())/len(rows) for k in keys}
        timing[mode]=state['elapsed_seconds']
        if mode!='base':
            summary=read(run/'online/online_summary.json');parity=read(root/f'parity_{mode}.json')
            if (summary['transfer_mode']!='native_9200' or summary['future_mode']!=mode or summary['gain']!=1 or
                summary['head_sha256']!=selected[mode]['checkpoint_sha256'] or summary['counts']['episodes']!=1839 or
                summary['teacher_calls']!=0 or not summary['head_retrained']):raise ValueError('online head contract differs')
            if parity['status']!='passed' or not parity['exact_actions_metrics']:raise ValueError('parity failed')
            if not read(run/'online/freeze_report.json')['comparison']['exact_match']:raise ValueError('parameter freeze failed')
            if not read(run/'online/world_freeze.json')['exact_match']:raise ValueError('world freeze failed')
    for mode in ('full','none'):
        for key in ('stage1_sha256','assets','commit','split','seed','environments','compile','episode_ids'):
            if sources[mode]['provenance'][key]!=sources['base']['provenance'][key]:raise ValueError('unpaired computation: '+key)
    names=sorted(set(scenes.values()));groups=[[i for i in scenes if scenes[i]==s] for s in names]
    counts=np.array([len(g) for g in groups]);draw=np.random.default_rng(2).integers(len(groups),size=(10000,len(groups)))
    comparisons={}
    for left,right in (('full','base'),('none','base'),('full','none')):
        totals=np.array([[sum(runs[left][i][k]-runs[right][i][k] for i in group) for k in keys] for group in groups])
        values=totals[draw].sum(1)/counts[draw].sum(1)[:,None]
        comparisons[left+'_minus_'+right]=dict(differences={k:metrics[left][k]-metrics[right][k] for k in keys},
            ci95={k:np.quantile(values[:,j],[.025,.975]).tolist() for j,k in enumerate(keys)},
            fixed=sum(runs[left][i]['success']>runs[right][i]['success'] for i in scenes),
            harmed=sum(runs[left][i]['success']<runs[right][i]['success'] for i in scenes))
    out=root/'final_report';result=dict(status='passed',episodes=1839,metrics=metrics,comparisons=comparisons,
        process_seconds=timing,source_files=sources,selected=selected,scene_bootstrap_samples=10000,
        limitation='single seed; reused development set; no correction for checkpoint selection; none retains actual future availability mask and first-stage fused information; q1 computed in both groups')
    atomic_json(out/'summary.json',result)
    lines=['# 9200 有/无未来特征导航对照','',
           '三组均完整覆盖1839条路线；两评分头在9200数据上从相同初始化训练6000步，以同一规则选点，倍率固定1。','',
           '|组别|SR %|SPL %|nDTW %|进程耗时 分钟|','|---|---:|---:|---:|---:|']
    for mode,label in [('base','9200基线'),('full','使用未来内容'),('none','不使用未来内容')]:
        m=metrics[mode];lines.append(f"|{label}|{100*m['success']:.4f}|{100*m['spl']:.4f}|{100*m['ndtw']:.4f}|{timing[mode]/60:.2f}|")
    c=comparisons['full_minus_none'];lines+=['',f"有未来减无未来：SR {100*c['differences']['success']:+.4f}个百分点，按场景配对95%区间 {[round(100*x,4) for x in c['ci95']['success']]}。",'',
      '无未来组仍使用一阶段已融合的信息和相同的未来有效性标记。两组均计算q1以保持候选支持一致，因此此处耗时不代表删除未来生成后的部署加速。', '',
      '单种子且开发集已参与选点，区间未校正选点偏差。完整逐项统计及来源SHA见summary.json。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status='passed',output=str(out))))

if __name__=='__main__':main()
