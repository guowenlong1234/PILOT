"""Validate full online comparison and report paired scene-cluster intervals."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();out=Path(a.output).resolve()
    state=read(root/'pipeline.json');assert state['status']=='completed' and state['exit_code']==0
    parity=read(root/'parity.json')
    assert parity['status']=='passed' and parity.get('exact_actions_metrics',parity['exact_logits_actions_metrics'])
    assert parity.get('logits_max_abs_error',0)<=parity.get('logits_tolerance',0)
    with gzip.open('data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz','rt') as f:
        scenes={str(ep['episode_id']):str(ep['scene_id']) for ep in json.load(f)['episodes']}
    runs={};sources={}
    for name in ('full_base','full_best'):
        run=root/name;status=read(run/'status.json');assert status['status']=='completed' and status['exit_code']==0
        f=list((run/'results').rglob('stats_ep_*.json'));assert len(f)==1
        rows=read(f[0]);assert set(rows)==set(scenes) and len(rows)==1839
        assert set(read(run/'provenance.json')['episode_ids'])==set(rows)
        assert all(math.isfinite(v) for r in rows.values() for v in r.values())
        runs[name]=rows;sources[name]=dict(path=str(f[0]),sha256=digest(f[0]),provenance=read(run/'provenance.json'))
    for key in ('stage1_sha256','assets','split','seed','environments','compile','episode_ids'):
        assert sources['full_base']['provenance'][key]==sources['full_best']['provenance'][key],key
    base_step=sources['full_base']['provenance'].get('stage1_iteration',6400)
    online=read(root/'full_best/online/online_summary.json')
    if base_step==9200:
        assert online['transfer_mode']=='6400_to_9200' and not online['head_retrained']
        assert online['deployment_base_sha256']==sources['full_base']['provenance']['stage1_sha256']
        assert online['head_training_base_sha256']==sources['full_best']['provenance']['head_training_base_sha256']
    assert online['counts']['episodes']==1839 and online['teacher_calls']==0
    assert read(root/'full_best/online/freeze_report.json')['comparison']['exact_match']
    assert read(root/'full_best/online/world_freeze.json')['exact_match']
    diag=read(root/'full_best/online/episode_diagnostics.json');assert set(diag)==set(scenes)
    assert all(r['completed'] for r in diag.values())
    for k in ('decisions','action_flips','future_valid_slots'):
        expected='rows' if k=='decisions' else k
        assert sum(r[k] for r in diag.values())==online['counts'][expected]
    keys=['success','spl','ndtw','sdtw','distance_to_goal','path_length','steps_taken','collisions']
    comparison=read(root/'comparison.json');per_scene={}
    for scene in sorted(set(scenes.values())):
        ids=[i for i,s in scenes.items() if s==scene]
        per_scene[scene]=dict(episodes=len(ids),differences={k:sum(runs['full_best'][i][k]-runs['full_base'][i][k] for i in ids)/len(ids) for k in keys},
            action_flips=sum(diag[i]['action_flips'] for i in ids))
    # Resample the 11 paired scenes, retaining all episodes within each draw.
    names=sorted(per_scene);counts=np.array([per_scene[s]['episodes'] for s in names])
    totals=np.array([[per_scene[s]['differences'][k]*per_scene[s]['episodes'] for k in keys] for s in names])
    draws=np.random.default_rng(2).integers(len(names),size=(10000,len(names)))
    estimates=totals[draws].sum(1)/counts[draws].sum(1)[:,None]
    intervals={k:np.quantile(estimates[:,j],[.025,.975]).tolist() for j,k in enumerate(keys)}
    result=dict(status='passed',comparison=comparison,source_files=sources,online=online,scenes=per_scene,parity=parity,
        paired_scene_bootstrap=dict(scenes=len(names),samples=10000,seed=2,ci95=intervals),
        limitation='One fixed model and seed on the development set; 11 scene clusters, intervals do not correct head selection.')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    labels={'success':'路线成功率SR','spl':'路径效率SPL','ndtw':'nDTW','sdtw':'SDTW','distance_to_goal':'终点距离（米）','path_length':'路径长度（米）','steps_taken':'步数','collisions':'碰撞率'}
    lines=['# E24 在线导航对照结果','', f'两组均完整覆盖R2R val_unseen的1839条路线；同机、8环境、相同种子和{base_step}步基座。E24使用在6400基座数据上训练的4750步头×1.5，修正裁剪到±1；本次没有重训评分头。','',
           '|指标|一阶段|一阶段+E24|差值|按场景配对95%区间|','|---|---:|---:|---:|---|']
    for k in keys:
        m=comparison['metrics'][k];scale=100 if k in ('success','spl','ndtw','sdtw') else 1
        ci=[round(v*scale,4) for v in intervals[k]]
        lines.append(f"|{labels[k]}{'（%/百分点）' if scale==100 else ''}|{m['base']*scale:.4f}|{m['e24']*scale:.4f}|{m['difference']*scale:+.4f}|{ci}|")
    t=comparison['success_transitions'];lines+=['',f"基线失败→E24成功：{t['fixed']}条；基线成功→E24失败：{t['harmed']}条；净增加成功：{t['fixed']-t['harmed']:+d}条。",'',
        f"E24实际改变移动选择{online['counts']['action_flips']}次，共{online['counts']['rows']}个决策。参数与世界模型冻结审计通过。",'',
        '## 实际导航耗时','']
    for name,label in [('full_base','一阶段'),('full_best','一阶段+E24')]:
        t=comparison['timing'][name];lines.append(f"- {label}：{t['elapsed_seconds']/60:.2f}分钟，平均{t['seconds_per_episode']:.3f}秒/路线。")
    lines+=['',f"零倍率门控：动作与导航指标完全一致；最大分数差{parity.get('logits_max_abs_error',0):.8f}，独立重复基线容差{parity.get('logits_tolerance',0):.8f}。",'',
        '耗时包含首次预测加载/编译和真实模拟器运行，不是纯评分头稳态延迟。','',
        '区间按11个场景成组配对重采样；场景数量有限，且开发集已参与离线选点，不能作为独立测试或多随机种子结论。', '',
        '每场景差值、逐路线接管/未来有效性诊断、运行配置与文件校验值保存在summary.json及远端原始产物中。']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status='passed',output=str(out))))


if __name__=='__main__':main()
