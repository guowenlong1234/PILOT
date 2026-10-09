# Progressive 测评机真实环境验收

2026-10-09 16:32—16:39（北京时间）。用户授权使用测评机计算资源后，在 RTX 3090、`gwl-etpr1-rae`、`etpr1_rae` 中完成回归、真实小采集、GPU 短训练/恢复及成对导航。没有运行正式长训练或全量导航。

## 代码与环境

- 工作目录：`/home/a6000/gwl/ETP-R1`；身份 `a6000/a6000`，连接经 `server` 跳板。
- 开始时工程干净、GPU 显存0、ETPNav容器已停止；没有停止或修改任何受保护任务。
- 原两端基点 `255f6de`；实现提交 `40f290f`；实际测试/导航修复提交 `e22f084`；独立审计脚本提交 `4b1b1cb`。
- 为遵守此前“不自动推送”的要求，仅提交本次代码，通过 Git bundle + fetch + fast-forward 交付；没有推送中央仓库，也没有切换分支。笔记本原有文档修改保留。
- Python 3.11.15、PyTorch 2.2.2+cu121、CUDA runtime 12.1、Transformers 4.49.0、Habitat/Habitat-Sim 0.3.3。真实导入结果保存为 `runtime_versions.json`。离线 CLI 的 metadata 查询显示 Habitat 未作为 distribution 安装，是项目自有运行时部署方式所致，不代表模块无法导入。

所有产物保留在测评机：

```
/home/a6000/gwl/ETP-R1/data/logs/progressive_acceptance_20261009/
```

未把数据或 checkpoint 复制回笔记本。

## 发现并修复的问题

正常运行环境的第一轮为 **16 failed、126 passed**。PyTorch 2.2 对 `lp[present,None]` 的布尔组合索引行为与笔记本 PyTorch 2.11 不一致。改为明确的 `lp[present].unsqueeze(-1)`，保持数学与 mask 语义不变。

同一组测试修复后 **142 passed，3条依赖警告，退出0，7.18秒**。此次使用正常模块导入路径，没有绕过 Habitat 注册。初次失败日志 `pytest.log` 和修复日志 `pytest_fixed.log` 均保留。

实际命令（测评机宿主机；下同）：

```bash
docker exec gwl-etpr1-rae bash -lc 'bash scripts/rgb_only_optimization_runtime.sh eval -m pytest -q tests/test_progressive* tests/test_stage2* tests/test_stage0_topk_query.py'
```

## 真实采集、GPU更新与恢复

基座使用已有批准资产 `/home/a6000/gwl/ETP-R1-stage2-e24/stage2_assets/ckpt.iter9200.pth`，SHA256为 `87bf7ad691a93abfe2d5030c2c314ef4e630c41d3abbe61fea3b38872686055e`。没有换用其他基座或放宽校验。

```bash
BASE=/home/a6000/gwl/ETP-R1-stage2-e24/stage2_assets/ckpt.iter9200.pth
OUT=data/logs/progressive_acceptance_20261009
python3 scripts/stage2_e24_job.py collect --machine eval --gpu 0 \
  --environments 1 --base-step 9200 --checkpoint "$BASE" --split train \
  --episodes 2 --output "$OUT/collect" --progressive none \
  --gain 1 --margin-threshold -1 --no-compile --trace
```

采集退出0，2条路线/12次决策，第一层28次、第二层20次世界模型查询。新schema完整校验通过。基座、融合、路点与CWP共971个tensor前后完全一致；世界模型314个tensor前后完全一致。不是模拟未来特征。

离线命令统一前缀为：

```bash
RUNNER=(docker exec -w /home/a6000/gwl/ETP-R1 gwl-etpr1-rae \
  bash scripts/rgb_only_optimization_runtime.sh eval scripts/progressive_stage2.py)
"${RUNNER[@]}" validate --data "$OUT/collect/episodes"
"${RUNNER[@]}" train --data "$OUT/collect/episodes" \
  --model-config configs/progressive/model.json --output "$OUT/train" \
  --steps 2 --batch-size 2 --device cuda:0
"${RUNNER[@]}" train --data "$OUT/collect/episodes" \
  --model-config configs/progressive/model.json --output "$OUT/train" \
  --steps 1 --batch-size 2 --device cuda:0 --resume "$OUT/train/state_step_000002.pt"
HEAD="$OUT/train/head_step_000003.pt"
"${RUNNER[@]}" evaluate --data "$OUT/collect/episodes" --checkpoint "$HEAD" \
  --device cuda:0 --output "$OUT/offline_evaluate.json"
for MODE in none certified; do
  "${RUNNER[@]}" replay --data "$OUT/collect/episodes" --checkpoint "$HEAD" \
    --device cuda:0 --pruning-mode "$MODE" --output "$OUT/replay_$MODE.json"
done
```

以上6项全部退出0，准确命令和退出码保存在 `offline_status.json`。训练完成第1、2步，恢复后完成第3步；保存训练状态及推理head。新head约241MB，SHA256为 `f6fe784cbc25da09baf58389aa709a870a4cdd855b1d82162cb2a8d996896530`。两种缓存重放均通过前缀一致性和完整展开argmax核对。这里的离线评价使用采集训练样本，只是接口验收，不作为泛化成绩。

## 同一新模型的成对真实导航

```bash
for MODE in none certified; do
  python3 scripts/stage2_e24_job.py online --machine eval --gpu 0 \
    --environments 1 --base-step 9200 --checkpoint "$BASE" --head "$HEAD" \
    --split val_unseen --episodes 2 --output "$OUT/nav_$MODE" \
    --progressive "$MODE" --gain 1 --margin-threshold -1 \
    --no-compile --trace --profile
done

docker exec gwl-etpr1-rae bash scripts/rgb_only_optimization_runtime.sh eval \
  scripts/audit_progressive_acceptance.py "$OUT" --output "$OUT/navigation_audit.json"
```

两次导航及独立审计全部退出0。两组使用相同head SHA、基座、资产、最大深度和预测契约；2条未见场景路线、各16次决策，**实际执行动作、候选身份/映射、逐路线指标完全一致**。不是仅比较日志中的某个临时赢家。冻结检查全部通过。

| 检查 | 完整展开 none | certified |
|---|---:|---:|
| 实际世界模型查询 | 85 | 41 |
| d=0结束的决策（含STOP） | 2 | 7 |
| d=1结束的决策 | 2 | 5 |
| d=2结束的决策 | 12 | 4 |
| 严格证书提前停止 | 0 | 9 |
| 第二阶段累计计时 | 10.450秒 | 5.072秒 |
| 整次导航作业（含加载） | 46.319秒 | 40.467秒 |

独立审计从停止行的实际FP32分数、剩余预算和数值余量重算了全部9个证书，并与完整展开赢家核对。`navigation_audit.json`保存详细统计。

查询少44次（约51.8%）。这里只跑了小样本和一个仅更新3步的head，计时是各一次且包含预热等因素；不能当作正式性能基准、训练收敛结果或导航收益证据。尚未运行完整训练集采集、正式训练、全量未见场景评测、重复测速及多环境真实吞吐对照。

16:38:54最终检查：GPU显存0、无计算进程；专用容器中无本次训练/导航残留。数据、日志、初次失败记录及所有新checkpoint保留。
