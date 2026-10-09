# 全量R2R未见场景推理与逐步分数归档

## 启动核查

北京时间2026-10-09 11:09:46，16路线预检通过后进入全量。预检记录125次决策、1386个候选分数，审计退出0；与上一轮未剪枝16路线结果比较，125次动作和逐路线指标完全一致。全量运行源码`8ff2b62`，provenance已确认1839个episode、关闭剪枝、正确评分头SHA。监督PID36539，当前全量启动器PID37142；核查时已写入16条v2决策记录，GPU100%负载，日志持续增长。

任务后台运行，**尚未完成全量评测**。基于此前128路线约519秒的速度粗估约2小时，实际以进度为准。结束后的完整性检查和统计已自动串联，不需要手动再跑导航。

## 任务与协议

2026-10-09，用户要求重新运行全量未见环境推理并记录每一次分数，供后续离线统计。沿用前一轮部署：持久9200导航基座＋原生9200 full评分头2000步、倍率1、最终±1裁剪。测评机RTX3090、4个环境、专用容器`gwl-etpr1-rae`及`etpr1_rae`环境；不使用训练机GPU。

数据为R2R `val_unseen` 全部1839路线、11场景。**关闭剪枝**，保留完整前瞻评分，同时记录每步的可剪枝证书。这样可以在同一轨迹上统计前瞻修正、动作翻转及可跳过数量，不会因为提前剪枝而缺失该步的完整评分。

输出根（测评机）：

```text
/home/a6000/gwl/ETP-R1/data/logs/full_score_archive_20261009/run_v1/
```

- `pipeline.json`：监督状态；只有全量推理和记录审计均通过才写completed。
- `preflight/`：16路线记录验收；未通过则不启动全量。
- `full/online/decisions.jsonl`：全量逐步分数，每行一次高层决策，每步flush。
- `full/provenance.json`、`launch.json`、`status.json`、`run.log`：路线名单、源码/模型哈希、精确命令、退出码、环境版本和运行日志。
- `full/online/progress.json`：完成路线数与查询/决策数量。
- `full/results/`：逐路线导航指标。
- `full/score_trace_audit.json`：完成后的完整性审计和离线计数，包括分差阈值累计数量、动作改选、证书通过数、场景统计及trace SHA256。
- 外层 `data/logs/full_score_archive_20261009/supervisor.log`：后台监督日志；`preflight.audit.log`、`full.audit.log`保留审计输出。

## 记录内容

格式为`stage2-decision-scores-v2`，保留旧字段并增加：

| 字段 | 含义 |
|---|---|
| `scene / episode / step` | 场景、路线、导航步唯一身份 |
| `global_vp_ids / ghost_indices` | 分数槽位对应的节点ID，以及可执行移动候选索引 |
| `topk_global_indices` | 实际选取的前五候选，稳定并列排序 |
| `logits / delta / refined_logits` | 基础分数、实际应用修正、实际浮点加法后的最终分数 |
| `base_action / action / executed_action` | 基础选择、二阶段选择、考虑强制停止后的高层执行动作 |
| `forced_stop` | 达到最大步数或无剩余路点时强制停止 |
| `future_valid_mask / invalid_reason` | 各TopK候选是否有有效未来，以及无效原因 |
| `effective_residual_bound / certificate / skip_reason` | 实际修正边界、不变性判断和实际跳过原因；本轮关闭剪枝，实际skip_reason为null |

STOP及已访问节点也保留分数槽位。不可执行槽位可能为`-Infinity`，读取时使用支持该既有日志格式的Python JSON解析；统计移动排序需筛选ghost_indices。STOP采用基础策略隔离规则，不能对所有refined_logits直接argmax当作最终动作。

本归档记录当前Top5、当前深度、当前模型的输出；可用于这些轨迹上的分数分布、间隔、翻转和证书数量统计。改变权重、候选预算、深度或动作策略可能产生新轨迹，不能仅靠已有分数声称完成新策略的导航评测。

## 实施与验收

日志补充与监督入口源码`96600bc`；累计阈值统计修正为`8ff2b62`。没有修改模型评分计算、权重或动作规则，只增加trace字段；测试在测评机完成，27 passed、退出0（3.24秒），日志位于`data/logs/full_score_archive_20261009/tests/run.log`，包含实际环境版本。

入口 `scripts/run_stage2_full_score_archive.py` 自动执行16路线→审计→1839路线→审计。宿主机使用薄nohup托管，不改真实推理命令。审计 `scripts/report_stage2_score_trace.py` 检查：完整路线覆盖、无缺步/重复、每条路线决策数等于high_level_step+1、候选身份和TopK、基础/最终选择、分数回加、残差边界、未来有效数，以及模型冻结。

原始数据保留测评机，不自动复制大日志到笔记本。当前运行状态以远端pipeline和progress为准，上述PID仅为启动时现场，不应据此操作未来复用的同号进程。
