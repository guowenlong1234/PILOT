# 结果占位符清单

本文中的斜体大写字段是结果键，不是推测值。任何键只有在以下元数据完整时才允许替换为数值：`dataset`、`split`、`languages`、`controller`、`sensors`、`success_threshold`、`max_steps`、`checkpoint_path`、`checkpoint_sha256`、`selection_rule`、`train_seed`、`eval_seed`、`code_commit`、`config_sha256`、`episode_count`、`raw_json`、`summary_statistic` 和 `confidence_interval_method`。

## 协议与环境键

| 键 | 含义 | 当前状态 |
|---|---|---|
| `DIRECT-R2R-PROTOCOL` | R2R-CE 直接比较的完整协议与原始结果索引 | 待同口径实验完成 |
| `DIRECT-RXR-PROTOCOL` | RxR-CE 直接比较的完整协议与原始结果索引 | 待同口径实验完成 |
| `RXR-EVAL-MANIFEST`、`RXR-SPLIT/LANGUAGES` | RxR 评测清单、划分、语言和 episode 数 | 待实际 manifest 固定 |
| `CKPT-SELECTION-RULE`、`N-SEEDS` | 预先登记的 checkpoint 选择函数与训练种子数 | 实验启动前固定 |
| `NWM-CKPT-SHA256` | RAE-NWM 权重哈希 | 从正式配置和启动日志读取 |
| `SFT-LR/BATCH/STEPS/SEEDS`、`GRPO-STEPS/SEEDS` | 训练预算与随机种子集合 | 按 R2R/RxR 配置分别记录 |
| `DATA-MANIFEST/...` | 数据、控制器、传感器、最大步数和成功阈值 | 从评测配置自动导出 |
| `PYTHON/.../GPU`、`TIMING-GPU` | 软件版本与计时硬件 | 从实际运行环境记录 |
| `ORACLE-SNAP-TOLERANCE` | 候选级真实渲染时允许的导航网格吸附距离 | oracle 实现前固定 |
| `DIAG-1200/2000-COUNTS` | 两个一步诊断点的请求、候选槽和决策行原始计数 | 从对应诊断 JSON 读取 |

## 主结果键

| 键模式 | 数据集/阶段 | 指标 | 原始来源要求 |
|---|---|---|---|
| `R2R-BASE-*` | R2R-CE RAE/DINOv2 SFT 基座 | NE、OSR；SR=63.73、SPL=55.61 已核验 | 完整 1,839 episode JSON |
| `R2R-LA-SFT-*` | R2R-CE 主动前视 SFT | NE、OSR、SR、SPL | 同协议完整 JSON 与种子聚合 |
| `R2R-NOLA-GRPO-*` | R2R-CE 相同强化预算的零残差 GRPO | NE、OSR、SR、SPL | 同起点、同预算完整 JSON |
| `R2R-LA-GRPO-*` | R2R-CE 冻结前视 GRPO | NE、OSR、SR、SPL | 同协议完整 JSON 与冻结审计 |
| `RXR-BASE-*`、`RXR-LA-SFT-*` | RxR-CE SFT 基座与主动前视 | SR、SPL、nDTW、SDTW | 完整语言设置与 manifest |
| `RXR-NOLA-GRPO-*`、`RXR-LA-GRPO-*` | RxR-CE 零残差与冻结前视 GRPO | SR、SPL、nDTW、SDTW | 同起点、同预算完整 JSON |

## 消融与机制键

| 键模式 | 对应分析 | 统计单位与约束 |
|---|---|---|
| `A1-R2R-{BASE,RAE,RGB,LA,GRPO}-{SR,SPL}` | R2R-CE 逐级消融 | 每行只改变声明因素；同控制器、预算与选择规则 |
| `A1-RXR-{BASE,RAE,RGB,LA,GRPO}-{SR,SPL,SDTW}` | RxR-CE 逐级消融 | 同上，并固定语言集合 |
| `A2-NOQ1-{SR,SPL}` | 参数量匹配、零未来标记的无 $q_1$ 对照 | 独立同预算训练，保留候选掩码 |
| `A2-ORACLE-{SR,SPL}` | 固定预测头的候选级真实渲染诊断 | 所有前五候选；克隆/恢复状态；报告覆盖率 |
| `A2-H{1,2,3}-{SR,SPL}` | 预测视野导航指标 | 固定历史、累计位姿、端点输入；各视野独立训练 |
| `A2-H{1,2,3}-{CLSERR,PATCHERR,VALID,MS,GB}` | 视野预测质量与成本 | 适配器前误差；毫秒/高层决策；峰值 GiB |
| `A3-{FIX,HARM,EFF,FAIL,SAME}-N` | episode 级类别计数 | 相同起点和 seed；以首次分歧归类 |
| `A3-{FIX,HARM,EFF,FAIL,SAME}-ALL` | 占全部 episode 比例 | 分母为完整配对 episode 数 |
| `A3-{FIX,HARM,EFF,FAIL}-DIV` | 占发生分歧 episode 比例 | 分母为至少一次动作分歧的 episode 数 |
| `A3-{FIX,HARM,EFF,FAIL,SAME}-DSPL` | 配对 SPL 差 | 报告场景聚类 bootstrap 95% 置信区间 |

## 摘要与结论键

`RESULT-SUMMARY` 只能由同协议主结果、逐级消融、$q_1$ 来源、$h=1,2,3$、翻转净价值和开销中已经完成的证据组成。`CONCLUSION-EVIDENCE` 必须与结果表保持相同主张强度；若多步、RxR 或 GRPO 中任一项未完成，则不在结论中写成既成结果。

