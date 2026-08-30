# 论文材料来源图谱

## 使用原则

本图谱记录中文逻辑母稿可以使用的工程证据及其边界。代码和配置可以证明“系统实现了什么”，但不能单独证明“方法有效”；完整评测 JSON 可以证明对应 checkpoint 在指定数据划分上的观测结果，但单次运行不能证明统计稳定性。尚在运行或尚未启动的实验统一标为“待回填”，不得在正文中写成已完成事实。

## 核心来源

| 来源编号 | 来源与路径 | 来源类型 | 可支持的论文内容 | 证据状态 | 使用边界 |
|---|---|---|---|---|---|
| S01 | `README.md` | 原版 ETP-R1 项目说明 | 原版三阶段框架、Gemini 增强数据、R2R/RxR 联合预训练、在线 SFT 和 GRPO | 已核验 | 只代表原版 ETP-R1 的公开设计与公开结果，不代表当前 RAE/NWM 扩展结果 |
| S02 | `assets/Overview.png` | 原版方法图 | 原版“联合预训练—在线 SFT—在线 RFT”总体流程 | 已核验 | 新论文若以主动前视为核心，需要重画，不能把旧图当作新方法全图 |
| S03 | `assets/result.png` | 原版结果表 | 原版 ETP-R1 在 R2R-CE/RxR-CE 的发布指标与对比对象 | 已核验 | 可作为原版强基线；不能用来代替当前分支的新实验 |
| S04 | `docs/ETP-R1_vs_ETPNav_diff_analysis.md` | 代码对比分析 | ETP-R1 相对 ETPNav 的训练范式、任务编码、图文融合与联合预训练差异 | 已核验 | 适合背景与系统沿革，不足以证明当前主动前视模块的新颖性 |
| S05 | `docs/superpowers/specs/2026-07-10-rae-dinov2-visual-encoder-design.md` | 技术设计 | 用 RAE/DINOv2 替换旧 CLIP 表征、特征契约和隔离运行环境 | 已核验 | 设计意图需要与最终代码和数值一致性测试共同引用 |
| S06 | `docs/rae-dinov2-eval-host-validation.md` | 环境/数值验证 | 线上编码器、离线特征与运行环境的一致性和可复现性 | 已核验 | 属于实现正确性证据，不直接证明导航性能收益 |
| S07 | `docs/rae-dinov2-pretrain-navigation-investigation-20260809.md` | 诊断报告 | 视觉接口迁移、预训练与在线导航衔接中的问题、修复依据和风险 | 已核验 | 诊断性结论需与最终实验区分；不宜写成方法贡献本身 |
| S08 | `docs/rae-dinov2-pretrain-operations.md` | 运行记录 | 联合预训练配置、恢复逻辑、checkpoint 与环境版本 | 已核验 | 用于复现细节，不承担效果结论 |
| S09 | `docs/rae-dinov2-r2r-sft-operations.md` | 运行记录 | R2R SFT 配置、评测流程和 checkpoint 同步机制 | 已核验 | 只支持实际执行过的配置 |
| S10 | `docs/rae-dinov2-r2r-grpo-server-operations.md` | 运行记录 | R2R GRPO 的起点、训练预算、评测与恢复约束 | 已核验 | 不应把工程托管脚本包装成算法创新 |
| S11 | `docs/ETP-R1_checkpoint_eval_comparison.md` | 结果汇总 | 原版 checkpoint 与当前 RAE/DINOv2 checkpoint 的同口径比较 | 已核验 | 需回到原始 JSON 交叉核对最终选点和数据划分 |
| S12 | `research.md` | 工程研究地图 | 已完成实验、远端产物路径、环境、失败诊断和运行历史 | 已核验并持续更新 | 是导航索引而非独立实验原始记录；关键数字要用原始结果复核 |
| S13 | `vlnce_baselines/models/encoders/rae_dinov2_encoder.py` 及相关编码器实现 | 源代码 | DINOv2/RAE 视觉特征的在线接口、维度与归一化路径 | 已核验 | 仅证明实现；性能因果需消融 |
| S14 | `vlnce_baselines/nwm/predictor.py` | 源代码 | 从导航上下文构造 NWM 条件并预测未来潜变量/CLS | 已核验 | 不能仅凭代码宣称预测准确或导航有效 |
| S15 | `vlnce_baselines/nwm/active_lookahead/dino_cwp_future.py` | 源代码 | DINO-CWP 候选生成、q1 未来 token 构造、故障回退与诊断量 | 已核验 | 需要报告候选覆盖、失败率、推理耗时和动作影响率 |
| S16 | `vlnce_baselines/nwm/active_lookahead/e24_joint.py` 及邻近模块 | 源代码 | E24 评分头、联合损失、回放微批次和增量动作评分 | 已核验 | 需要用“无 E24/无未来预测/不同 top-k”等对照解释收益来源 |
| S17 | `vlnce_baselines/ss_trainer_ETP_R1.py` | 源代码 | 主动前视在在线 SFT rollout、联合训练、冻结检查和日志中的完整接入 | 已核验 | 代码复杂度较高，正文需抽象成清晰的数据流而非实现细节堆积 |
| S18 | `vlnce_baselines/GRPO_trainer_ETP_R1.py` | 源代码 | 图拓扑策略的在线 GRPO、组内相对优势、KL 约束和闭环采样 | 已核验 | 当前新分支 GRPO 的最终效果尚未完成，不得提前写成提升 |
| S19 | `run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml` | 配置 | 当前 R2R 原生 CLS + E24 联合 SFT 的正式参数 | 已核验 | 论文表格需记录最终实际运行参数，而非默认值 |
| S20 | `run_r2r/grpo_native_cls_e24_frozen.yaml` | 配置 | R2R 主动前视 GRPO 的冻结和训练边界 | 已核验、实验待完成 | 只能用于方法与计划实验描述，结果保留占位符 |
| S21 | `docs/e24-joint-sft-operations.md` | 运行方案 | E24 联合 SFT 的启动、检查点、同步和评测约束 | 已核验 | 属于复现材料，不等于结果证据 |
| S22 | `docs/native-cls-e24-grpo-operations.md` | 运行方案 | 原生 CLS + E24 冻结 GRPO 的计划流程 | 已核验、实验待完成 | 正文结果必须待实际运行后回填 |
| S23 | 训练机 `data/logs/.../rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft` | 原始 checkpoint/日志 | 两轮 R2R SFT 的收敛速度与最终最佳结果 | 已完成 | 现有证据支持样本效率改善；最终上限差异需随机种子或更谨慎措辞 |
| S24 | 测评机对应两轮 R2R SFT `eval_watch_val_unseen/.../eval_results/` | 完整评测 JSON | 1,839 个 R2R `val_unseen` episode 的 SR、SPL、nDTW、SDTW 等 | 已完成 | checkpoint 从同一验证集多次选择会产生选择偏差，需独立确认或说明 |
| S25 | 测评机当前 R2R GRPO `eval_results/` | 完整评测 JSON | 当前 RAE/DINOv2 GRPO 与 SFT 起点、官方 GRPO 的同口径比较 | 已完成 | 当前最佳 GRPO 未超过官方 GRPO，不能写成全面领先 |
| S26 | 训练机 `data/logs/active_lookahead/native_cls_e24_joint_sft/...` | 运行日志/checkpoint | 当前主动前视联合 SFT 的损失、梯度、覆盖、动作翻转和训练进度 | 运行中 | 2026-08-28 只观察到第 2800 次左右，不代表最终收敛 |
| S27 | 测评机 `data/logs/active_lookahead/native_cls_e24_joint_eval/.../stats_ckpt_*_val_unseen.json` | 完整评测 JSON | 主动前视联合 SFT 第 200–2400 次 checkpoint 的完整 R2R 指标 | 阶段性完成 | 截至核验时尚未超过基座；后续结果需继续回填 |
| S28 | 测评机同目录 `lookahead_ckpt_*_val_unseen.json` | 机制诊断 JSON | q0/q1 成功率、未来候选覆盖、CWP 空预测率、动作翻转率与耗时 | 阶段性完成 | 可证明链路稳定和实际干预强度，但不能自动推出性能提升 |
| S29 | 计划中的 R2R 主动前视 GRPO | 待运行实验 | 强化阶段是否能把未来预测转化为成功率和路径效率收益 | 待回填 | 所有表格单元格用 `[待填]`，正文使用预期/研究问题措辞 |
| S30 | 计划中的 RxR SFT 与 GRPO | 待运行实验 | 多语言长指令场景下的迁移性和联合训练价值 | 待回填 | 至少报告 `val_unseen` 的 NE、SR、SPL、nDTW、SDTW，并与原版 ETP-R1 同口径比较 |

## 已核验的阶段性数字

- 原生兼容 RAE/DINOv2 第二轮 R2R SFT 最佳 checkpoint 为第 14200 次：SR `0.6373028820`，SPL `0.5560538836`。这些数字来自 1,839 个 `val_unseen` episode 的完整评测汇总。
- 当前 RAE/DINOv2 GRPO 最佳 checkpoint 为第 750 次：SR `0.6411092985`，SPL `0.5542831948`。相对 SFT 起点，SR 略升而 SPL 略降；相对原版官方 GRPO 的 SR `0.6541598439`、SPL `0.5586563945`，尚未领先。
- 主动前视联合 SFT 的阶段性评测截至第 2400 次。已观测 checkpoint 中最高 SR 为第 2000 次的 `0.6329526917`，最高 SPL 为第 1200 次的 `0.5519294549`，均不足以支持超过第 14200 次基座的结论。
- 主动前视评测的 q0/q1 NWM 请求成功率为 `1.0`；未来候选有效率在已抽查的第 1200/2000 次约为 `0.466/0.496`；动作翻转率约为 `0.00286/0.00171`。这些数字说明系统稳定运行，但干预十分稀疏。

## 待回填实验槽位

1. R2R：基座 SFT、主动前视联合 SFT、各自的 GRPO，以及与原版 ETP-R1/ETPNav 的同口径比较。
2. RxR：基座 SFT、主动前视联合 SFT、GRPO 和完整多语言 `val_unseen` 指标。
3. 模块消融：CLIP 与 DINOv2/RAE、无 NWM、有 q0 无 q1、oracle q1、预测 q1、无 E24、不同 top-k、不同动作增量尺度。
4. 稳定性：至少三个随机种子，报告均值与标准差；如成本过高，至少对核心配置重复运行并明确局限。
5. 效率：参数量、训练显存、每步耗时、每 episode 推理时间，以及 q0/q1 NWM 占用。
6. 机制：未来预测质量、候选覆盖率、动作翻转率、翻转前后成功/失败分组，以及“修正/伤害”统计。

