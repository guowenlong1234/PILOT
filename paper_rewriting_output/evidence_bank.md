# 证据库

## 证据等级

- E1：完整数据划分上的原始评测 JSON，且配置、checkpoint 和指标口径可追溯。
- E2：真实模型/数据上的数值一致性、冻结检查、恢复检查或完整回归测试。
- E3：源代码、配置和设计文档证明系统具备某项机制。
- E4：正在运行的中间结果，只能描述截点状态。
- E5：计划实验，只能形成问题、表格和占位符。

| Evidence ID | 证据等级 | 来源 | 可支持内容 | 不能支持内容 | 稿件用途 |
|---|---|---|---|---|---|
| E01 | E1 | 原版 ETP-R1 R2R/RxR 发布 checkpoint 的完整 val_unseen 结果；汇总见 research.md 与 assets/result.png | 旧 ETP-R1 的联合预训练—SFT—GRPO 训练范式及对应阶段结果 | 当前 RAE/NWM/E24 扩展已经达到相同或更高性能 | 背景基线、实验比较行 |
| E02 | E1 | 现代 RAE/DINOv2 第二轮 R2R SFT iter14200，完整 1,839 episode 评测 | 当前主动前视实验的固定基础 checkpoint；SR 0.6373028820、SPL 0.5560538836 | 多种子稳定性或独立测试集泛化 | 主结果基座、消融起点 |
| E03 | E1 | 当前 RAE/DINOv2 GRPO iter750 完整评测 | 在现有预算下 GRPO 相对 SFT 的 SR/SPL 变化和停止/效率问题 | 新主动前视冻结式 GRPO 已有效 | 训练范式背景、局限性 |
| E04 | E4 | native_cls_e24_joint_sft 第 200–2400 次完整评测 | 正式训练早期 checkpoint 的实际趋势；截至截点未稳定超过 E02 | 最终最优结果、收敛结论 | 初稿结果占位说明，不进入摘要强结论 |
| E05 | E2 | 原生 257-token NWM parity、EMA 314/314 严格加载、CLS/完整序列余弦一致性 | ETP-R1 兼容层忠实执行指定 RAE-NWM 原生 CLS+patch 推理 | 未来预测对真实导航有帮助 | 方法正确性、复现性 |
| E06 | E2 | q0/CWP/q1 单 episode、batch/row failure、shape 与非有限值检查 | 一步预测链路实际运行且具备安全回退 | 二步/三步预测已经实现 | 方法与运行时安全 |
| E07 | E2 | tests/test_stage0_e24_joint.py、test_grpo_frozen_lookahead.py 等 | Top-5 adapter 只修改 CLS；STOP/非 Top-5 保持；梯度和冻结边界符合设计 | 模块带来最终性能收益 | 方法契约、强化阶段冻结说明 |
| E08 | E3 | vlnce_baselines/nwm/predictor.py、active_lookahead/dino_cwp_future.py | 历史上下文、NWM 条件、原生 token 拆分及 q1 构造的数据流 | 数值准确性和导航收益 | 方法公式和伪代码 |
| E09 | E3 | active_lookahead/native_cls_adapter.py、joint_e24.py、SFT trainer replay | 独立 CLS adapter、E24 有界残差、完整 logits 交叉熵和延迟回放 | 相比其他设计最优 | 方法细节与消融设计 |
| E10 | E4 | lookahead_ckpt_1200/2000_val_unseen.json | q0/q1 成功率 1.0、future-valid 约 0.466/0.496、动作翻转率约 0.00286/0.00171、推理耗时 | 极低动作翻转必然带来正收益 | 机制诊断、实验 3/6 的设计依据 |
| E11 | E5 | 用户确认的逐级模块实验 | Base→RAE/DINOv2→RGB-NWM→E24→GRPO 的边际贡献 | 当前已完成公平单变量对照 | 表 2 占位、结果单元 R1 |
| E12 | E5 | 用户确认的未来证据实验 | 无 q1、oracle q1、predicted q1 的上界与误差来源 | oracle 或多步结果已存在 | 表 3 占位、结果单元 R2 |
| E13 | E5 | 用户确认的 1/2/3 步预测视野实验 | 更远未来信息、误差累积和计算成本的关系 | 三步优于一步 | 图 3/表 4 占位、结果单元 R3 |
| E14 | E5 | 用户确认的动作翻转配对分析 | 主动前视对相同 episode 的修正、伤害与无变化类型 | 翻转具有因果净收益 | 图 4/表 5 占位、结果单元 R4 |
| E15 | E5 | 用户确认的预测误差/覆盖/Top-K/上下文联合分析 | 预测质量与决策收益的条件关系 | 单一相关性证明因果 | 图 5/表 6 占位、结果单元 R5 |
| E16 | E5 | 计划中的 R2R/RxR 主实验与冻结式 GRPO | 跨数据集和强化阶段验证 | 当前已经完成 | 主结果表占位 |

## 当前可直接写入的事实数字

1. RAE/DINOv2 R2R SFT 基座 iter14200：完整 val_unseen SR 63.7303%，SPL 55.6054%。
2. 主动前视中间 checkpoint：第 2000 次 SR 63.2953%、SPL 53.5741%；第 1200 次 SR 62.4796%、SPL 55.1929%。这些数值仅描述截点，不作为最终方法结论。
3. 一步前视诊断：q0/q1 请求成功率为 100%；有效未来候选率约 46.6%–49.6%；动作翻转率约 0.17%–0.29%。

## 回填规则

所有计划实验必须记录数据划分、控制器、checkpoint、训练预算、随机种子、选择规则和原始结果路径。若二步或三步预测尚未实现，正文只说明它们属于视野长度实验设计；只有运行日志和评测 JSON 产生后才回填数值。

