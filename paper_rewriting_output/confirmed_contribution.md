# Confirmed Contribution

## Core Contribution

| Field | Content |
|---|---|
| Main contribution statement | 本文提出一种面向在线拓扑候选的受约束表征世界模型主动前视框架：冻结的原生 CLS+patch 世界模型在不查询未来模拟器观测的条件下预测候选相关未来表征，候选残差评分头对前五个 ghost 施加有界重排；监督阶段学习前视模块，强化阶段复用 rollout 缓存的动作对齐残差与掩码，只更新基础导航策略。 |
| Contribution type | new method，辅以 new empirical analysis；主类型为新的拓扑候选主动前视方法。 |
| One-sentence reviewer payoff | 审稿人应看到一种可审计的方式，把高维未来表征引入图式 VLN-CE 决策，同时用候选范围、残差幅度和梯度冻结限制世界模型误差对基础策略的破坏。 |

## Why This Contribution Is Needed

| Field | Content |
|---|---|
| Field problem | 连续环境视觉语言导航要求智能体把长程语言目标转化为闭环移动，而在线拓扑图虽然降低了规划难度，候选选择仍容易受当前观测不充分、遮挡和状态分布偏移影响。 |
| Specific gap | 现有图式 VLN-CE 缺少一种与 ghost 候选动作空间直接对齐、无需访问真实未来观测、还能跨 SFT 与 GRPO 保持明确训练边界的原生未来表征注入方式。 |
| Concrete challenge | 原生 CLS+patch 预测位于高维语义潜在空间，历史上下文可能不足，q1 候选可能无效，多步递推会累积误差；若未来信号直接覆盖基础动作分数，少量错误预测就可能改变停止与其他候选决策。冻结参数也不自动保证 GRPO 的当前、旧和参考策略使用相同残差，因此必须固定候选选择和残差输入语义。 |
| Why prior work leaves it unresolved | ETPNav/ETP-R1 建立了拓扑规划与分阶段训练，但不具备本文的原生未来表征候选重排；NavMorph 表明世界模型可用于 VLN-CE，却没有解决本文的在线 ghost 对齐、Top-5 有界残差和 SFT 学习/GRPO 冻结契约；NavGRPO 研究 VLN 的组相对优化，但不提供本文的冻结主动前视模块。 |

## How This Paper Responds

| Field | Content |
|---|---|
| Design response | 系统保留基础导航分数，使用四帧历史上下文驱动冻结 RAE-NWM 预测原生 CLS+256 patch 序列；DINO 候选航点预测器构造候选未来条件，独立 CLS 适配器与候选残差评分头仅重排前五个移动候选，停止和其他候选不直接加残差。GRPO 在 rollout 时缓存完整动作对齐残差、有效掩码和旧概率，更新时不重新选择前五候选。当前一步预测是主方法，第二、第三步明确作为待实现的受控视野扩展。 |
| Evidence required | 需要完整 R2R 与 RxR 主结果；Base→RAE/DINOv2→RGB-NWM→E24→GRPO 逐级消融；无 q1、oracle q1、predicted q1 和 1/2/3 步预测对照；动作翻转修正/伤害统计；预测误差、future-valid、Top-K、上下文长度、延迟与导航指标的联合分析；公平控制器、预算、选点规则和不确定性说明。 |
| Evidence available | 已有原生 257-token NWM 严格 EMA 加载与数值一致性；q0/CWP/q1 推理成功；DINOv2、NWM、CWP 冻结与梯度边界通过测试；联合 SFT checkpoint 保存/恢复和完整 R2R val_unseen 逐点评测已运行；旧 ETP-R1 的 R2R/RxR SFT→GRPO 结果可作为训练范式背景；当前一步预测的覆盖率、动作翻转率和推理耗时已有诊断记录。 |
| Evidence missing | 当前联合 SFT 尚无稳定超过基座的结果；二步和三步预测、oracle q1、严格模块消融、动作翻转因果分组、完整效率曲线、新主动前视 GRPO、RxR 主实验和多训练种子仍需完成。缺失证据意味着当前稿件只能把性能提升写成待验证假设，不能写成已证实结论。 |

## Claim Boundary

| Field | Content |
|---|---|
| Strong claims allowed | 可以声称本文实现了原生 CLS+patch 的无未来观测一步预测、拓扑 ghost 对齐、前五候选有界重排、候选级回退和 GRPO 缓存残差契约，并建立了覆盖第一至第三步视野、预测质量和动作干预的验证协议。第二、第三步只能称为计划扩展，直至代码、测试与结果齐备。 |
| Claims to soften or avoid | 在最终主结果、消融和不确定性证据完成前，避免声称主动前视稳定提升导航、普遍优于基线、达到新 SOTA、显著增强泛化，或首次把世界模型/GRPO 用于 VLN。 |
| Novelty risk | 最强反对意见是 NavMorph 已将世界模型用于 VLN-CE，NavGRPO 已将组相对强化学习用于 VLN。本文的回应只能依靠更窄的技术组合与证据：原生 CLS+patch 预测到在线 ghost 的直接对齐、Top-5 有界残差、SFT 学习/GRPO 冻结契约以及多视野长度机制分析。 |
| Significance risk | 最强反对意见是模块只改变极少数动作、增加大量推理开销且未带来稳定指标收益。论文必须用翻转修正/伤害、oracle 上界、1/2/3 步误差累积、性能—开销曲线和完整消融回答；若净收益不足，应把结论收缩为机制边界而非强性能贡献。 |
