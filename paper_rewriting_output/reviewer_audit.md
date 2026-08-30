# Reviewer-Aware Audit

## 1. Reviewer Value Map

| Reviewer criterion | What reviewers/editors want | Our manuscript evidence | Current weakness | Revision action |
|---|---|---|---|---|
| Novelty | 相对最接近工作有一句可引用、可验证的差异，而不是模块拼接。 | 引言收窄到原生 CLS+patch 与在线 ghost 对齐；相关工作含 NavMorph 功能比较表；基础 ETP-R1 与新增内容已分开。 | 新颖性仍依赖主动前视真正影响决策，而现有翻转率很低。 | 完成候选级真实渲染、参数匹配无未来和翻转净价值分析；若收益不足，收缩主张。 |
| Significance | 方法对导航决策的影响不仅存在，而且相对成本具有实际意义。 | 前五候选有界残差、候选级回退、误差/覆盖/成本联合分析设计。 | 缺少完整主结果、净修正率和性能--成本曲线。 | 完成 R2R/RxR SFT/GRPO、动作翻转和延迟/显存实验。 |
| Technical soundness | 概率语义、时间索引、梯度路径、oracle 和失败处理均无隐含假设。 | $q_0/q_1/h$ 方程；SFT 更新矩阵；GRPO rollout 缓存残差契约；候选级真实渲染协议。 | h=2/3 与真实渲染仍是协议，尚无实现与日志。 | 实现后增加逐视野 smoke、缓存 residual/mask/Top-K 一致性和模拟器状态恢复测试。 |
| Evidence sufficiency | 主结果、最近基线、单变量消融、机制诊断和不确定性足以排除常见替代解释。 | 同协议主表与文献背景表分离；逐级消融；无未来/真实渲染/预测；翻转与机制分析。 | 大多数结果仍是具名键，且尚无多种子统计。 | 按 `results_placeholder_manifest.md` 生成所有原始 JSON 与清单后再回填。 |
| Clarity | 读者一次即可理解基础方法、新增模块、预测步数和结果边界。 | 方法图、机制图、NavMorph 表、参数更新表、变体协议表和候选级定义。 | 个别表在中文 Word 中可能过宽；结果键会增加视觉密度。 | Word/PDF 全页渲染检查；真实数值回填后再次压缩表头和段落。 |
| Venue fit | 贡献强度、篇幅、图表与目标期刊读者和模板一致。 | 当前采用通用 IEEE 期刊逻辑，机器人导航与具身智能定位清楚。 | 具体期刊未定，Transactions 与 Letters 的篇幅和贡献门槛不同。 | 实验完成后选择期刊，再生成长文版或精简版英文稿和匿名包。 |

## 2. Reviewer Objection Register

| Likely objection | Where triggered | Severity | What the reviewer may say | Preemptive fix | Status |
|---|---|---|---|---|---|
| 冻结参数不保证三种策略使用同一残差 | 方法：冻结式 GRPO | CRITICAL | “上游特征和动态 Top-K 变化会污染概率比。” | rollout 缓存完整动作残差、掩码和旧概率；更新不重算 Top-K；增加逐元素一致性测试。 | RESOLVED-DRAFT / TEST TO EXTEND |
| h=2/3 被写成已实现 | 摘要、方法、实验 | CRITICAL | “证据只支持一步，稿件却把多步当作现有算法。” | 全文标为计划受控扩展；以虚线表示；实现、测试和完整结果前不使用现在时结论。 | RESOLVED-DRAFT / EXPERIMENT OPEN |
| oracle q1 有选择偏差或泄漏 | 来源对照 | CRITICAL | “未执行候选没有真实未来，oracle 不可复现。” | 所有前五候选均克隆/恢复模拟器状态；固定预测链位姿；同编码器；不可达即无效并报告覆盖。 | PROTOCOL RESOLVED / IMPLEMENTATION OPEN |
| 主表混合不可比协议 | 主结果 | CRITICAL | “旧发布值与现代控制器行相邻会诱导错误比较。” | 同协议 R2R/RxR 独立表；旧 ETP-R1 另列文献背景且禁止求差。 | RESOLVED-DRAFT |
| 与 NavMorph 的增量不清楚 | 引言与相关工作 | MAJOR | “世界模型用于 VLN-CE 已有先例。” | 增加八维功能比较表；把贡献收窄到原生标记、逐 ghost 受限干预和缓存残差 GRPO。 | RESOLVED-DRAFT / SIGNIFICANCE OPEN |
| E24 是未定义内部代号 | 摘要与方法 | MAJOR | “读者无法从名称理解结构与作用。” | 正式名称改为候选残差评分头；方法给输入、三轮融合、tanh 上界和代码内部名。 | RESOLVED-DRAFT |
| STOP 隔离表述过强 | 方法与贡献 | MAJOR | “MOVE 分数变化仍会改变 STOP 概率。” | 统一写停止分数不直接加残差；区分普通 softmax 与 GRPO 保持停止概率质量的分布。 | RESOLVED-DRAFT |
| 低翻转率使贡献可能无实际意义 | 结果与讨论 | MAJOR | “模块几乎从不改变动作，却增加大量计算。” | episode 级与同状态级翻转分析；预测误差、覆盖、分数间隔、残差和成本联合报告。 | EXPERIMENT OPEN |
| 逐级消融不是单变量 | 逐级模块消融 | MAJOR | “变化可能来自额外参数、训练预算或强化更新。” | 每行登记唯一变化、起点、参数、样本量与计算；GRPO 增加同预算零残差对照。 | PROTOCOL RESOLVED / EXPERIMENT OPEN |
| 结果占位无法审计 | 摘要与全部结果表 | CRITICAL | “自由文本占位不能防止错填 split 或指标。” | 全部改为具名键；`results_placeholder_manifest.md` 绑定协议、原始 JSON、checkpoint、种子和统计方法。 | RESOLVED-DRAFT / DATA OPEN |
| 动作翻转被过度解释为因果 | 动作翻转分析 | MAJOR | “首次分歧后状态不同，单步动作不能解释终局。” | 分离 episode 配对与同状态一步反事实；以首次分歧归类；明确只作配对关联。 | RESOLVED-DRAFT / EXPERIMENT OPEN |
| 图表不可读或仍是文字占位 | 方法图、机制图、结果表 | MAJOR | “核心流程与单位无法从图表独立理解。” | 生成可编辑 Graphviz 图；拆分导航与成本表；Word/PDF 全页渲染检查。 | RESOLVED-DRAFT / NUMERIC CURVES OPEN |
| 目标期刊未定 | 全稿 | MAJOR | “稿件长度、匿名与模板不符合具体 venue。” | 保留中文逻辑母稿；实验闭环后选择期刊，再套模板并准备匿名/作者版本。 | OPEN |

## 3. Editorial Fit Map

- **Venue fit:** 当前目标是英文机器人与具身智能期刊，采用通用 IEEE 长文逻辑。主题属于连续机器人导航、视觉语言决策和世界模型交叉范围；具体期刊、篇幅和匿名规则仍为 OPEN。
- **Editor-facing value:** 若实验闭环成立，稿件提供一种可审计的候选级未来表征接口，并把世界模型误差、动作干预与强化更新边界放在同一框架内，而不是再次泛称“世界模型用于导航”。
- **Desk-reject risks:** 结果键未清空（OPEN）；主性能和机制证据不完整（OPEN）；目标模板未应用（OPEN）；两张结构图已生成（RESOLVED-DRAFT）；引用机制和章节数已通过自动检查（RESOLVED）；作者/匿名版、预印本差异、AI 辅助披露和数据可用性声明尚未准备（OPEN）。

