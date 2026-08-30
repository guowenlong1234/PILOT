# 优秀论文写法学习档案（Agent B）

## 使用边界

本档案服务于“英文机器人与具身智能期刊、通用 IEEE 期刊结构”的中文逻辑母稿设计。外部论文只用于学习章节组织、问题提出、证据编排和语言强度，不把任何外部论文的实验数值、领先结论或优越性当作 ETP-R1 的结果。ETP-R1 的事实边界以本地 `README.md`、后续实验记录和用户确认结果为准；尚未完成的 R2R/RxR、SFT/GRPO 实验只能写作 `[待填]`。

## Exemplar Inventory

### A. 目标期刊/场景范例（6 篇）

| ID | 论文与出版场景 | 年份 | 选择理由与主要学习点 | 来源渠道与 URL/路径 |
|---|---|---:|---|---|
| J1 | **ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments**, *IEEE Transactions on Pattern Analysis and Machine Intelligence* | 2025（在线发表 2024） | 与本项目的拓扑规划、VLN-CE 场景最接近；学习“连续环境困难 → 拓扑抽象的必要性 → 规划/控制分解 → 双基准验证”的期刊型主线，以及方法总览后分别解释感知、规划、控制、训练与推理的层级。 | Crossref/DOI 官方出版元数据：<https://doi.org/10.1109/TPAMI.2024.3386695>；arXiv 官方元数据与全文：<https://arxiv.org/abs/2304.03047> |
| J2 | **ESceme: Vision-and-Language Navigation with Episodic Scene Memory**, *International Journal of Computer Vision* | 2024 | 学习如何把“同一场景中的历史经验未被利用”收束为单一缺口，再用问题定义、场景记忆、候选增强三个连续小节解释机制，最后用主结果、消融和定性案例闭环。 | Crossref/DOI：<https://doi.org/10.1007/s11263-024-02159-8>；arXiv：<https://arxiv.org/abs/2303.01032> |
| J3 | **Talk2Nav: Long-Range Vision-and-Language Navigation with Dual Attention and Spatial Memory**, *International Journal of Computer Vision* | 2020 | 虽非近期论文，但它是长距离语言导航的成熟期刊范例；学习先明确输入、目标和长程依赖，再把语言理解、视觉关注、空间记忆和动作选择写成因果链，而不是模块清单。 | Crossref/DOI：<https://doi.org/10.1007/s11263-020-01374-3>；arXiv：<https://arxiv.org/abs/1910.02029> |
| J4 | **Symmetry-aware Neural Architecture for Embodied Visual Navigation**, *International Journal of Computer Vision* | 2023 | 学习“归纳偏置为什么适合该任务”的论证：先指出现有网络忽略某类结构，再解释结构如何进入网络，随后分别验证准确性、数据效率和跨环境泛化。适合借鉴到“为何 RAE/DINOv2 表征与拓扑策略需要特定接口”的写法。 | Crossref/DOI：<https://doi.org/10.1007/s11263-023-01909-4>；arXiv：<https://arxiv.org/abs/2112.09515> |
| J5 | **A Survey of Embodied AI: From Simulators to Research Tasks**, *IEEE Transactions on Emerging Topics in Computational Intelligence* | 2022 | 学习 IEEE 长文如何先定义范围和分类轴，再按模拟器、任务、方法、数据集、指标逐层展开，并用表格承担高密度比较；可用于相关工作开头建立 VLN、VLN-CE、离散/连续环境和训练范式的边界。 | Crossref/DOI：<https://doi.org/10.1109/TETCI.2022.3141105>；arXiv：<https://arxiv.org/abs/2103.04918> |
| J6 | **Embodied Navigation with Multi-modal Information: A Survey from Tasks to Methodology**, *Information Fusion* | 2024 | 学习近期综述的“任务分类 → 感知模态 → 融合/决策方法 → 数据与指标 → 开放问题”组织方式。对 ETP-R1 最有价值的是按问题维度综合文献，而不是逐篇罗列。 | Crossref/DOI 官方元数据：<https://doi.org/10.1016/j.inffus.2024.102532> |

### B. 近期高质量 VLN/VLN-CE/具身导航范例（8 篇）

| ID | 论文与出版场景 | 年份 | 选择理由与主要学习点 | 来源渠道与 URL/路径 |
|---|---|---:|---|---|
| F1 | **BEVBert: Multimodal Map Pre-training for Language-guided Navigation**, ICCV | 2023 | 学习“地图表征 + 多模态预训练任务 + 下游微调”的三段式方法论，以及如何用预训练任务与下游能力一一对应，而不把预训练仅写成数据规模增加。 | arXiv 官方元数据与全文：<https://arxiv.org/abs/2212.04385> |
| F2 | **Scaling Data Generation in Vision-and-Language Navigation (ScaleVLN)**, ICCV | 2023 | 学习数据论文的完整论证：为什么现有数据不足、如何控制生成质量与多样性、怎样训练、如何证明收益来自数据而非不可控变量。与 ETP-R1 的 Gemini 标注和 R2R/RxR 联合预训练叙事直接相关，但具体数据结论不可照搬。 | 项目官方页：<https://scalevln.github.io/>；arXiv：<https://arxiv.org/abs/2307.15644> |
| F3 | **NaviLLM: Towards Multi-task Multimodal Large Language Models for Generalized Embodied Navigation**（arXiv 题名为 *Towards Learning a Generalist Model for Embodied Navigation*）, CVPR | 2024 | 学习多任务统一论文如何先提出统一任务模式，再解释输入输出模式、任务特定提示和训练混合，最后按任务族、泛化与消融分层报告，避免“所有任务挤在一张表里”。 | arXiv 官方元数据与全文：<https://arxiv.org/abs/2312.02010> |
| F4 | **NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation**, Robotics: Science and Systems | 2024 | 学习从“离散候选动作/预建图依赖”转向“历史视频 + 当前观测 → 下一步动作”的问题重述；方法部分把模型、VLN-CE 建模、训练数据收集和协同训练分开，实验再区分模拟、真实机器人和泛化。 | RSS 官方 DOI：<https://doi.org/10.15607/RSS.2024.XX.079>；arXiv：<https://arxiv.org/abs/2402.15852> |
| F5 | **NavGPT-2: Unleashing Navigational Reasoning Capability for Large Vision-Language Models**, ECCV | 2024 | 学习把“推理能力”落到可训练接口和可观测行为，而不是只展示思维链案例；相关工作围绕零样本推理与可训练导航模型形成张力，实验同时覆盖结果、推理可视化与部件消融。 | Springer/ECCV 官方 DOI：<https://doi.org/10.1007/978-3-031-72667-5_15>；arXiv：<https://arxiv.org/abs/2407.12366> |
| F6 | **MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-Language Navigation**, ACL | 2024 | 学习“地图如何进入提示、推理如何转成动作、错误后如何自适应重规划”的链式描述；实验结构从设置与基线到主结果、分析和案例，适合学习对闭环规划能力的可解释展示。 | ACL Anthology 官方出版页：<https://aclanthology.org/2024.acl-long.529/>；arXiv：<https://arxiv.org/abs/2401.07314> |
| F7 | **NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models**, AAAI | 2024 | 学习早期 LLM 导航论文如何明确“零样本能力探索”的边界，把视觉感知器、提示管理器、推理—动作协同分别交代，并用定性与定量证据区分“会解释”和“会导航”。 | AAAI 官方 DOI：<https://doi.org/10.1609/aaai.v38i7.28597>；arXiv：<https://arxiv.org/abs/2305.16986> |
| F8 | **VLN-R1: Vision-Language Navigation via Reinforcement Fine-Tuning**, arXiv 预印本 | 2025 | 与本项目 GRPO/RFT 主题最直接；学习把数据引擎、监督微调和强化微调写成递进因果链，并明确奖励、采样、优化目标和消融。因其为预印本，只作为结构与近期技术语境范例，不把其结论当成已充分同行评审的事实。 | arXiv 官方元数据与全文：<https://arxiv.org/abs/2506.17221> |

### C. 本项目上下文（不是外部范例）

| ID | 材料 | 用途 | 来源渠道与路径 |
|---|---|---|---|
| L1 | ETP-R1 README | 约束母稿当前叙事：拓扑图式 VLN-CE、Gemini 生成预训练数据、R2R/RxR 联合预训练、在线 SFT、基于 GRPO 的在线 RFT；其中性能表述必须等待实际实验回填。 | 本地材料：`/home/sia/project/ETP-R1/README.md` |
| L2 | PaperSpine 配置与本地来源索引 | 约束场景、语言、占位符和来源优先级；确认 `tier=pro`、中文逻辑母稿、通用 IEEE 期刊结构、未完成实验不得虚构。 | 本地配置：`/home/sia/project/ETP-R1/paper_rewriting_output/paper_spine_config.json`；本地索引：`/home/sia/project/ETP-R1/paper_rewriting_output/reference_materials/source_index.md` |

## Structural Patterns

1. **引言采用“任务价值—关键张力—具体缺口—方案概念—贡献与证据”五步推进。** 成熟论文不会从模型名称起笔。先用一段界定 VLN-CE 的目标和连续控制困难；第二段承认拓扑图方法的结构优势，同时指出其在数据规模或训练范式上的不足；第三段解释为什么简单扩大监督学习仍不够；第四段再引出 ETP-R1；最后用平行贡献句说明“数据、训练、验证”三类贡献。这样可避免把 README 中的多个亮点堆成同一层级。来源：arXiv 全文（J1 <https://arxiv.org/abs/2304.03047>；F2 <https://arxiv.org/abs/2307.15644>；F8 <https://arxiv.org/abs/2506.17221>）与本地 README（`/home/sia/project/ETP-R1/README.md`）。

2. **相关工作按“问题轴”综合，而不是按时间逐篇摘要。** 建议设置三条轴：① VLN-CE 的拓扑图规划与连续控制；② 导航预训练、数据扩展与跨任务联合训练；③ 大模型/策略模型的监督微调和强化微调。每小节都用“已有方法解决了什么—仍缺什么—本文处于哪里”收尾，使相关工作直接支撑贡献边界。来源：期刊综述与近期论文（J5 DOI <https://doi.org/10.1109/TETCI.2022.3141105>；J6 DOI <https://doi.org/10.1016/j.inffus.2024.102532>；F5 <https://arxiv.org/abs/2407.12366>；F7 <https://arxiv.org/abs/2305.16986>）。

3. **方法部分先给统一符号和总览，再沿信息流拆分。** 推荐顺序是：任务定义与符号 → 框架总览 → 视觉/语言与拓扑状态表示 → 数据构建和联合预训练 → 在线 SFT → 在线 GRPO/RFT → 推理与连续控制。每小节开头先写“本节解决上一阶段遗留的什么问题”，结尾写输出如何成为下一阶段输入。来源：J1 的方法层级（<https://arxiv.org/abs/2304.03047>）、F1 的地图预训练层级（<https://arxiv.org/abs/2212.04385>）、F4 的模型/任务建模/训练分层（<https://arxiv.org/abs/2402.15852>）、F8 的数据引擎/SFT/RFT 分层（<https://arxiv.org/abs/2506.17221>）。

4. **把数据工程写成可审查的方法，而不是一句“我们生成了更多数据”。** 数据小节至少回答来源、轨迹采样、提示/标注流程、低幻觉或质量控制、去重/过滤、数据量统计、R2R 与 RxR 混合策略和泄漏检查。训练增益必须通过同模型、同预算的受控比较来归因。来源：ScaleVLN 项目页与全文（F2 <https://scalevln.github.io/>、<https://arxiv.org/abs/2307.15644>），以及本地数据叙事（`/home/sia/project/ETP-R1/README.md`）。

5. **把三阶段训练写成能力递进，而不是检查点流水账。** 联合预训练回答“如何获得语言—拓扑路径基础对齐”；在线 SFT 回答“如何适应闭环交互和专家纠偏”；在线 RFT 回答“如何直接优化长程导航结果”。每阶段需明确输入数据、优化目标、可训练/冻结模块、初始化来源和输出检查点。来源：F3 的统一训练结构（<https://arxiv.org/abs/2312.02010>）、F8 的 SFT/RFT 结构（<https://arxiv.org/abs/2506.17221>）和本地检查点层级（`/home/sia/project/ETP-R1/README.md`）。

6. **实验部分按“贡献承诺”组织。** 推荐顺序：实验设置与实现细节；R2R-CE 主结果；RxR-CE 主结果；训练阶段递进比较（预训练→SFT→GRPO）；数据与模型部件消融；奖励/采样/冻结策略消融；泛化、效率和定性案例；局限与失败案例。每个结果小节首句写清要验证哪项贡献，末句只总结表格支持的结论。来源：J1（<https://arxiv.org/abs/2304.03047>）、J2（<https://arxiv.org/abs/2303.01032>）、F3（<https://arxiv.org/abs/2312.02010>）、F8（<https://arxiv.org/abs/2506.17221>）。

7. **期刊版比会议版多承担“解释”和“边界”。** 除主结果外，应补充训练稳定性、超参数敏感性、计算成本、不同初始条件、失败案例和局限；附录给出提示模板、奖励定义、数据统计和复现实节。这样扩展的是证据链，而不是重复描述网络结构。来源：TPAMI/IJCV 期刊范例（J1 DOI <https://doi.org/10.1109/TPAMI.2024.3386695>；J2 DOI <https://doi.org/10.1007/s11263-024-02159-8>；J4 DOI <https://doi.org/10.1007/s11263-023-01909-4>）。

8. **图表承担叙事节点。** 总览图应表达数据生成、联合预训练、任务特定 SFT/RFT 和推理之间的箭头关系；主表按 R2R-CE/RxR-CE 分开；训练阶段表体现增量；消融表一次只回答一个问题；定性图展示拓扑决策或错误恢复。正文先提出阅读问题，再引用图表，最后解释机制，不能只写“如表所示”。来源：J1（<https://arxiv.org/abs/2304.03047>）、F4（<https://arxiv.org/abs/2402.15852>）、F6（ACL 官方页 <https://aclanthology.org/2024.acl-long.529/>）与本地总览图索引（`/home/sia/project/ETP-R1/assets/Overview.png`）。

## Rhetorical Patterns

1. **用受限比较建立缺口。** 推荐写法不是“图方法无法利用大规模数据”，而是“现有图方法主要围绕结构化规划和模仿学习优化，尚未系统检验数据扩展与闭环强化微调能否协同提升拓扑策略”。它既承认前人贡献，也把本文问题限定为可验证命题。来源：J1 <https://arxiv.org/abs/2304.03047>；F2 <https://arxiv.org/abs/2307.15644>；F8 <https://arxiv.org/abs/2506.17221>。

2. **贡献使用“提出—构建—验证”的动词梯度。** “提出”留给框架或训练范式，“构建”用于数据集/训练管线，“验证”用于实验；不要每条都写“首次”。三条贡献应分别对应方法图、数据/训练细节和结果章节。来源：J1 <https://arxiv.org/abs/2304.03047>；F1 <https://arxiv.org/abs/2212.04385>；F4 <https://arxiv.org/abs/2402.15852>。

3. **机制主张后立即给出可检验后果。** 例如，若主张联合预训练改善跨任务基础，应预告 R2R/RxR 单独与联合数据的受控消融；若主张 GRPO 改善闭环策略，应预告 SFT 与 RFT 的同初始化比较、训练曲线和冻结检查。来源：J4 <https://arxiv.org/abs/2112.09515>；F8 <https://arxiv.org/abs/2506.17221>；本地实验规划 `paper_spine_config.json`。

4. **把“规模”写成质量、覆盖和训练信号三者的共同作用。** 对 Gemini 数据不能只强调数量；应论证指令多样性、轨迹覆盖、低幻觉控制以及与拓扑路径的对齐方式，再用数据量与过滤策略消融区分原因。来源：F2 项目页 <https://scalevln.github.io/>；F1 <https://arxiv.org/abs/2212.04385>；本地 README `README.md`。

5. **用因果过渡连接三个训练阶段。** 典型句法是“预训练提供 X，但仍缺少 Y；因此进行在线 SFT。SFT 学会专家纠偏，但目标仍受逐步监督限制；因此进一步用 RFT 直接优化序列级奖励”。这个过渡比简单写 “then/finally” 更能说明为什么每一步必要。来源：F3 <https://arxiv.org/abs/2312.02010>；F8 <https://arxiv.org/abs/2506.17221>；本地 README `README.md`。

6. **结果段遵循“问题—证据—解释—边界”。** 先问某设计是否有效，再指向指定行/列，随后解释与机制是否一致，最后说明只在何数据集、划分和指标上成立。避免从单一指标直接推导“更强推理能力”。来源：J2 <https://arxiv.org/abs/2303.01032>；F6 <https://arxiv.org/abs/2401.07314>；F7 <https://arxiv.org/abs/2305.16986>。

7. **定性案例用于解释定量现象，不用于替代定量证据。** 路径可视化应选能展示拓扑长程规划、错误恢复或 RFT 前后行为差异的案例，并同时给失败例。来源：J3 <https://arxiv.org/abs/1910.02029>；F5 <https://arxiv.org/abs/2407.12366>；F6 ACL 官方页 <https://aclanthology.org/2024.acl-long.529/>。

8. **新颖性主张分级。** 在完成系统检索前，可写“to our knowledge”并限定到“closed-loop online RFT for graph-based VLN-CE policies”；若证据不足，降级为“we investigate”或“we instantiate”。预印本只能说明近期方向，不能单独支撑“首个”。来源：F8（预印本边界）<https://arxiv.org/abs/2506.17221>；本地配置（不得虚构）`/home/sia/project/ETP-R1/paper_rewriting_output/paper_spine_config.json`。

## Language Patterns

1. **段落首句只承担一个功能。** 引言段分别使用“定义任务”“指出张力”“提出方法”“概括证据”，方法段使用“本模块解决什么”，实验段使用“本实验检验什么”。一个段落不要同时介绍背景、方法和结果。来源：J1 <https://arxiv.org/abs/2304.03047>；J2 <https://arxiv.org/abs/2303.01032>；F4 <https://arxiv.org/abs/2402.15852>。

2. **关键名词保持一词一义。** 建议固定使用 `graph-based VLN-CE policy`（图式 VLN-CE 策略）、`joint pre-training`（联合预训练）、`online supervised fine-tuning`（在线监督微调）、`reinforcement fine-tuning`（强化微调）和 `GRPO`；首次出现给全称和定义，之后不在 RFT/RL/GRPO 之间随意切换。来源：IEEE 期刊风格 J1 <https://doi.org/10.1109/TPAMI.2024.3386695>；本地 README `/home/sia/project/ETP-R1/README.md`。

3. **贡献句使用平行语法。** 可采用：`We construct ...`; `We develop ...`; `We evaluate ...`。每句先给对象，再给作用，最后给证据范围。不要把“novel、effective、robust、general”连续堆叠。来源：F1 <https://arxiv.org/abs/2212.04385>；F3 <https://arxiv.org/abs/2312.02010>；F4 <https://arxiv.org/abs/2402.15852>。

4. **对比连接词体现逻辑关系。** `However` 用于真正矛盾，`In contrast` 用于方法差异，`Consequently` 用于因果结果，`Building on this foundation` 用于阶段递进。不要每段都用 `Moreover` 罗列。来源：J3 <https://arxiv.org/abs/1910.02029>；F6 <https://arxiv.org/abs/2401.07314>；F8 <https://arxiv.org/abs/2506.17221>。

5. **主张强度与证据匹配。** 有完整对照时用 `outperforms/improves`；只有趋势时用 `suggests/is consistent with`；尚未回填时写 `[待填]`，不能提前写 `establishes a new state of the art`。跨数据集才可说 `generalizes across benchmarks`，真实机器人测试才可谈 sim-to-real。来源：F4（模拟/真实环境分开）<https://arxiv.org/abs/2402.15852>；本地配置 `/home/sia/project/ETP-R1/paper_rewriting_output/paper_spine_config.json`。

6. **表格解读给精确定位，不复述整表。** 推荐句框：`Compared with the SFT checkpoint under the same initialization, GRPO changes [metric] by [待填], while [metric] remains [待填].` 随后解释其是否支持序列级优化假设。来源：J1 <https://arxiv.org/abs/2304.03047>；F8 <https://arxiv.org/abs/2506.17221>。

7. **方法描述优先主动、具体的动词。** 使用 `encodes, aggregates, samples, scores, updates, freezes`，少用空泛的 `handles, leverages, enhances`；如写“enhances”，必须说明增强了何种表示或优化信号。来源：F1 <https://arxiv.org/abs/2212.04385>；F5 <https://arxiv.org/abs/2407.12366>；F7 <https://arxiv.org/abs/2305.16986>。

8. **限制段明确“未证明什么”。** 可写：当前证据限定于 MP3D/Habitat 中的 R2R-CE 与 RxR-CE；数据生成依赖外部模型；GRPO 的计算成本与奖励敏感性需要报告；若无真实机器人实验，不宣称部署可靠性。来源：J5 <https://arxiv.org/abs/2103.04918>；F4 <https://arxiv.org/abs/2402.15852>；本地配置与 README。

## 面向 ETP-R1 母稿的直接落地建议

建议总结构为：**1 引言；2 相关工作；3 问题定义与框架总览；4 高质量拓扑轨迹数据与联合预训练；5 在线 SFT 与 GRPO 强化微调；6 实验；7 讨论与局限；8 结论**。其中第 6 节必须让每个小节回答一个贡献问题：联合数据是否必要、RAE/DINOv2 表征是否带来可归因变化、在线 SFT 提供什么、GRPO 在同初始化下新增什么、R2R 与 RxR 是否一致、收益是否伴随路径长度或训练稳定性代价。所有具体数字先写 `[待填]`，只有在实验记录确认后再替换。

来源：综合 J1/F1/F2/F4/F8 的结构学习（<https://arxiv.org/abs/2304.03047>、<https://arxiv.org/abs/2212.04385>、<https://arxiv.org/abs/2307.15644>、<https://arxiv.org/abs/2402.15852>、<https://arxiv.org/abs/2506.17221>）及本地权威材料（`/home/sia/project/ETP-R1/README.md`、`/home/sia/project/ETP-R1/paper_rewriting_output/paper_spine_config.json`）。
