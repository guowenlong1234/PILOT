# 相关工作 A：联网检索与写作逻辑

日期：2026-09-05。范围：连续环境视觉语言导航与拓扑规划。对应正文：`paper/drafts/TopoForesight_TASE_初稿骨架.before-related-work.md` 的 II-A。

## 1. 检索范围和实际阅读

先通过 Google 检索任务综述、航点与地图方法、近期视频导航路线及混合地图表示，再读取 CVF 官方论文库、ACL Anthology、arXiv 原始论文页面与作者项目页。搜索结果用于定位文献，正文的技术陈述依据原始来源。

本次原始来源覆盖 14 项研究：R2R、RxR、VLN-CE、WPN、候选航点预测、Cross-Modal Map Learning、DUET、GridMM、BEVBert、ETPNav、ETP-R1、DGNav、NaVid、StreamVLN。另浏览 CVPR 2022、ICCV 2021 和 ICCV 2023 官方论文目录中与导航有关的条目。没有把检索结果或目录条目都当作已精读文献。

实际搜索词：

- `vision language navigation continuous environments topological mapping survey 2025 2026`
- `VLN CE waypoint prediction WS-MGMap GridMM BEVBert`
- `StreamVLN NaVid MapNav continuous navigation 2025 paper`
- `BEVBert Multimodal Map Pre-training Language Guided Navigation DUET`

阅读深度：14 项研究的原始摘要/方法概述及书目信息；WPN、DUET、GridMM、BEVBert 另外下载原始 PDF 并提取前 4 页，核对任务范围、空间表示和动作抽象。ETPNav、ETP-R1、DGNav 的联网摘要与此前本地阅读相互参照。本次未审计这些工作的代码或复核实验结果。

公开页面的临时原文、提取文本、检索记录及 4 份 PDF 保存在 `/tmp/etpr1-related-a-web-20260905/`，作为本次操作的临时材料，不加入仓库；长期保留的来源和取舍记录如下。

## 2. 写作逻辑

初版按任务、航点、地图和后续改进组织；第二版显式区分两类决策核心，但分类解释过重。根据用户反馈，融合两版后，让技术路线的区别体现在文献叙述中，删除对分类标准的专门解释；最新修订将结尾的优势与局限单列，形成以下五段：

1. **任务背景与近期大模型方法。** 简述 R2R、RxR 和 VLN-CE，自然过渡到 NaVid、StreamVLN 如何利用视觉语言大模型处理观测历史并生成动作。
2. **任务专用策略与航点。** 转入面向导航设计的跨模态策略，用 WPN 和 Hong 等的方法说明中间目标如何衔接指令理解与连续运动。
3. **空间记忆与在线拓扑。** 从局部观测外的规划需求出发，介绍 DUET、GridMM、BEVBert，再落到 ETPNav 的在线拓扑组织和规划回溯。保留 DUET 的离散环境限定。
4. **后续改进。** 概括 DGNav 和 ETP-R1 对图结构与训练的改进。
5. **路线优势、局限与本文定位。** 根据用户最新要求，结尾单列一段，说明拓扑候选既支持空间记忆上的规划与回溯，也方便将未来预测与具体候选对应；随后以 HNR [8] 已讨论的视野不足为依据，指出位置与连接关系不能直接提供到达后的视觉内容，自然引出本稿的候选未来证据。保留对大模型工作的正面、具体描述，不增加对其他路线的笼统批评。

保留原有 13 项直接相关研究和参考文献 [13]–[18]，最新修订另外复用已有 HNR [8] 支撑候选视野不足的陈述，未增加书目条目。这里的“任务专用”指决策网络的设计，不能解释为不使用任何预训练视觉或语言模型；同样，采用大模型也不意味着不使用地图。本轮修订仅改变 A 节组织和表述，不重新检索文献或改动引用条目。

## 3. 论点与来源对应

| 研究 / 正文编号 | 原始来源 | 本次支持的陈述与取舍 |
|---|---|---|
| R2R [1] | [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html) | 真实室内场景的自然语言导航基准；仅用作任务起点 |
| RxR [3] | [ACL Anthology](https://aclanthology.org/2020.emnlp-main.356/) | 多语言指令及词语与位姿的时空对齐；不把离散 RxR 原论文直接称为 RxR-CE 提出论文 |
| VLN-CE [2] | [arXiv](https://arxiv.org/abs/2004.02857) | 连续三维环境与低层动作，放宽已知环境拓扑、短距离理想化移动等假设 |
| WPN [13] | [ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Krantz_Waypoint_Models_for_Instruction-Guided_Navigation_in_Continuous_Environments_ICCV_2021_paper.html)、[作者项目](https://jacobkrantz.github.io/waypoint-vlnce/) | 指令与全景视觉共同驱动相对航点预测，并比较动作空间表达方式；不与后来的独立候选生成直接混为一谈 |
| 候选航点预测 [6] | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Bridging_the_Gap_Between_Learning_in_Discrete_and_Continuous_Environments_CVPR_2022_paper.html) | 预测可通行候选，为离散高层策略迁移到连续环境提供接口 |
| NaVid [14] | [arXiv](https://arxiv.org/abs/2402.15852) | 单目 RGB 视频和指令到下一步动作；页面标注 RSS 2024 接受。正文不引用其性能宣称，也不笼统评价地图路线与视频路线谁更优 |
| StreamVLN [15] | [arXiv v2](https://arxiv.org/abs/2507.05240v2)、[作者项目](https://streamvln.github.io/) | 快更新对话上下文与慢更新历史记忆，支持流式导航；使用实际读取的 v2 信息 |
| DUET [16] | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Think_Global_Act_Local_Dual-Scale_Graph_Transformer_for_Vision-and-Language_Navigation_CVPR_2022_paper.html) | 全局图上粗粒度规划与局部细粒度语言理解；原文评测 REVERIE、SOON、R2R，不称其为原生连续环境系统 |
| GridMM [17] | [ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_GridMM_Grid_Memory_Map_for_Vision-and-Language_Navigation_ICCV_2023_paper.html) | 历史特征投影到动态自我中心栅格地图，按指令聚合区域线索；原文包括离散任务与 R2R-CE |
| BEVBert [18] | [arXiv v2](https://arxiv.org/abs/2212.04385v2) | 局部度量地图与全局拓扑图、多模态地图预训练；全文包括 R2R-CE，页面标注 ICCV 2023 |
| ETPNav [7] | [arXiv](https://arxiv.org/abs/2304.03047) | 在探索中自组织预测航点、在线拓扑建图、跨模态规划与避障执行；承认其已有长程规划能力 |
| ETP-R1 [9] | [arXiv](https://arxiv.org/abs/2512.20940) | 扩展指令数据、R2R/RxR 联合预训练及在线监督与强化微调；仅说明训练路线，不复制其“首次”或最优结果主张 |
| DGNav [10] | [arXiv](https://arxiv.org/abs/2601.21751) | 根据航点分布调整图阈值，并融合视觉、语言、几何信息更新连接；只描述机制，不沿用原文所有安全性或“短视”概括 |
| Cross-Modal Map Learning（本节未引用） | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Georgakis_Cross-Modal_Map_Learning_for_Vision_and_Language_Navigation_CVPR_2022_paper.html) | 对观测与未观测区域预测俯视语义图，再生成航点路径。可留作 B 节地图预测路线的参考；其存在提醒我们不能声称所有地图方法都只记录已观测信息 |

## 4. 书目和表述检查

- 新条目 [13]、[16]、[17] 的作者、标题、会议与页码来自 CVF 官方页面及 BibTeX。
- NaVid、StreamVLN、BEVBert 使用核对过的 arXiv 版本作为新增条目的可追溯来源。已知接受信息在上表记录；不填未经核实的会议页码或 DOI。
- StreamVLN 首版发表于 2025-07-07，本次访问的最新版为 2026-07-09 的 v2，页面标注 Accepted to ICRA 2026。因此新增条目 [15] 按 v2 标为 2026 年，不是把原始首发年份改成 2026。
- BEVBert 首版为 2022 年，采用的 v2 为 2023 年，因此 [18] 标明版本与年份。
- 原条目 [6] 存在源间页码差异：CVF 页面为 15439–15449，而 [Crossref 出版社登记](https://api.crossref.org/works/10.1109/CVPR52688.2022.01500) 为 15418–15428，DOI 与当前稿一致。本次保留既有 IEEE/DOI 口径，不将版本差异误判为引用错误。
- Crossref 对 BEVBert 的题名检索返回不匹配文献，未采用这些结果；其来源改用已核对的 arXiv 页面和 PDF。
- 当前保留原有 [1]–[12] 编号，新条目追加为 [13]–[18]，避免改动其他章节。全文完成后再按首次引用顺序统一编号。
- 没有将搜索摘要、模型介绍网站或本项目尚未完成的实验当成性能证据；A 节只陈述已有工作的任务和方法。

## 5. 本轮修改范围

初次写作将 II-A 提纲替换为正文，并追加 6 项参考文献；后续保留两条技术路线的区分，以连贯论述融合初版与分类版，最新修订单列优势、局限与本文定位段，全部原有引用保留。引言、II-B、II-C、方法和实验未修改。保留用户已暂存修改，未执行 Git 提交、暂存或远端同步。
