# 本地 TASE 与 VLN 论文写作参考

阅读日期：2026-09-05。用途：为 PILOT / TopoForesight 后续逐节写作建立参考，不根据尚未完成的实验改变方法定义。

## 阅读范围与依据

浏览 `/home/sia/project/论文/TASE/` 下 8 篇 PDF 的摘要、应用说明和章节结构，重点阅读 HEIGHT、ViTeC 的引言及相关工作，并参考语义建图、LE-Nav 的问题组织。另阅读 ETPNav、ETP-R1、HNR、NavMorph、DreamNav 和 DGNav 的摘要、引言与相关工作，针对 HNR 补查后继航点和前瞻图的实现描述。没有逐式审计所有论文，也没有复核各论文的实验结果。

原始 PDF 为主要依据；目录内三份中英对照 Markdown 仅作导航和辅助。视觉检查了 ViTeC 第 1–2 页、HEIGHT 第 1 页、HNR 第 1、5 页、ETP-R1 第 1 页、NavMorph 第 3 页。页码均为 PDF 文件页序。

文件夹名称不等同于出版证明：ViTeC、eLabrador、目标语义导航和 ObjectNav 综述带正式 TASE 卷页信息；HEIGHT 文件明确标注接受信息；另一些文件仍带 arXiv 或通用 LaTeX 模板页眉。本次不据文件名断定最终出版版本，也不据样本推定投稿硬性要求。

## 1. TASE 的格式和表达可以怎样借鉴

### 篇章结构服务于论文内容

8 份样本均出现 Abstract、Note to Practitioners 和 Index Terms。Note to Practitioners 是面向应用者的说明，通常回答应用场景、现有困难、方法如何帮助使用者，以及适用条件或后续改进；它与强调技术问题和贡献的摘要分工不同。当前指定稿没有这部分，后续宜补一段，正式要求和篇幅以投稿时的作者指南为准。

章节结构并不统一：ViTeC 将任务定义、总体方法、各模块和训练都放在 Method 下；HEIGHT 单列 Preliminaries，并分别组织仿真、真实环境实验及讨论；语义建图论文甚至把引言分成 Motivation、Challenges、Contributions。不要把“八个大节”或“引言必须分三个小节”当作 TASE 规则。我们的正文可保留任务定义—方法—训练结构，最终根据篇幅再合并。

双栏正文、罗马数字主节、字母小节、编号公式和数字引用在样本中常见。图放在单栏或跨栏位置取决于信息量。当前 Markdown 阶段先保持标题层级、术语、公式和引用一致，排版时再套正式模板。

### 用具体失败情境组织动机

HEIGHT 从人与静态障碍共存的场景出发，指出两类局限，再提出一个统领性问题，随后解释分离表示和异质交互建模如何回应问题。语义建图论文用道路、人行道和草地几何相似的例子说明为什么需要语义。ViTeC 用房间与物体的关系解释常识为何参与导航。

对我们的启发：保留“两条入口相似、关键地标在转角之后”的例子，让它贯穿候选到达预测和进一步查询的动机。三个研究问题可以作为这个问题的三个方面，不必写成三个并列的大课题。

### 引言说明设计作用，方法解释执行细节

参考论文在引言中已经会介绍模块，但通常先交代模块要补足的能力。后续稿应把每个设计与其作用紧密连接：到达预测补充候选表示，后继查询获得更远证据，有界修正控制直接干预，选择性触发减少无须执行的预测。

上一版引言中的最不利/最有利分数区间推导、逐候选回退细节、$K/k$ 定义和完整图制作要求偏细，可移至方法或绘图笔记。引言保留“若任何允许的修正都无法改变移动候选赢家，则省略深层查询”的直观解释。结果句仍按用户偏好完整书写，随后括注“待实验验证”。

### 贡献数量和实验位置无需机械模仿

HEIGHT 列 5 点，ViTeC 列 3 点，若干论文把实验验证单列一项。没有必要为了模仿样本给我们增加第四项或把工程缓存提升为独立贡献。保留三个技术作用不同的贡献，但分别说明机制与作用，避免摘要、方法概述和贡献列表反复复述同一句话。

## 2. VLN 论文的内容与论述手法

| 参考论文 | 本次关注位置 | 可借鉴的写法 | 与本稿的关系 |
|---|---|---|---|
| ETPNav | 引言、相关工作，PDF 第 1–3 页 | 先解释航点方法解决了什么，再把局限与地图、设计分析、控制器逐项对应 | 说明在线拓扑基础，不把其长程规划能力概括为“只会局部决策” |
| ETP-R1 | 引言、相关工作，PDF 第 1–2 页 | 在已有拓扑优势上定位训练方面的缺口，图 1 直接比较不同流程 | 本稿沿用训练基础，新增主线落在未来信息如何参与决策 |
| HNR / Lookahead Exploration | 引言、相关工作、§3.2.4、§3.3，PDF 第 1–2、5–6 页 | 图 1 展示当前视角与候选未来视角的区别；预测表示与前瞻规划分别解释 | 最接近工作之一，必须正面比较，详见下一节 |
| NavMorph | 引言、相关工作、方法总览，PDF 第 1–3 页 | 用统一问题串起潜在动态、前瞻规划和记忆更新；图 2 显示模块之间的信息流 | 区分其动态建模和在线适应目标与本稿冻结预测模型的候选接口 |
| DreamNav | 引言、相关工作，PDF 第 1–2 页 | 感知成本、规划视野、动作语义三个问题与模块一一对应，图 1 作概念对比 | 可学对应关系；其零样本、自我视角、轨迹级动作设置不能直接套入本稿 |
| DGNav / 动态拓扑感知 | 引言、相关工作，PDF 第 1–2 页 | 把图密度和连接问题收束到一个明确主题，并配指令情境说明 | 关注图构造与连接，本稿关注未来证据；不照搬“短视”或安全性等宽泛判断 |

图的分工尤其值得借鉴：HNR、ViTeC、DreamNav 和 ETP-R1 都在早期用图说明问题或概念差异，详细网络图后置。建议本稿以后让图 1 聚焦候选信息如何从当前观测扩展到到达预测、再到后继预测，完整模块、冻结状态和回退流程放方法框架图。先明确图要解释什么，再决定是否拆图和重新编号。

## 3. 本次发现的关键相近工作：HNR

HNR 已经做了以下事情：

1. 预测候选位置的未来语义特征，避免只依赖当前视角；
2. 在候选位置预测 12 个方向的特征与深度；
3. 用预测深度输入航点预测器，提出候选周围更远的可通行位置（§3.2.4，PDF 第 5 页）；
4. 将已访问节点、候选节点和前瞻节点组成图，编码并评估未来分支（§3.3，PDF 第 5–6 页）。

因此，“预测未来表征”“由预测信息产生更远位置”“将未来信息与候选对应”都不能仅凭这些概括认定为我们的独有贡献。此前主线明确了方法内部逻辑，但不能代替对相近工作的技术比较。

后续应围绕可核对的差异组织论述：

| 比较维度 | HNR 原文 | 本稿定义与此前代码核对 |
|---|---|---|
| 未来表示来源 | 特征云、层次神经辐射表示与多方向预测 | 冻结的动作条件表征世界模型，根据历史与查询条件预测潜在特征 |
| 后继位置来源 | 候选位置的预测深度，经航点预测器提出位置 | 候选预测 patch 表征，经空间查询提议器输出角度与距离 |
| 未来信息如何参与决策 | 前瞻节点加入图，与候选等节点联合编码并评估分支 | 第一阶段增强候选表示，基础评分后第二阶段对 Top-$K$ 候选施加有界修正 |
| 深层预算与跳过机制 | 本次重点阅读段落不足以断言其不存在其他预算控制 | 显式 Top-$K$ 与修正上界；决策不变性触发在此前核对时仍待正式集成 |

这些是机制比较，尚不等于已经证明新颖性或优越性。当前本地主工作区此前核对的默认展开为 $k=1$；本次没有重新审计其他工作区的新实现，旧实现状态不能延伸为所有分支的现状判断。

## 4. 后续写作采用的具体方向

1. 引言继续保留已建立的逻辑，但压缩实现解释，用一个贯穿性的导航例子说明前瞻价值。
2. 在引言中更具体地承认 HNR、NavMorph 等已有前瞻机制，用候选表示增强—基础评分—选择性分数校正说明本稿定位。
3. 相关工作建议分为“连续环境视觉语言导航与拓扑规划”“未来表征预测与导航前瞻”“选择性计算与受约束决策”。第三类的外部文献本次尚未专门检索，不能只靠本项目机制写成已有研究综述。
4. 强化微调放到基础策略和训练方法背景中说明；除非后续论述需要，不与选择性计算硬合为一个相关工作小节。
5. 补写面向应用者的说明，说明方法支持的高层导航环节、输入与计算条件、适用边界；不重复摘要的模块清单。
6. 方法正文按任务定义、总体流程、两阶段机制、干预与计算约束、训练衔接展开。实现待办和完整证明分别进入笔记与附录。

初次阅读仅建立写作参考，未改动论文正文。随后按用户要求完成第二轮引言修订：以“经过沙发后进入厨房”的走廊情境贯穿动机，补充 Dreamwalker、HNR 和 NavMorph 的具体机制，保留双阶段与有界修正的直观说明，精简符号和证明细节。

### 引言精简后保留的写作与绘图事项

- 默认 $k=1$、$K/k$ 区分、多步展开状态、候选级回退和决策不变性证明继续在问题定义与方法部分展开，不在引言重复。
- 图 1 已改为概念图图注，重点呈现当前观测、到达预测与后继预测如何辅助同一候选选择；画面中的预测必须与真实观测清楚区分。
- 完整方法图仍需说明基础策略路径、冻结和可训练模块、候选回退及选择性触发。实线突出默认路径，扩展和待集成机制另作标识；后续绘图时再安排方法图位置和编号，当前没有全篇重编号。
- 结果句保留完整表述并括注“待实验验证”，触发机制在引言中仍标明待集成。

## 5. 本地来源索引

以下均为本次实际打开并提取文本的文件；阅读深度以开头范围说明为准。

- [2024_Real_Time_Metric_Semantic_Mapping_for_Autonomous_Navigation_in_Outdoor_Environments](</home/sia/project/论文/TASE/2024_Real_Time_Metric_Semantic_Mapping_for_Autonomous_Navigation_in_Outdoor_Environments.pdf>)（12 页）。
- [2025_Efficient_Alignment_of_Unconditioned_Action_Prior_for_Language_Conditioned_Pick_and_Place](</home/sia/project/论文/TASE/2025_Efficient_Alignment_of_Unconditioned_Action_Prior_for_Language_Conditioned_Pick_and_Place.pdf>)（16 页）。
- [2025_eLabrador_Wearable_Navigation_System_for_Visually_Impaired_Individuals](</home/sia/project/论文/TASE/2025_eLabrador_Wearable_Navigation_System_for_Visually_Impaired_Individuals.pdf>)（17 页）。
- [2026_HEIGHT_Heterogeneous_Interaction_Graph_Transformer_for_Robot_Navigation](</home/sia/project/论文/TASE/2026_HEIGHT_Heterogeneous_Interaction_Graph_Transformer_for_Robot_Navigation.pdf>)（20 页）。
- [2026_Learning_to_Tune_Like_an_Expert_Interpretable_and_Scene_Aware_Navigation](</home/sia/project/论文/TASE/2026_Learning_to_Tune_Like_an_Expert_Interpretable_and_Scene_Aware_Navigation.pdf>)（13 页）。
- [A_Survey_of_Object_Goal_Navigation](</home/sia/project/论文/TASE/A_Survey_of_Object_Goal_Navigation.pdf>)（17 页）。
- [Goal-Oriented_Visual_Semantic_Navigation_Using_Semantic_Knowledge_Graph_and_Transformer](</home/sia/project/论文/TASE/Goal-Oriented_Visual_Semantic_Navigation_Using_Semantic_Knowledge_Graph_and_Transformer.pdf>)（11 页）。
- [Visual_and_Textual_Commonsense-Enhanced_Layout_Learning_for_Vision-and-Language_Navigation](</home/sia/project/论文/TASE/Visual_and_Textual_Commonsense-Enhanced_Layout_Learning_for_Vision-and-Language_Navigation.pdf>)（14 页）。
- [ETPNav](</home/sia/project/论文/ETPNav.pdf>)（21 页）。
- [ETP_r1](</home/sia/project/论文/ETP_r1.pdf>)（8 页）。
- [神经辐射](</home/sia/project/论文/神经辐射.pdf>)（14 页）。
- [NavMorph](</home/sia/project/论文/NavMorph.pdf>)（19 页）。
- [dreamNav](</home/sia/project/论文/dreamNav.pdf>)（8 页）。
- [动态拓扑感知](</home/sia/project/论文/动态拓扑感知.pdf>)（12 页）。

## 6. 方法 A 的结构与初稿（2026-09-06）

本轮重新阅读本地 ETPNav、ETP-R1、HNR、NavMorph、DGNav 的方法开头；重点查看任务说明如何接入方法、是否继续细分标题，以及基础框架介绍的篇幅。没有依据本轮阅读重述各论文性能结论。视觉复核 DGNav 与 ETP-R1 的 PDF 第 3 页。

| 论文 | 方法开头的安排 | 对本稿的启发 |
|---|---|---|
| ETPNav | 方法开头用 Task Setup 和 Overview 段落，之后进入 §3.1 Topological Mapping | 先定义输入和行动层级，建图细节按方法需要展开 |
| ETP-R1 | 方法开头简述任务与总体思路，明确沿用 ETPNav 建图和控制，省略重复细节 | 借用框架清楚归属，不重新介绍全部基线网络 |
| HNR | §3.1 Navigation Setups 从任务和观测直接过渡到候选预测流程 | 单个任务小节可以自然连接方法，无须再拆任务/基础框架两级标题 |
| NavMorph | Task Definition 与 Framework Overview 作为方法开头的段落引导 | 符号在叙述中按需引入，避免独立符号清单 |
| DGNav | III-A Problem Formulation and Overview 在同一小节内连接任务、在线拓扑图与框架 | 最接近本稿当前结构；本稿已有 B 总览，因此 A 不提前展开全部新增模块 |

据此，删除 III-A 下的两个编号小标题，写成三个自然段：任务、时间步和观测历史；拓扑图及 ghost 候选的来源；候选动作集合、执行方式和基础评分的定义。复用引用 [2]、[7]、[9]，不新增文献。缓存更新/失效、STOP 残差及软最大概率的边界留在 C、D、E 的已有说明中，A 只保留候选均值位置与后续到达查询的对应关系。

为避免直接照搬他文假设，还核对当前工作区 `vlnce_baselines/models/graph_utils.py` 的 `update_graph`、`get_node_embeds` 和候选位置校验，确认重复候选的位置均值及视觉聚合；查看 `ss_trainer_ETP_R1.py` 的 `_nav_gmap_variable` 和行动转换，以及 `ETP_R1_vilmodel_cmt.py` 的已访问节点评分掩码。正文明确已访问节点用于图推理及路径回溯，最终探索候选与 STOP 分开。`s_{t,i}` 沿用本稿定义，表示第一阶段增强之后、第二阶段修正之前的评分。本次为本地写作和只读代码核对，没有运行训练或评测。
