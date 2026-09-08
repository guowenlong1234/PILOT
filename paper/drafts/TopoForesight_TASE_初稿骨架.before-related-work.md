# PILOT: Progressive World-Model Lookahead over Online Topologies for Vision-Language Navigation

> 来源：`docs/TopoForesight_TASE_初稿骨架.before-related-work.docx`  
> 整理说明：保留原稿内容、公式、表格、图注与待补占位；将 Word 标题层级、列表和表格转换为 Markdown，便于在 VS Code 中查看与继续修改。

## 摘要

视觉语言导航（VLN）要求智能体理解自然语言指令，并在未见过的环境中进行连续感知和决策。在线拓扑方法能够利用历史观测组织长程导航，但对未访问候选的判断仍主要依赖当前视觉与拓扑记忆，难以获知候选方向之后可能出现的场景。为解决这一问题，本文提出PILOT，一种面向在线拓扑视觉语言导航的渐进式世界模型前瞻框架。PILOT利用冻结的表征世界模型，在动作执行前预测与拓扑候选对应的未来视觉信息。该框架首先为多个候选补充到达后的预测表征，再对少量关键候选进行进一步前瞻，并利用预测结果改善候选选择。与此同时，PILOT通过限制预测对基础策略的修正范围，并仅在可能影响当前决策时投入深入前瞻计算，提高方法的可靠性和计算效率。大量实验在R2R-CE和RxR-CE数据集上进行。结果表明，PILOT能够稳定改善视觉语言导航性能，并在导航效果与计算开销之间取得良好平衡。代码见：`https://github.com/____/PILOT`。

**关键词：** 视觉语言导航；具身智能；拓扑规划；表征世界模型；预测前瞻；计算自适应推理；移动机器人。

## I. 引言

根据自然语言指令在陌生环境中完成导航，是室内服务机器人实现自然人机交互的重要能力。视觉语言导航（Vision-and-Language Navigation，VLN）要求智能体理解指令中的地标、方向和动作顺序，并将其与沿途视觉观测相对应，最终到达目标位置[1]。在连续环境中，智能体还需要通过实际运动逐步探索场景，并在局部观测下作出长程决策[2]。当关键地标被转角或障碍物遮挡时，当前观测可能不足以判断后续路径是否符合指令，错误的分支选择则会带来额外探索与回溯。

在线拓扑方法通过组织空间记忆来支持长程导航。航点预测器从当前观测中提出局部可通行位置[6]，拓扑图将这些候选与已访问位置及其连接关系逐步整合，使策略能够在图上选择高层目标，再由低层控制器完成运动[7], [9]。这种方式简化了连续导航的决策过程，但尚未访问候选的表示仍受到已有观测的限制。例如，对于“沿走廊前进，经过沙发后进入厨房”的指令，两条走廊入口可能具有相似外观，而沙发和厨房只有在进一步移动后才会显现。智能体虽然能够观察入口并记录其空间关系，却尚未获得到达候选位置后的视角。因此，对候选到达后及继续前进后的场景进行预测，有望为当前选择补充与后续指令相关的视觉线索。

已有研究开始利用未来预测辅助导航。Dreamwalker通过生成候选位置的全景图像进行前瞻规划[11]；HNR预测候选位置的语义特征与深度，并据此提出更远航点、构造未来路径树[8]；NavMorph则通过潜在动态建模和记忆更新支持前瞻决策与在线适应[12]。这些工作为候选未来表示和进一步展开提供了基础。本文在此基础上关注未来信息参与在线拓扑决策的组织方式：如何先利用到达预测补充候选表示，再将进一步预测集中于少量候选的比较，同时约束预测对原有评分的修正。在上述走廊场景中，到达预测可以补充入口内侧的视觉线索；若区分路径所需的地标仍在更远处，则需要继续提出后继查询。然而，预测视野的扩展也伴随着额外计算和误差累积，需要同时考虑未来证据的使用方式与进一步查询的必要性。

为此，本文提出PILOT，一种面向在线拓扑视觉语言导航的渐进式世界模型前瞻框架。PILOT使用冻结的表征世界模型，根据已观测历史和查询位姿预测未来视觉特征，并通过由广到深的两个阶段将其用于候选选择。第一阶段为多个有效候选预测到达后的视觉表征，并将预测结果融入候选表示，供基础拓扑策略完成初始评分。第二阶段对初始评分靠前的少量移动候选进一步查询：空间查询提议器直接读取候选的预测视觉表征，生成后继位置的方向与距离，再由世界模型预测该位置的视觉信息。候选评分模块结合指令与后继预测，对基础分数进行增量修正。由此，到达预测用于补充候选表示，后继预测用于辅助候选比较，两者分别作用于基础评分之前和之后，在不读取真实未来观测的条件下形成渐进式前瞻。

为限制不可靠预测的直接影响，PILOT仅对选中的候选进行有界分数修正，并在候选预测无效时保留其基础分数。修正上界还为选择性计算提供了依据：当允许范围内的任何修正都无法改变当前移动候选赢家时，即可省略第二阶段查询（触发机制待集成与验证）。结合少步世界模型推理，该设计同时控制单次预测的开销和深层查询的次数，使额外计算集中于仍可能影响候选选择的导航状态。

**图1. PILOT的渐进式前瞻思路。** 面对入口外观相似、关键地标尚不可见的候选路径，智能体首先预测各候选到达后的视觉表征，补充初始评分所需的信息；随后对少量高分候选提出后继查询，利用更远位置的预测辅助候选比较。两阶段分别承担候选表示增强和候选评分校正的作用（图待绘制）。

本文在R2R-CE和RxR-CE基准上评估PILOT，并通过模块、未来信息来源及计算预算的消融，分析两阶段前瞻和选择性计算的作用。实验表明，PILOT能够稳定改善导航性能（待实验验证），并通过选择性前瞻降低额外推理开销，在导航效果与计算成本之间取得更好的平衡（待实验验证）。

本文的主要贡献如下：

1. **双阶段候选前瞻。** 提出PILOT框架，将候选到达预测与后继预测分别用于基础评分前的表示增强和评分后的决策校正，形成由广到深的拓扑候选前瞻过程。
2. **表征驱动的后继查询。** 利用候选的预测视觉表征生成角度—距离查询，并将后继预测作为原候选的评分证据，将冻结表征世界模型的空间查询与在线候选比较相衔接。
3. **受约束的选择性计算。** 结合候选预算、有界分数修正与局部回退控制预测干预，并利用修正上界判断深层查询是否可能改变移动候选赢家，配合少步推理减少额外计算（触发机制待集成，计算收益待验证）。

---

## II. 相关工作

### A. 连续环境视觉语言导航与拓扑规划

视觉语言导航要求智能体理解自然语言指令，并在环境中到达指定目标。R2R[1]和RxR[3]为这一任务提供了室内导航基准，VLN-CE[2]进一步将预定义导航图上的视点选择扩展到连续环境中的运动执行。近年来，视觉语言大模型被用于学习从观测和指令到导航动作的映射。NaVid[14]利用单目RGB视频构建历史时空上下文，并结合指令生成下一步动作；StreamVLN[15]通过快更新的对话上下文与慢更新的历史记忆处理持续视觉输入，支持长序列下的流式导航。这些方法将大规模预训练模型的视觉语言能力引入导航，并围绕历史信息的保留与使用改进决策过程。

另一类研究围绕导航任务设计专用的跨模态策略，通过航点抽象和空间记忆支持指令跟随。其中，航点方法将连续运动转化为中间目标的选择，减轻直接处理长序列低层动作的负担。Krantz等[13]研究由语言和全景视觉共同引导的航点预测，并分析动作表达方式对导航效果与执行效率的影响。Hong等[6]进一步通过预测局部可通行候选，将面向高层动作设计的策略迁移到连续环境，使指令引导的目标选择与低层运动执行能够分开处理。

为支持超出局部观测范围的规划，显式地图被用于组织沿途获得的空间信息。DUET[16]在离散导航中结合全局拓扑图上的规划与局部观测中的细粒度语言理解；GridMM[17]将历史视觉特征整合到动态扩展的栅格记忆中，BEVBert[18]则通过局部度量地图与全局拓扑图的结合，兼顾局部空间理解和长程规划。在连续环境中，ETPNav[7]沿探索过程整合预测航点，在线构建包含已访问节点与待探索候选的拓扑图，并将跨模态目标选择与避障控制相衔接，使智能体能够利用逐步积累的空间结构进行规划与回溯。

在此基础上，后续研究进一步改进图结构与策略学习。DGNav[10]根据航点分布调整图构建阈值，并融合视觉、语言和几何信息更新连接权重；ETP-R1[9]则将扩展指令数据、R2R与RxR联合预训练，以及在线监督和强化微调引入拓扑策略学习。这些研究从空间表示和训练方式上完善了在线拓扑导航。

在线拓扑表示将长程导航组织为明确的候选目标选择，既支持基于空间记忆的规划与回溯，也为将未来预测与具体候选相对应提供了基础。然而，候选的位置和连接关系并不能直接揭示到达后的视觉内容；在转角或遮挡处，已有观测仍可能不足以判断后续路径是否符合指令[8]。本文沿用任务专用的在线拓扑策略，将候选到达与后继状态的预测分别用于表示增强和有界评分修正，为候选比较补充未来视觉证据。

### B. 世界模型与导航动态建模

世界模型通过学习环境随动作发生的变化，为智能体预测行动后果和选择后续行为提供依据。早期研究探索了如何从高维观测中学习适合规划的内部状态。PlaNet[19]结合确定性与随机性状态转移，在潜在空间中预测未来并优化动作序列；Dreamer[20]进一步利用想象得到的潜在轨迹学习策略，将环境预测与行为学习联系起来。MuZero[21]则围绕奖励、价值和策略学习用于搜索的隐式状态，表明世界模型也可以以决策所需的信息为建模目标，而不要求恢复完整观测。这些工作说明，状态表示及其学习目标需要与下游决策方式共同考虑。

未来观测生成提供了另一种利用预测的方式，使智能体能够通过合成的视觉过程评估行动或学习行为。IRIS[22]以离散图像编码和自回归Transformer构建模拟环境，DIAMOND[23]则采用扩散模型生成未来观测，并展示了视觉细节对行为学习的作用。在视觉语言导航中，Dreamwalker[11]通过生成候选位置的全景图像支持前瞻规划。面向更一般的视觉导航，NWM[24]将历史观测编码到变分自编码器的潜在空间，以导航平移、转角和时间跨度为条件，通过条件扩散Transformer预测未来视觉状态。该模型能够沿给定动作序列合成未来过程，用于轨迹规划与候选轨迹评估，将可控视觉生成与导航决策相衔接。

随着预训练视觉表征的发展，一些研究开始直接预测可供决策使用的未来特征。DINO-WM[25]在冻结的DINOv2空间中学习动作条件的patch特征预测，并通过优化动作序列使预测状态接近目标表征；V-JEPA 2[26]则在视频表征预训练后引入动作条件预测，形成用于机器人规划的V-JEPA 2-AC。在导航中，HNR[8]利用层次神经辐射表示预测候选视点的语义特征，NavMorph[12]结合潜在动态与上下文记忆支持前瞻决策。这些方法通过不同机制组织未来视觉信息，为直接在表征空间中进行环境推演和决策提供了依据。

稠密视觉表征也可以与生成式动态建模结合。表征自编码器RAE[27]将预训练视觉编码器与重建解码器配对，使语义丰富的视觉特征成为生成模型的工作空间。基于这一设计，RAE-NWM[28]将NWM的导航动态建模从压缩的VAE潜在空间转移到稠密DINOv2表征，以保留更丰富的空间结构。其条件生成网络结合历史表征、导航运动和预测时间跨度，通过流匹配学习生成过程的速度场，并经数值积分得到未来特征。预测状态可在表征空间中继续展开，下游规划直接使用这些特征，图像解码则用于可视化与像素层面的评估。由此，RAE-NWM在预训练视觉表征与动作条件生成之间建立了联系，为导航提供可直接使用的未来视觉证据。本文以冻结的RAE-NWM为前瞻基础，将其预测结果用于拓扑候选表示、后继查询和评分，研究表征世界模型与在线拓扑决策的结合。

### C. 前瞻规划与决策

将环境预测用于行动选择，需要进一步组织可能的未来并评估其与任务目标的关系。除直接优化动作序列外，预测也可以作为策略的附加输入，由策略学习如何利用未来信息。Imagination-Augmented Agents[29]将模型生成的想象轨迹编码后融入策略，使决策同时利用真实观测与预测上下文。在视觉语言导航中，Wang等[30]提出的Look Before You Leap利用环境模型预测后续状态与奖励，由前瞻策略递归产生动作，再将多条预测轨迹的编码与当前策略状态结合。这些研究为通过未来推演补充当前决策提供了基础。

面向连续环境，前瞻规划进一步与可通行航点及空间结构相结合。Dreamwalker[11]在生成的未来场景中进行蒙特卡洛树搜索，反复选择、扩展并评估行动分支，根据预测的目标接近程度选择当前航点。HNR[8]利用候选位置的预测深度提出后继航点，将已访问节点、候选节点和前瞻节点共同编码，并通过汇集分支内的目标评分并行比较未来路径。面向目标图像导航，NWM[24]既可以通过采样优化规划轨迹，也可以模拟外部策略提出的候选轨迹，根据预测终态与目标的视觉相似性进行重排。这些方法分别通过树搜索、图上分支评估和轨迹重排将未来预测转化为当前行动依据，说明预测信息的作用还取决于候选如何展开以及未来结果如何被评估。

前瞻的范围与深度也影响其实际价值。Dreamwalker[11]对搜索轮数和规划视野的分析表明，增加模拟会提高计算开销，而更长的预测还可能因误差累积而降低导航收益。围绕何时获取更多信息，Active VLN[31]学习探索的触发、方向和停止时机，并利用实际探索获得的观测更新导航决策；在模型内部计算层面，Hamrick等[32]通过元控制策略选择需要调用的预测模型及想象迭代次数，将任务损失与计算成本共同纳入优化。前者分配环境交互，后者分配内部推演，两者都表明额外信息的获取应与当前决策需要相联系。

本文将这一思路落实到拓扑候选的分阶段评估：首先利用到达预测增强候选表示，再依据基础评分筛选少量候选，由其预测状态提出后继空间查询，并将更远的未来证据用于有界评分修正。由此，未来信息分别在基础评分前后参与表示构建与候选比较。针对第二阶段的计算，本文进一步利用修正上界判断额外预测是否仍可能改变当前移动候选赢家，在允许的修正均无法改变该选择时省略深层查询（触发机制待集成）。这一设计将继续前瞻的条件与候选评分的可变范围直接联系起来。

---

## III. 方法

### A. 问题定义与基础导航框架

本文研究连续环境中的视觉语言导航任务[2]。智能体从未知室内环境中的给定位姿出发，根据自然语言指令 $I=(w_1,\ldots,w_L)$，通过前进、转向和停止等低层动作到达目标区域。本文以一次高层导航决策为时间步 $t$，将该时刻获得的全景RGB-D观测记为 $o_t$，供前瞻使用的近期观测历史记为 $H_t=(o_{\max(1,t-c+1)},\ldots,o_t)$，其中 $c$ 为上下文长度。导航过程中，智能体根据已获得的观测逐步建立空间记忆，在最大步数限制内到达指令指定的位置并停止，同时尽量减少不必要的移动。

本文采用ETP-R1[9]的拓扑导航框架，其在线建图与低层执行沿用ETPNav[7]的设计。每个决策时刻，航点预测器根据深度观测提出局部可通行位置，并依据空间关联将其纳入在线拓扑图 $G_t=(V_t,E_t)$。其中，$V_t=V_t^{\mathrm{vis}}\cup V_t^{\mathrm{ghost}}$ 包含已访问节点和尚未到达的候选节点，$E_t$ 表示节点间的连接关系。候选节点称为ghost节点，其基础视觉表示 $g_i$ 来自已访问位置对该候选方向的观测，而非候选到达后的真实视角。同一候选被重复检测时，图更新过程合并其位置估计并聚合对应视觉特征；本文的候选到达查询以更新后的均值位置为准。已访问节点保留沿途空间信息，候选节点则表示可供后续探索的目标。

高层策略结合语言指令、节点视觉表示与图中空间关系，对可执行候选和停止动作进行评分，动作集合记为 $\mathcal A_t=V_t^{\mathrm{ghost}}\cup\{\mathrm{STOP}\}$，最终动作分布记为 $\pi(a_t\mid I,H_t,G_t)$。选择移动候选后，规划器利用图中已有路径确定行进路线，再由低层控制器执行；已访问节点参与空间推理和路径回溯，不作为新的探索目标参与候选竞争。STOP用于结束导航，与移动候选分开处理。本文在这一决策接口上引入未来表征，后文将第一阶段表示增强之后、第二阶段残差校正之前的候选评分记为 $s_{t,i}$，称为基础评分，以区分随后由更远未来证据产生的分数修正。

### B. 方法总体框架

本文的前瞻框架围绕以下设计目标展开：

1. 不读取未执行候选的真实未来模拟器观测；
2. 未来证据必须与具体ghost候选一一对齐；
3. 世界模型和空间查询失败只能局部影响相应候选；
4. 候选分数修正具有显式上界；
5. 深层查询预算应随当前决策是否可改变而变化。

大写 $K$ 表示第二阶段深入处理的移动候选数量。小写 $k\geq1$ 表示从候选到达状态 $q_0$ 开始的空间查询展开步数：

$$
q_0\rightarrow q_1\rightarrow\cdots\rightarrow q_k.
$$

默认 $k=1$。因此，默认方法只执行一次 $q_0\rightarrow q_1$ 展开，但算法和公式采用一般 $k$ 步形式。

给出从输入到最终动作的完整公式：

$$
\{I,H_t,G_t\}
\rightarrow
\{\hat z_{0,i}\}_{i\in V_t^{\mathrm{ghost}}}
\rightarrow
\tilde G_t
\rightarrow
\mathbf{s}_t
\rightarrow
\mathcal C_t^{K}
\rightarrow
\{\hat z_{k,i}\}_{i\in\mathcal C_t^{K}}
\rightarrow
\boldsymbol\delta_t
\rightarrow
\tilde{\mathbf s}_t.
$$

其中，$\mathcal C_t^{K}$ 是基础策略选出的Top-$K$移动候选，$\delta_{t,i}$ 是有界候选残差。

### C. 候选到达预测与表示增强

对于候选 $i$，根据其拓扑位置和当前历史构造候选到达查询 $q_{0,i}$：

$$
\hat z_{0,i}=W(H_t,q_{0,i}),
$$

其中 $W$ 是冻结的动作条件表征世界模型。预测结果包含CLS和patch潜在表征。第一阶段的作用是增强候选表示，而不是直接输出动作：

$$
\tilde g_i=F_{\mathrm{fuse}}
\left(g_i,\hat z_{0,i},\gamma_i\right),
$$

其中 $g_i$ 是基础ghost表示，$\gamma_i$ 包含候选几何、一致性或有效性信息。

需要说明的实现边界：

- 只在历史上下文充分时执行预测；
- 缓存与ghost身份和最终均值位置绑定；
- 缓存缺失、过期或预测失败时回退到基础表示；
- 世界模型、DINOv2和查询提议器保持冻结。

**图2图注（需要生成的候选几何与缓存示意图）：**  
第一阶段候选到达状态 $q_0$ 的构造和缓存语义。图片应包含当前机器人位姿、在线拓扑节点、多个ghost候选、每个候选的最终均值位置、局部坐标到世界坐标的转换，以及同一ghost重复观测后的缓存更新/失效。用不同颜色区分已访问节点、有效ghost、无效缓存和候选到达查询。图中应明确没有读取候选位置的真实RGB观测。

### D. 预测状态引导的后继查询与候选校正

#### 1) 基础策略评分与候选选择

增强后的图表示输入基础拓扑策略：

$$
\mathbf{s}_t=P_{\mathrm{base}}(I,H_t,\tilde G_t).
$$

从所有可执行移动ghost中进行稳定排序，得到候选集合：

$$
\mathcal C_t^K=\operatorname{StableTopK}
\left(\{s_{t,i}\mid i\in V_t^{\mathrm{ghost}}\},K\right).
$$

并列时沿用基础规划器的候选顺序。STOP不属于 $\mathcal C_t^K$。

#### 2) 预测状态驱动的空间查询

对每个 $i\in\mathcal C_t^K$，以第一阶段预测状态为起点：

$$
\hat z_{0,i}=W(H_t,q_{0,i}).
$$

对 $r=0,\ldots,k-1$ 递推：

$$
\Delta p_{r+1,i}=Q(\hat z_{r,i}),
$$

$$
q_{r+1,i}=T(q_{r,i},\Delta p_{r+1,i}),
$$

$$
\hat z_{r+1,i}=W(H_t,q_{r+1,i}).
$$

其中 $Q$ 是读取预测表征的空间查询提议器，输出局部角度-距离位移；$T$ 将局部位移累积到查询位姿。历史上下文 $H_t$ 在同一候选展开中保持固定，不向世界模型提供真实未来帧。

默认 $k=1$，终端未来表征为 $\hat z_{1,i}$。当 $k>1$ 时，前一预测状态继续产生下一查询，直至 $\hat z_{k,i}$。

**图3图注（需要生成的 $k$ 步预测查询链示意图）：**  
预测状态引导的空间查询递推。图片应以一个Top-$K$候选为例，显示 $q_0,q_1,\ldots,q_k$ 的局部位姿、每一步的预测CLS+patch表征、查询提议器输出的角度和距离，以及位姿累积关系。实线突出默认 $k=1$，浅色重复单元表示 $k=2,3$。需要明确所有查询均由预测状态产生，不读取真实未来观测。

#### 3) 终端未来表征与候选残差评分

候选残差评分头读取指令、候选拥有者表示、基础分数、候选几何和终端未来表征：

$$
u_{t,i}=F_{\mathrm{score}}
\left(
x_{t,i}^{\mathrm{owner}},
X_t^{\mathrm{text}},
\hat z_{k,i},
s_{t,i},
\phi_{t,i}
\right).
$$

使用双曲正切限制残差：

$$
\delta_{t,i}=
\Delta_{\max}\tanh(u_{t,i}),
\qquad
|\delta_{t,i}|\leq\Delta_{\max}.
$$

最终分数为：

$$
\tilde s_{t,i}=
\begin{cases}
s_{t,i}+\delta_{t,i}, & i\in\mathcal C_t^K
\text{且未来表征有效},\\
s_{t,i}, & \text{其他情况}.
\end{cases}
$$

普通SFT路径中，STOP分数不直接增加残差，但移动候选分数变化仍可能间接改变普通softmax下的STOP概率，因此不能表述为“STOP完全不受影响”。

### E. 前瞻干预与计算约束

第二阶段的候选预算限制深入处理的范围，有界残差限制评分修正的幅度。在此基础上，候选级回退处理无效预测，决策不变性条件用于判断是否需要继续查询，积分步数则决定单次预测的计算成本。

#### 1) 候选级回退

定义有效掩码 $m_{t,i}\in\{0,1\}$。当候选历史不足、缓存无效、查询提议失败或世界模型预测失败时：

$$
m_{t,i}=0,\qquad \delta_{t,i}=0.
$$

该候选恢复基础分数，其他有效候选继续计算。若一行所有深层候选都无效，则整个深层分支恢复基础动作分布。

#### 2) 决策不变性触发

设基础移动赢家为 $w$。对每个可执行移动候选定义残差上界：

$$
b_i=
\begin{cases}
\Delta_{\max}, & i\in\mathcal C_t^K,\\
0, & i\notin\mathcal C_t^K.
\end{cases}
$$

如果

$$
s_{t,w}-b_w>
\max_{j\neq w}(s_{t,j}+b_j),
$$

则任何合法残差都无法改变当前移动赢家，可以跳过从 $q_0$ 到 $q_k$ 的第二阶段深层查询链。基础STOP获胜时，同样不启动只负责移动候选重排的查询链。

需要明确：

- 该证明只保证动作赢家在合法残差范围内不变；
- 它不保证导航性能不下降；
- 它不证明第一阶段 $q_0$ 表示增强可以跳过；
- 严格不等号用于避免并列规则变化。

**图4图注（需要生成的决策不变性触发示意图）：**  
基于有界残差的选择性前瞻。图片应以候选分数区间展示基础赢家的最坏下界 $s_w-b_w$ 和竞争者的最好上界 $s_j+b_j$。左侧展示区间不相交、可以跳过深层查询的情况；右侧展示区间重叠、必须执行Top-$K$查询链的情况。另设STOP分支。图注应强调这是动作不变性证明，不是性能安全保证。

#### 3) 少步欧拉推理与复杂度

世界模型使用欧拉积分完成潜在表征预测。设每次世界模型调用使用 $n$ 个积分步，第二阶段处理 $K$ 个候选并展开 $k$ 步，则其主要世界模型调用成本与 $Kk$ 成正比：

$$
\mathcal O(Kk\,C_W(n)),
$$

其中 $C_W(n)$ 是一次 $n$ 步世界模型推理成本。默认设置为：

$$
K=____,\qquad k=1,\qquad n=10.
$$

50、10和4积分步用于采样步数消融，不应称为蒸馏。

---

## IV. 训练方法

### A. 训练阶段总览

训练分为：

1. 联合预训练；
2. 在线SFT；
3. 前瞻模块联合SFT；
4. 冻结前瞻模块的GRPO。

旧ETP-R1已经包含联合预训练、SFT和GRPO。本文重点说明新增前瞻模块在SFT和GRPO之间的参数与数据契约。

**图5图注（需要生成的训练阶段与冻结边界图）：**  
TopoForesight训练流程。图片应按时间从左到右显示预训练、基础SFT、前瞻联合SFT和冻结式GRPO。对每一阶段标出导航策略、DINOv2、表征世界模型、空间查询提议器、CLS适配器、候选残差评分头的“冻结/训练”状态。GRPO部分应显示rollout缓存候选索引、有效掩码、动作对齐残差和旧策略概率，更新阶段不重新选择Top-$K$。

### B. 前瞻联合SFT

基础导航损失为：

$$
\mathcal L_{\mathrm{base}}
=-\log \pi_{\mathrm{base}}(a_t^\star).
$$

使用未来残差后的调整分布损失为：

$$
\mathcal L_{\mathrm{future}}
=-\log \pi_{\mathrm{future}}(a_t^\star).
$$

总损失写为：

$$
\mathcal L_{\mathrm{SFT}}
=\mathcal L_{\mathrm{base}}
+\lambda_{\mathrm{future}}
\mathcal L_{\mathrm{future}}
+\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}.
$$

需要在正式稿中给出参数更新表，明确哪些输入在进入未来评分头前停止梯度，以及基础策略、融合层、CLS适配器和残差头分别接受哪一项损失。

### C. 冻结前瞻模块的GRPO

在rollout时计算并缓存：

- Top-$K$候选索引；
- 候选有效掩码；
- 动作对齐残差向量；
- 行为策略旧概率；
- 必要的STOP概率质量信息。

更新时current、old和reference策略复用同一份候选索引、掩码和残差，避免动态Top-$K$变化污染概率比和KL项。GRPO只更新基础导航策略，世界模型、查询提议器、融合层、适配器和候选残差评分头保持冻结。

GRPO目标写为：

$$
\mathcal L_{\mathrm{GRPO}}
=-\mathbb E
\left[
\min
\left(
r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
-\beta D_{\mathrm{KL}}
\right],
$$

其中具体奖励、优势归一化和STOP处理在正式稿中补充。

### D. 保存、恢复和可复现性

正式稿应说明：

- 模型和训练状态的保存格式；
- 优化器、调度器、混合精度缩放器和随机状态；
- 每个rank的世界模型随机生成器；
- 基础checkpoint、NWM、DINO-CWP和残差头的SHA-256来源；
- 恢复时拒绝不兼容状态的条件。

这些内容作为可复现性支撑，不单独包装成科学创新。

---

## V. 实验

### A. 研究问题

实验围绕以下问题组织：

1. RQ1：双阶段前瞻是否优于对应基础拓扑策略？
2. RQ2：第一阶段 $q_0$ 和第二阶段终端 $q_k$ 是否分别必要？
3. RQ3：真实未来、预测未来和无未来之间存在多大差距？
4. RQ4：展开步数 $k=1,2,3$ 如何影响导航性能、误差和计算成本？
5. RQ5：候选预算 $K$、预测有效率和误差怎样影响动作修正？
6. RQ6：决策不变性触发和少步欧拉推理能减少多少开销？
7. RQ7：冻结式GRPO能否在保留前瞻证据的同时进一步优化闭环行为？

### B. 数据集与评测协议

#### 1) R2R-CE

训练划分：____  
验证划分：____  
最终评测episode数：____  
语言：____  
控制器：____

#### 2) RxR-CE

训练划分：____  
验证划分：____  
最终评测episode数：____  
语言：____  
控制器：____

#### 3) 指标

报告NE、OSR、SR、SPL、nDTW和SDTW。所有比例指标统一使用百分数或0至1尺度，表内不得混用。另报告平均高层动作数、路径长度、碰撞、端到端决策延迟和峰值显存。

### C. 实现细节

| 项目 | R2R-CE | RxR-CE |
|---|---:|---:|
| 训练GPU | ____ | ____ |
| 环境数/每卡 | ____ | ____ |
| 全局batch | ____ | ____ |
| 学习率 | ____ | ____ |
| SFT更新次数 | ____ | ____ |
| GRPO更新次数 | ____ | ____ |
| 候选预算 $K$ | ____ | ____ |
| 默认展开步数 $k$ | 1 | 1 |
| 世界模型积分步数 | 10 | 10 |
| 历史上下文长度 | ____ | ____ |
| 残差上界 $\Delta_{\max}$ | ____ | ____ |

正式稿补充Python、PyTorch、Transformers、CUDA、Habitat和Habitat-Sim版本。

### D. 对比方法

按以下组别组织：

1. 经典与近期VLN-CE方法；
2. 图式/拓扑式方法；
3. 世界模型或前瞻式导航方法；
4. 本文统一现代运行时下的直接基础策略；
5. 本文各阶段变体。

旧论文报告值与当前同协议结果必须分开标注，不能直接求差。

### E. 主结果

**表I：R2R-CE同协议主结果。**

| 方法 | 未来前瞻 | GRPO | NE↓ | OSR↑ | SR↑ | SPL↑ | nDTW↑ | SDTW↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 基础拓扑策略 | 否 | 否 | ____ | ____ | ____ | ____ | ____ | ____ |
| TopoForesight-SFT（$k=1$） | 是 | 否 | ____ | ____ | ____ | ____ | ____ | ____ |
| TopoForesight-GRPO（$k=1$） | 是 | 是 | ____ | ____ | ____ | ____ | ____ | ____ |

正文结果模板：

> 在R2R-CE上，TopoForesight-SFT相较基础策略的SR和SPL分别变化____和____。冻结式GRPO进一步使____变化____。结合____指标可以看出____。

**表II：RxR-CE同协议主结果。**

| 方法 | 未来前瞻 | GRPO | NE↓ | OSR↑ | SR↑ | SPL↑ | nDTW↑ | SDTW↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 基础拓扑策略 | 否 | 否 | ____ | ____ | ____ | ____ | ____ | ____ |
| TopoForesight-SFT（$k=1$） | 是 | 否 | ____ | ____ | ____ | ____ | ____ | ____ |
| TopoForesight-GRPO（$k=1$） | 是 | 是 | ____ | ____ | ____ | ____ | ____ | ____ |

正文结果模板：

> 在语言更长、场景覆盖更广的RxR-CE上，完整方法在____指标上取得____，表明____。与此同时，____指标的变化说明____。

### F. 双阶段模块消融

**表III：双阶段前瞻消融。**

| 第一阶段 $q_0$ 融合 | 第二阶段 $q_k$ 重排 | 参数匹配控制 | SR↑ | SPL↑ | SDTW↑ | 动作翻转率 | 延迟 |
|---|---|---|---:|---:|---:|---:|---:|
| 否 | 否 | 否 | ____ | ____ | ____ | ____ | ____ |
| 是 | 否 | 否 | ____ | ____ | ____ | ____ | ____ |
| 否 | 是 | 否 | ____ | ____ | ____ | ____ | ____ |
| 是 | 是 | 否 | ____ | ____ | ____ | ____ | ____ |
| 是 | 随机/零未来 | 是 | ____ | ____ | ____ | ____ | ____ |

需要回答：

- 第一阶段是否改善候选表示；
- 第二阶段是否提供独立决策价值；
- 完整收益是否只是来自额外参数；
- 两阶段是否存在互补关系。

### G. 未来来源与预测质量

**表IV：默认 $k=1$ 下的未来来源对照。**

| $q_1$ 来源 | 是否可部署 | CLS误差↓ | patch误差↓ | future-valid↑ | SR↑ | SPL↑ | 延迟 |
|---|---|---:|---:|---:|---:|---:|---:|
| 无 $q_1$ | 是 | — | — | ____ | ____ | ____ | ____ |
| 真实渲染 $q_1$ | 否，诊断用 | 0 | 0 | ____ | ____ | ____ | ____ |
| 预测 $q_1$ | 是 | ____ | ____ | ____ | ____ | ____ | ____ |

真实渲染只作为诊断输入，不称为可部署结果或严格性能上界。正式稿需说明如何克隆和恢复模拟器状态，确保不会污染真实导航轨迹。

### H. 展开步数 $k$ 消融

**表V：前瞻展开步数消融。**

| $k$ | 终端状态 | 累计查询误差↓ | future-valid↑ | SR↑ | SPL↑ | 单决策延迟↓ | 峰值显存↓ |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1（默认） | $q_1$ | ____ | ____ | ____ | ____ | ____ | ____ |
| 2 | $q_2$ | ____ | ____ | ____ | ____ | ____ | ____ |
| 3 | $q_3$ | ____ | ____ | ____ | ____ | ____ | ____ |

正文结果模板：

> 从 $k=1$ 增加到 $k=2$ 时，____；继续增加到 $k=3$ 后，____。这表明更远未来信息与预测误差之间呈现____关系。综合导航指标和延迟，默认选择 $k=1$ 的原因是____。

**图6图注（需要生成的展开步数性能-误差-成本图）：**  
不同展开步数 $k=1,2,3$ 的联合比较。建议采用三联图：左图为SR/SPL，中央为CLS、patch和累计空间查询误差，右图为端到端延迟与峰值显存。每个点应带置信区间或跨种子误差条。突出默认 $k=1$，但不得用视觉样式预先暗示它一定最优。

### I. 候选预算与上下文消融

**表VI：候选预算 $K$ 和历史上下文长度。**

| $K$ | 上下文长度 | future-valid↑ | Top-$K$教师覆盖率↑ | SR↑ | SPL↑ | 延迟↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | ____ | ____ | ____ | ____ | ____ | ____ |
| 3 | ____ | ____ | ____ | ____ | ____ | ____ |
| 5 | ____ | ____ | ____ | ____ | ____ | ____ |
| 10 | ____ | ____ | ____ | ____ | ____ | ____ |

需要注意：本实验中的大写 $K$ 与展开步数小写 $k$ 必须在图表标题和正文中明确区分。

### J. 计算自适应前瞻

#### 1) 世界模型积分步数

**表VII：50、10和4步欧拉推理。**

| 积分步数 | 终端CLS差异↓ | 查询角度差异↓ | 动作一致率↑ | SR↑ | SPL↑ | NWM延迟↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | ____ | ____ | ____ | ____ | ____ | ____ |
| 10（默认） | ____ | ____ | ____ | ____ | ____ | ____ |
| 4 | ____ | ____ | ____ | ____ | ____ | ____ |

#### 2) 决策不变性触发

**表VIII：按需触发的性能与成本。**

| 设置 | 触发率 | 跳过率 | 不变性违反次数 | SR↑ | SPL↑ | 平均决策延迟↓ | 每episode NWM调用↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 始终执行深层查询 | ____ | 0 | — | ____ | ____ | ____ | ____ |
| 决策不变性触发 | ____ | ____ | 0/____ | ____ | ____ | ____ | ____ |

**图7图注（需要生成的性能-成本帕累托图）：**  
TopoForesight不同计算配置的性能-成本关系。横轴为端到端高层决策延迟或每episode世界模型调用次数，纵轴分别显示SR和SPL。点应包含不同 $K$、$k$、积分步数和是否启用决策不变性触发。标出基础策略、默认 $k=1$ 完整方法和最省计算配置。

### K. 动作翻转与行为分析

把每个episode按首次动作分歧分为：

1. 基础失败、前瞻成功；
2. 基础成功、前瞻失败；
3. 两者结果相同但路径效率改变；
4. 从未翻转动作。

**表IX：动作翻转的修正与伤害。**

| 类型 | episode数 | 占比 | 平均分数间隔 | 平均残差 | future-valid | 结果变化 |
|---|---:|---:|---:|---:|---:|---:|
| 失败变成功 | ____ | ____ | ____ | ____ | ____ | ____ |
| 成功变失败 | ____ | ____ | ____ | ____ | ____ | ____ |
| 结果不变 | ____ | ____ | ____ | ____ | ____ | ____ |
| 未翻转 | ____ | ____ | ____ | ____ | ____ | ____ |

动作翻转与最终结果是配对关联，不能仅凭一次分歧声称严格因果。

### L. 定性结果与失败案例

**图8图注（需要生成的成功、失败和回退案例图）：**  
至少展示四类代表性episode：一是基础策略在相似走廊间犹豫、前瞻修正成功；二是世界模型预测错误导致候选伤害；三是future-valid不足而触发候选级回退；四是决策不变性触发跳过深层查询。每个案例应包含自然语言指令、简化拓扑图、候选基础分数、终端 $q_k$ 查询位置、残差、最终动作和轨迹结果。真实观测与预测表征可用检索出的近邻图像或降维热图表示，但必须明确标注，避免把预测表征伪装成真实RGB。

### M. 鲁棒性与部署分析

根据时间和资源选择以下实验：

- RGB噪声、深度噪声和朝向误差；
- 历史上下文缺帧；
- 候选位置扰动；
- 场景长度和指令长度分组；
- R2R与RxR不同语言分组；
- 计算资源变化下的吞吐。

若没有真实机器人实验，应在本节明确说明，并避免声称实时真实机器人部署能力。

---

## VI. 讨论、局限与未来工作

### A. 为什么候选级未来表征有意义

讨论表征世界模型与完整RGB生成的区别：本文不要求生成可视视频，而只要求产生能支持候选比较的潜在证据。分析这种设计在计算、语义抽象和误差可解释性上的利弊。

### B. 展开步数 $k$ 的边界

讨论 $k$ 增大后可能出现的位姿误差、预测误差和有效率下降。强调 $k=1$ 是默认完整方法，而 $k=2,3$ 消融用于揭示递归前瞻的适用边界，不预设“看得越远越好”。

### C. 计算自适应不等于安全保证

决策不变性触发仅保证有界候选残差不能改变当前赢家。它不验证地图正确性、世界模型真实性、底层碰撞安全或最终任务成功。

### D. 模拟器与真实机器人差距

讨论Matterport3D静态场景、传感器模型、位姿精度、动态人群和真实计算平台差异。说明未来需要真实机器人、跨建筑和动态障碍评测。

### E. 数据与模型依赖

讨论预训练数据、世界模型训练域、DINOv2表征、语言分布和计算资源依赖，以及对未见物体、长指令和多语言场景的潜在影响。

---

## VII. 结论

本文提出TopoForesight，一种面向在线拓扑视觉语言导航的双阶段表征世界模型前瞻框架。第一阶段预测候选到达状态 $q_0$ 并增强ghost表示；第二阶段从Top-$K$候选出发，由预测状态递归生成空间查询并前瞻至 $q_k$，默认 $k=1$。候选残差评分头以有界方式校正候选排序，候选级回退和决策不变性触发进一步限制预测干预与深层计算。R2R-CE和RxR-CE实验表明____。对 $k=1,2,3$、候选预算、未来来源和推理预算的分析进一步说明____。未来将研究真实机器人部署、动态环境和更稳健的长视野预测。

---

## 附录建议

### 附录A：决策不变性证明

给出候选级残差区间、严格不等号、稳定并列规则、STOP情况和单一可执行候选情况的完整证明。

### 附录B：网络结构与训练超参数

列出融合层、CLS适配器、候选残差评分头的维度、参数量、初始化、优化器、学习率和梯度裁剪。

### 附录C：评测协议和结果来源

列出每个表格行对应的配置、提交、checkpoint、原始JSON路径、控制器、随机种子和选择规则。

### 附录D：额外定性案例

增加不同场景类型、不同指令长度和不同语言下的成功、失败及回退案例。

---

## 参考文献

[1] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould, and A. van den Hengel, “Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 3674–3683, doi: 10.1109/CVPR.2018.00387.

[2] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 104–120, doi: 10.1007/978-3-030-58604-1_7.

[3] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding,” in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2020, pp. 4392–4412, doi: 10.18653/v1/2020.emnlp-main.356.

[4] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Nießner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3D: Learning from RGB-D Data in Indoor Environments,” in Proc. Int. Conf. 3D Vision (3DV), 2017, pp. 667–676, doi: 10.1109/3DV.2017.00081.

[5] M. Savva et al., “Habitat: A Platform for Embodied AI Research,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 9338–9346, doi: 10.1109/ICCV.2019.00943.

[6] Y. Hong, Z. Wang, Q. Wu, and S. Gould, “Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 15418–15428, doi: 10.1109/CVPR52688.2022.01500.

[7] D. An, H. Wang, W. Wang, Z. Wang, Y. Huang, K. He, and L. Wang, “ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 5130–5145, 2025, doi: 10.1109/TPAMI.2024.3386695.

[8] Z. Wang, X. Li, J. Yang, Y. Liu, J. Hu, M. Jiang, and S. Jiang, “Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 13753–13762, doi: 10.1109/CVPR52733.2024.01305.

[9] S. Ye, S. Mao, Y. Cui, X. Yu, S. Zhai, W. Chen, S. Zhou, R. Xiong, and Y. Wang, “ETP-R1: Evolving Topological Planning with Reinforcement Fine-Tuning for Vision-Language Navigation in Continuous Environments,” arXiv:2512.20940, 2025.

[10] J. Peng, J. Guo, Y. Xu, Y. Liu, J. Yan, X. Ye, H. Li, and X. Wang, “Dynamic Topology Awareness: Breaking the Granularity Rigidity in Vision-Language Navigation,” arXiv:2601.21751, 2026.

[11] H. Wang, W. Liang, L. Van Gool, and W. Wang, “Dreamwalker: Mental Planning for Continuous Vision-Language Navigation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 10839–10849, doi: 10.1109/ICCV51070.2023.00998.

[12] X. Yao, J. Gao, and C. Xu, “NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 5536–5546, doi: 10.1109/ICCV51701.2025.00525.

[13] J. Krantz, A. Gokaslan, D. Batra, S. Lee, and O. Maksymets, “Waypoint Models for Instruction-Guided Navigation in Continuous Environments,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 15162–15171. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2021/html/Krantz_Waypoint_Models_for_Instruction-Guided_Navigation_in_Continuous_Environments_ICCV_2021_paper.html

[14] J. Zhang, K. Wang, R. Xu, G. Zhou, Y. Hong, X. Fang, Q. Wu, Z. Zhang, and H. Wang, “NaVid: Video-based VLM Plans the Next Step for Vision-and-Language Navigation,” arXiv:2402.15852, 2024. [Online]. Available: https://arxiv.org/abs/2402.15852

[15] M. Wei et al., “StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling,” arXiv:2507.05240v2, 2026. [Online]. Available: https://arxiv.org/abs/2507.05240v2

[16] S. Chen, P.-L. Guhur, M. Tapaswi, C. Schmid, and I. Laptev, “Think Global, Act Local: Dual-Scale Graph Transformer for Vision-and-Language Navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 16537–16547. [Online]. Available: https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Think_Global_Act_Local_Dual-Scale_Graph_Transformer_for_Vision-and-Language_Navigation_CVPR_2022_paper.html

[17] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “GridMM: Grid Memory Map for Vision-and-Language Navigation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 15625–15636. [Online]. Available: https://openaccess.thecvf.com/content/ICCV2023/html/Wang_GridMM_Grid_Memory_Map_for_Vision-and-Language_Navigation_ICCV_2023_paper.html

[18] D. An, Y. Qi, Y. Li, Y. Huang, L. Wang, T. Tan, and J. Shao, “BEVBert: Multimodal Map Pre-training for Language-guided Navigation,” arXiv:2212.04385v2, 2023. [Online]. Available: https://arxiv.org/abs/2212.04385v2

[19] D. Hafner et al., “Learning Latent Dynamics for Planning from Pixels,” arXiv:1811.04551, 2018. [Online]. Available: https://arxiv.org/abs/1811.04551

[20] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to Control: Learning Behaviors by Latent Imagination,” arXiv:1912.01603, 2019. [Online]. Available: https://arxiv.org/abs/1912.01603

[21] J. Schrittwieser et al., “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model,” arXiv:1911.08265, 2019. [Online]. Available: https://arxiv.org/abs/1911.08265

[22] V. Micheli, E. Alonso, and F. Fleuret, “Transformers are Sample-Efficient World Models,” arXiv:2209.00588, 2022. [Online]. Available: https://arxiv.org/abs/2209.00588

[23] E. Alonso et al., “Diffusion for World Modeling: Visual Details Matter in Atari,” arXiv:2405.12399, 2024. [Online]. Available: https://arxiv.org/abs/2405.12399

[24] A. Bar, G. Zhou, D. Tran, T. Darrell, and Y. LeCun, “Navigation World Models,” arXiv:2412.03572, 2024. [Online]. Available: https://arxiv.org/abs/2412.03572

[25] G. Zhou, H. Pan, Y. LeCun, and L. Pinto, “DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning,” arXiv:2411.04983, 2024. [Online]. Available: https://arxiv.org/abs/2411.04983

[26] M. Assran et al., “V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning,” arXiv:2506.09985, 2025. [Online]. Available: https://arxiv.org/abs/2506.09985

[27] B. Zheng, N. Ma, S. Tong, and S. Xie, “Diffusion Transformers with Representation Autoencoders,” arXiv:2510.11690, 2025. [Online]. Available: https://arxiv.org/abs/2510.11690

[28] M. Zhang, W. Shen, F. Zhang, H. Qin, Z. Pei, and Z. Meng, “RAE-NWM: Navigation World Model in Dense Visual Representation Space,” arXiv:2603.09241, 2026. [Online]. Available: https://arxiv.org/abs/2603.09241

[29] T. Weber et al., “Imagination-Augmented Agents for Deep Reinforcement Learning,” arXiv:1707.06203, 2017. [Online]. Available: https://arxiv.org/abs/1707.06203

[30] X. Wang, W. Xiong, H. Wang, and W. Y. Wang, “Look Before You Leap: Bridging Model-Free and Model-Based Reinforcement Learning for Planned-Ahead Vision-and-Language Navigation,” arXiv:1803.07729, 2018. [Online]. Available: https://arxiv.org/abs/1803.07729

[31] H. Wang, W. Wang, T. Shu, W. Liang, and J. Shen, “Active Visual Information Gathering for Vision-Language Navigation,” arXiv:2007.08037, 2020. [Online]. Available: https://arxiv.org/abs/2007.08037

[32] J. B. Hamrick, A. J. Ballard, R. Pascanu, O. Vinyals, N. Heess, and P. W. Battaglia, “Metacontrol for Adaptive Imagination-Based Optimization,” arXiv:1705.02670, 2017. [Online]. Available: https://arxiv.org/abs/1705.02670
