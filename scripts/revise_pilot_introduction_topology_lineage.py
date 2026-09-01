#!/usr/bin/env python3
"""Revise PILOT's introduction around the topological VLN lineage."""

from __future__ import annotations

from docx import Document

from update_pilot_introduction_docx import (
    DOCUMENT_PATH,
    find_paragraph,
    format_reference,
    paragraph_index,
    remove_paragraph,
    replace_text_preserve_run,
)


INTRO_PARAGRAPHS = [
    (
        "使机器人能够依据自然语言完成长距离移动，是服务机器人从执行预设目标走向自然人机交互的"
        "重要能力。视觉语言导航（Vision-and-Language Navigation，VLN）要求智能体将指令中的地标、"
        "方向、空间关系和动作顺序与第一视角观测持续对齐，并在未见环境中到达指定位置。R2R在真实"
        "室内扫描场景上建立了这一任务的经典基准[1]；VLN-CE进一步移除离散导航图等理想化条件，使"
        "智能体必须在连续三维环境中将高层决策落实为可执行运动[2]；RxR则引入多语言指令和更密集的"
        "时空对齐，进一步提高了长程指令跟随的复杂性[3]。Matterport3D和Habitat分别提供真实室内"
        "场景数据与可复现的具身仿真平台[4], [5]。在这种连续、部分可观测的设置中，一次错误的高层"
        "分支选择往往会引入额外运动、回溯和误差累积，因此导航策略既要理解语言，也要形成可靠的"
        "长程空间判断。"
    ),
    (
        "围绕VLN-CE中的长时程决策，现有方法大体形成了端到端策略与模块化规划两类技术路线。端到端"
        "方法直接根据视觉和语言输入预测低层动作或连续轨迹；模块化方法则将导航分解为可通行位置预测、"
        "高层目标选择和低层运动执行。本文沿用后一条路线，关注以在线拓扑图为核心的高层导航策略。"
        "相比直接在冗长的低层动作序列上进行决策，模块化拓扑方法把连续环境抽象为已访问位置、候选"
        "位置及其连接关系，使智能体能够在图上完成长程规划和回溯，再由控制器将选定的高层目标转换为"
        "实际运动。"
    ),
    (
        "这条技术路线经历了从局部航点到在线拓扑规划的持续发展。CWP利用航点预测器将连续导航由低层"
        "运动决策提升为可通行航点选择，从而缩小策略的动作空间[6]；ETPNav进一步将导航过程中产生的"
        "航点组织为在线演化的拓扑图，通过已访问节点和ghost候选支持全局规划、历史记忆与错误回溯"
        "[7]；HNR在ETPNav框架上预测候选位置的多层未来语义表征，并构建未来路径树以并行比较可能的"
        "导航分支[8]；ETP-R1则从数据规模和训练范式入手，通过联合预训练、在线监督微调和闭环强化"
        "微调增强图式策略[9]；近期的DGNav进一步根据场景复杂度和指令语义动态调整拓扑粒度与连接"
        "关系[10]。这些工作表明，在线拓扑图不仅是一种环境记忆形式，也是连接语言理解、长程规划和"
        "连续运动执行的结构化决策接口。"
    ),
    (
        "在上述进展中，未来环境预测已经显示出改善导航决策的潜力。除HNR的候选未来表征与路径树外，"
        "Dreamwalker通过生成候选位置的未来全景图像进行心理规划[11]，NavMorph则利用潜在世界模型"
        "支持环境理解和前瞻规划[12]。因此，本文并不把“为导航预测未来”作为尚未被研究的问题。对于"
        "ETPNav式在线拓扑策略，更具体的问题在于：如何把未来预测组织成一个与原有ghost动作空间直接"
        "对齐、由广到深且受计算预算约束的决策过程。较广泛候选的到达状态预测需要在基础评分前改善"
        "候选表示，而更深入的未来推演应集中在少量难以区分的关键候选上；与此同时，后续查询位置不能"
        "依赖未执行分支的真实观测，预测误差也不应无界地改写基础策略。换言之，问题不再是“是否需要"
        "未来信息”，而是“未来信息应当以何种层次进入拓扑候选，以及何时值得为更深前瞻付出计算”。"
    ),
    (
        "为此，本文提出PILOT，一种面向在线拓扑视觉语言导航的渐进式世界模型前瞻框架。PILOT使用"
        "冻结的表征世界模型，在机器人执行动作之前生成与拓扑候选对齐的未来视觉表征。第一阶段面向"
        "较广泛的有效ghost候选预测其到达状态q_0，并在基础策略评分之前增强候选表示；第二阶段从基础"
        "策略选出的Top-K移动候选出发，由预测状态递归产生空间查询并前瞻至q_k，再利用终端未来表征"
        "对候选排序进行有界修正。该过程形成从候选表示增强到候选决策校正的由广到深前瞻。本文采用"
        "一般的k步形式统一描述这一过程，默认k=1，即一次q_0→q_1展开；该设置已经构成完整的候选预测、"
        "空间查询和决策校正闭环。"
    ),
    (
        "考虑到世界模型预测并非始终可靠，且深入查询会带来额外开销，PILOT进一步采用受约束的计算"
        "自适应设计。深层前瞻仅作用于有限数量的高价值候选，单个候选预测失败时进行局部回退，并通过"
        "有界残差限制未来表征对基础策略的直接影响。与此同时，系统使用少步推理降低单次预测成本，"
        "并根据残差上界判断深层前瞻是否可能改变当前候选赢家；若任何合法修正都无法改变决策，则跳过"
        "第二阶段查询并保留基础动作。因此，PILOT并不以世界模型替代原有拓扑策略，而是将其作为一种"
        "候选级、按需使用的未来证据来源。"
    ),
]


FIGURE_CAPTION = (
    "图1. PILOT总体框架。给定语言指令、历史视觉观测和在线拓扑图，第一阶段预测各有效ghost候选的"
    "到达状态q_0并增强其表示；基础策略完成初始评分后，第二阶段仅对Top-K移动候选从q_0递归前瞻"
    "至q_k，并利用终端未来表征生成有界残差以重排候选。默认k=1；当有界残差不可能改变当前决策时，"
    "跳过第二阶段深层查询。待绘制图片应区分基础拓扑路径、冻结模块与可训练模块，并标明候选级回退"
    "和深层查询的执行/跳过分支。"
)


EVALUATION_PARAGRAPH = (
    "本文在R2R-CE和RxR-CE基准上对PILOT进行系统评估。除与对应基础拓扑策略进行比较外，实验围绕"
    "未来信息来源、展开步数k、候选预算K、预测质量、候选覆盖率以及端到端延迟等方面展开，以考察"
    "未来信息的决策价值及其与计算成本之间的关系。"
)


CONTRIBUTIONS = [
    (
        "提出PILOT，一种面向在线拓扑候选的渐进式世界模型前瞻框架。该框架先为较广泛的ghost候选"
        "预测到达状态，再对少量高价值候选进行深入前瞻，实现从候选表示增强到候选决策校正的由广到"
        "深处理。"
    ),
    (
        "提出预测状态引导的空间查询机制。未来表征不仅作为候选评分证据，还用于产生下一查询位姿，"
        "从而在不访问真实未来观测的条件下形成可递归展开的预测闭环；统一方法采用k步形式描述，默认"
        "设置为一次q_0→q_1展开。"
    ),
    (
        "提出受约束的计算自适应前瞻机制，通过Top-K候选预算、有界残差、候选级回退、少步推理和"
        "决策不变性触发限制预测的作用范围，并将深入前瞻计算集中到可能改变当前动作的候选决策。"
    ),
]


ORGANIZATION_PARAGRAPH = (
    "本文其余部分组织如下：第二节回顾连续环境视觉语言导航、拓扑规划与导航前瞻；第三节定义任务、"
    "符号和基础拓扑策略；第四节介绍PILOT的整体框架与关键机制；第五节说明训练方法；第六节给出"
    "实验设置与分析；第七节讨论适用范围、局限与未来工作；第八节总结全文。"
)


REFERENCES = [
    "[1] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould, and A. van den Hengel, “Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 3674–3683, doi: 10.1109/CVPR.2018.00387.",
    "[2] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 104–120, doi: 10.1007/978-3-030-58604-1_7.",
    "[3] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding,” in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2020, pp. 4392–4412, doi: 10.18653/v1/2020.emnlp-main.356.",
    "[4] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Nießner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3D: Learning from RGB-D Data in Indoor Environments,” in Proc. Int. Conf. 3D Vision (3DV), 2017, pp. 667–676, doi: 10.1109/3DV.2017.00081.",
    "[5] M. Savva et al., “Habitat: A Platform for Embodied AI Research,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 9338–9346, doi: 10.1109/ICCV.2019.00943.",
    "[6] Y. Hong, Z. Wang, Q. Wu, and S. Gould, “Bridging the Gap Between Learning in Discrete and Continuous Environments for Vision-and-Language Navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 15418–15428, doi: 10.1109/CVPR52688.2022.01500.",
    "[7] D. An, H. Wang, W. Wang, Z. Wang, Y. Huang, K. He, and L. Wang, “ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 5130–5145, 2025, doi: 10.1109/TPAMI.2024.3386695.",
    "[8] Z. Wang, X. Li, J. Yang, Y. Liu, J. Hu, M. Jiang, and S. Jiang, “Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2024, pp. 13753–13762, doi: 10.1109/CVPR52733.2024.01305.",
    "[9] S. Ye, S. Mao, Y. Cui, X. Yu, S. Zhai, W. Chen, S. Zhou, R. Xiong, and Y. Wang, “ETP-R1: Evolving Topological Planning with Reinforcement Fine-Tuning for Vision-Language Navigation in Continuous Environments,” arXiv:2512.20940, 2025.",
    "[10] J. Peng, J. Guo, Y. Xu, Y. Liu, J. Yan, X. Ye, H. Li, and X. Wang, “Dynamic Topology Awareness: Breaking the Granularity Rigidity in Vision-Language Navigation,” arXiv:2601.21751, 2026.",
    "[11] H. Wang, W. Liang, L. Van Gool, and W. Wang, “Dreamwalker: Mental Planning for Continuous Vision-Language Navigation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 10839–10849, doi: 10.1109/ICCV51070.2023.00998.",
    "[12] X. Yao, J. Gao, and C. Xu, “NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 5536–5546, doi: 10.1109/ICCV51701.2025.00525.",
]


def main() -> None:
    document = Document(DOCUMENT_PATH)
    intro_heading = find_paragraph(document, "I. 引言")
    related_heading = find_paragraph(document, "II. 相关工作")
    paragraphs = document.paragraphs
    intro_index = paragraph_index(paragraphs, intro_heading)
    related_index = paragraph_index(paragraphs, related_heading)

    contribution_paragraphs = [
        p for p in paragraphs[intro_index + 1:related_index]
        if p.style.name == "List Number"
    ]
    if len(contribution_paragraphs) != 3:
        raise RuntimeError(f"引言贡献列表数量异常：{len(contribution_paragraphs)}")
    for paragraph, text in zip(contribution_paragraphs, CONTRIBUTIONS):
        replace_text_preserve_run(paragraph, text)

    first_contribution = contribution_paragraphs[0]
    last_contribution = contribution_paragraphs[-1]
    paragraphs = document.paragraphs
    first_index = paragraph_index(paragraphs, first_contribution)
    for paragraph in list(paragraphs[intro_index + 1:first_index]):
        remove_paragraph(paragraph)
    for text in INTRO_PARAGRAPHS:
        first_contribution.insert_paragraph_before(text, style="Normal")
    first_contribution.insert_paragraph_before(FIGURE_CAPTION, style="Caption")
    first_contribution.insert_paragraph_before(EVALUATION_PARAGRAPH, style="Normal")
    first_contribution.insert_paragraph_before("本文的主要贡献如下：", style="Normal")

    paragraphs = document.paragraphs
    last_index = paragraph_index(paragraphs, last_contribution)
    related_index = paragraph_index(paragraphs, related_heading)
    for paragraph in list(paragraphs[last_index + 1:related_index]):
        remove_paragraph(paragraph)
    related_heading.insert_paragraph_before(ORGANIZATION_PARAGRAPH, style="Normal")

    reference_heading = find_paragraph(document, "参考文献")
    paragraphs = document.paragraphs
    reference_index = paragraph_index(paragraphs, reference_heading)
    for paragraph in list(paragraphs[reference_index + 1:]):
        remove_paragraph(paragraph)
    for reference in REFERENCES:
        paragraph = document.add_paragraph(reference, style="Normal")
        format_reference(paragraph)

    document.save(DOCUMENT_PATH)

    verify = Document(DOCUMENT_PATH)
    texts = [p.text.strip() for p in verify.paragraphs]
    a = texts.index("I. 引言")
    b = texts.index("II. 相关工作")
    intro_text = "\n".join(texts[a:b])
    for required in ("CWP", "ETPNav", "HNR", "ETP-R1", "DGNav", "Dreamwalker", "NavMorph"):
        if required not in intro_text:
            raise RuntimeError(f"引言缺少技术脉络项：{required}")
    if "现有拓扑方法缺少候选未来视觉证据" in intro_text:
        raise RuntimeError("仍存在过度概括的旧缺口表述")
    if not any("CVPR52733.2024.01305" in text for text in texts):
        raise RuntimeError("HNR参考文献校验失败")
    if not any("2601.21751" in text for text in texts):
        raise RuntimeError("DGNav参考文献校验失败")
    print(DOCUMENT_PATH)


if __name__ == "__main__":
    main()
