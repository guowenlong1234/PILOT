#!/usr/bin/env python3
"""Rewrite the PILOT introduction and add its verified starter references."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


DOCUMENT_PATH = Path("/home/sia/project/ETP-R1/docs/TopoForesight_TASE_初稿骨架.docx")


INTRO_PARAGRAPHS = [
    (
        "使机器人能够依据自然语言完成长距离移动，是服务机器人从执行预设目标走向自然人机交互的"
        "重要能力。视觉语言导航（Vision-and-Language Navigation，VLN）要求智能体将指令中的地标、"
        "方向、空间关系和动作顺序与第一视角观测持续对齐，并在未见环境中到达指令指定的位置。"
        "R2R在真实室内扫描场景上建立了这一任务的经典基准[1]；VLN-CE进一步移除离散导航图等理想化"
        "条件，使智能体必须在连续三维环境中将高层决策落实为可执行运动[2]；RxR则引入多语言指令和"
        "更密集的时空对齐，进一步提高了长程指令跟随的复杂性[3]。Matterport3D和Habitat分别提供"
        "真实室内场景数据与可复现的具身仿真平台[4], [5]。在这一连续、部分可观测的设置中，一次错误"
        "的高层分支选择往往会引入额外运动、回溯和误差累积，因此导航策略既要理解语言，也要形成"
        "可靠的长程空间判断。"
    ),
    (
        "为缓解部分观测和长时程决策带来的困难，近年来的VLN方法逐步从局部循环记忆发展到显式的"
        "地图与拓扑表示。BEVBert将局部度量图与全局拓扑图结合，用于学习具有空间感知能力的多模态"
        "表示[6]；GridMM通过栅格记忆聚合历史观测并支持跨模态推理[7]；ETPNav维护已访问节点和仍可"
        "探索的ghost候选，将连续环境中的长程决策压缩为图上的候选选择[8]。在此基础上，ETP-R1通过"
        "扩大训练数据并引入强化微调，进一步增强了图式策略的学习能力[9]。这些研究表明，显式空间"
        "结构能够有效组织历史、支持回溯，并为连续环境中的长程规划提供稳定的决策接口。"
    ),
    (
        "然而，在线拓扑图擅长回答“智能体到过哪里以及还可以去哪里”，却不会自然给出“选择某个未"
        "访问候选后可能看到什么”。ghost候选的表示仍主要来自当前全景、局部几何和历史图状态。"
        "当候选位于转角之后、受到遮挡，或多个走廊和房间具有相似外观时，不同候选在当前证据下可能"
        "难以区分。此时，即使历史被完整记录，策略仍缺少与每个候选直接对齐的未来视觉线索；错误选择"
        "往往只有在机器人实际进入对应区域后才能被发现，并需要通过回溯加以纠正。"
    ),
    (
        "世界模型为弥补这一信息缺口提供了可行方向：它可以在动作执行之前预测潜在未来，并将预测"
        "结果作为当前决策的补充证据。Dreamwalker利用心理规划研究了连续视觉语言导航中的未来推演"
        "[10]，NavMorph进一步采用潜在世界模型增强连续环境中的环境理解与规划[11]。这些工作验证了"
        "未来预测对导航的潜力，但将世界模型接入在线拓扑策略仍需要解决三个相互关联的问题：预测的"
        "未来信息如何与具体候选一一对应；在不能读取真实未来观测的条件下，后续查询位置如何由预测"
        "状态产生；以及如何限制预测误差和额外计算对基础策略的影响。若对所有候选进行同等深度的"
        "展开，不仅计算代价较高，多步误差也可能累积。因此，一个适用于拓扑导航的前瞻机制不仅要"
        "回答“未来可能是什么”，还需要决定“对哪个候选看、向前看多远以及何时值得继续看”。"
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
        "第二阶段查询并保留基础动作。因此，PILOT并不以世界模型替代原有导航策略，而是将其作为一种"
        "候选级、按需使用的未来证据来源。"
    ),
]


FIGURE_CAPTION = (
    "图1. PILOT总体框架。给定语言指令、历史视觉观测和在线拓扑图，第一阶段预测各有效ghost候选的"
    "到达状态q_0并增强其表示；基础策略完成初始评分后，第二阶段仅对Top-K移动候选从q_0递归前瞻"
    "至q_k，并利用终端未来表征生成有界残差以重排候选。默认k=1；当有界残差不可能改变当前决策时，"
    "跳过第二阶段深层查询。待绘制图片应区分冻结模块与可训练模块，并标明候选级回退路径。"
)


EVALUATION_PARAGRAPH = (
    "本文在R2R-CE和RxR-CE基准上对PILOT进行系统评估。除与对应基础拓扑策略进行比较外，实验"
    "围绕未来信息来源、展开步数k、候选预算K、预测质量、候选覆盖率以及端到端延迟等方面展开，"
    "以考察未来信息的决策价值及其与计算成本之间的关系。"
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
    "本文其余部分组织如下：第二节回顾拓扑式视觉语言导航、导航世界模型和计算自适应推理；第三节"
    "定义任务、符号和基础拓扑策略；第四节介绍PILOT的整体框架与关键机制；第五节说明训练方法；"
    "第六节给出实验设置与分析；第七节讨论适用范围、局限与未来工作；第八节总结全文。"
)


REFERENCES = [
    "[1] P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould, and A. van den Hengel, “Vision-and-Language Navigation: Interpreting Visually-Grounded Navigation Instructions in Real Environments,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 3674–3683, doi: 10.1109/CVPR.2018.00387.",
    "[2] J. Krantz, E. Wijmans, A. Majumdar, D. Batra, and S. Lee, “Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2020, pp. 104–120, doi: 10.1007/978-3-030-58604-1_7.",
    "[3] A. Ku, P. Anderson, R. Patel, E. Ie, and J. Baldridge, “Room-Across-Room: Multilingual Vision-and-Language Navigation with Dense Spatiotemporal Grounding,” in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2020, pp. 4392–4412, doi: 10.18653/v1/2020.emnlp-main.356.",
    "[4] A. Chang, A. Dai, T. Funkhouser, M. Halber, M. Nießner, M. Savva, S. Song, A. Zeng, and Y. Zhang, “Matterport3D: Learning from RGB-D Data in Indoor Environments,” in Proc. Int. Conf. 3D Vision (3DV), 2017, pp. 667–676, doi: 10.1109/3DV.2017.00081.",
    "[5] M. Savva et al., “Habitat: A Platform for Embodied AI Research,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2019, pp. 9338–9346, doi: 10.1109/ICCV.2019.00943.",
    "[6] D. An, Y. Qi, Y. Li, Y. Huang, L. Wang, T. Tan, and J. Shao, “BEVBert: Multimodal Map Pre-training for Language-Guided Navigation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, arXiv:2212.04385.",
    "[7] Z. Wang, X. Li, J. Yang, Y. Liu, and S. Jiang, “GridMM: Grid Memory Map for Vision-and-Language Navigation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 15579–15590, doi: 10.1109/ICCV51070.2023.01432.",
    "[8] D. An, H. Wang, W. Wang, Z. Wang, Y. Huang, K. He, and L. Wang, “ETPNav: Evolving Topological Planning for Vision-Language Navigation in Continuous Environments,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 47, no. 7, pp. 5130–5145, 2025, doi: 10.1109/TPAMI.2024.3386695.",
    "[9] S. Ye, S. Mao, Y. Cui, X. Yu, S. Zhai, W. Chen, S. Zhou, R. Xiong, and Y. Wang, “ETP-R1: Evolving Topological Planning with Reinforcement Fine-Tuning for Vision-Language Navigation in Continuous Environments,” arXiv:2512.20940, 2025.",
    "[10] H. Wang, W. Liang, L. Van Gool, and W. Wang, “Dreamwalker: Mental Planning for Continuous Vision-Language Navigation,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 10839–10849, doi: 10.1109/ICCV51070.2023.00998.",
    "[11] X. Yao, J. Gao, and C. Xu, “NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2025, pp. 5536–5546, doi: 10.1109/ICCV51701.2025.00525.",
]


def remove_paragraph(paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)


def replace_text_preserve_run(paragraph, text: str) -> None:
    first_rpr = None
    for run in paragraph.runs:
        if run._element.rPr is not None:
            first_rpr = deepcopy(run._element.rPr)
            break
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)
    run = paragraph.add_run(text)
    if first_rpr is not None:
        if run._element.rPr is not None:
            run._element.remove(run._element.rPr)
        run._element.insert(0, first_rpr)


def format_reference(paragraph) -> None:
    fmt = paragraph.paragraph_format
    fmt.left_indent = Inches(0.25)
    fmt.first_line_indent = Inches(-0.25)
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(3)
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    for run in paragraph.runs:
        run.font.size = Pt(8.5)
        rpr = run._element.get_or_add_rPr()
        rfonts = rpr.get_or_add_rFonts()
        rfonts.set(qn("w:ascii"), "Calibri")
        rfonts.set(qn("w:hAnsi"), "Calibri")
        rfonts.set(qn("w:eastAsia"), "SimSun")


def find_paragraph(document, exact_text: str):
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == exact_text:
            return paragraph
    raise RuntimeError(f"未找到段落：{exact_text}")


def paragraph_index(paragraphs, target) -> int:
    for index, paragraph in enumerate(paragraphs):
        if paragraph._p is target._p:
            return index
    raise RuntimeError(f"段落不在当前文档序列中：{target.text[:40]}")


def main() -> None:
    document = Document(DOCUMENT_PATH)
    intro_heading = find_paragraph(document, "I. 引言")
    related_heading = find_paragraph(document, "II. 相关工作")

    paragraphs = document.paragraphs
    intro_index = paragraph_index(paragraphs, intro_heading)
    related_index = paragraph_index(paragraphs, related_heading)
    old_contribution_start = None
    for index in range(intro_index + 1, related_index):
        if paragraphs[index].text.strip().startswith("提出一种双阶段候选级世界模型前瞻框架"):
            old_contribution_start = index
            break
    if old_contribution_start is None:
        raise RuntimeError("未找到现有贡献列表")

    contribution_paragraphs = paragraphs[old_contribution_start:old_contribution_start + 3]
    if any(p.style.name != "List Number" for p in contribution_paragraphs):
        raise RuntimeError("贡献列表结构与预期不一致")

    for paragraph, text in zip(contribution_paragraphs, CONTRIBUTIONS):
        replace_text_preserve_run(paragraph, text)

    for paragraph in list(document.paragraphs[intro_index + 1:old_contribution_start]):
        remove_paragraph(paragraph)

    first_contribution = contribution_paragraphs[0]
    for text in INTRO_PARAGRAPHS:
        first_contribution.insert_paragraph_before(text, style="Normal")
    first_contribution.insert_paragraph_before(FIGURE_CAPTION, style="Caption")
    first_contribution.insert_paragraph_before(EVALUATION_PARAGRAPH, style="Normal")
    first_contribution.insert_paragraph_before("本文的主要贡献如下：", style="Normal")

    paragraphs = document.paragraphs
    last_contribution_index = paragraph_index(paragraphs, contribution_paragraphs[-1])
    related_index = paragraph_index(paragraphs, related_heading)
    for paragraph in list(paragraphs[last_contribution_index + 1:related_index]):
        remove_paragraph(paragraph)
    related_heading.insert_paragraph_before(ORGANIZATION_PARAGRAPH, style="Normal")

    checklist_heading = find_paragraph(document, "写作完成前检查清单")
    existing_reference = None
    for paragraph in document.paragraphs:
        if paragraph.text.strip() == "参考文献":
            existing_reference = paragraph
            break
    if existing_reference is not None:
        paragraphs = document.paragraphs
        ref_index = paragraph_index(paragraphs, existing_reference)
        checklist_index = paragraph_index(paragraphs, checklist_heading)
        for paragraph in list(paragraphs[ref_index:checklist_index]):
            remove_paragraph(paragraph)

    checklist_heading.insert_paragraph_before("参考文献", style="Heading 1")
    for reference in REFERENCES:
        paragraph = checklist_heading.insert_paragraph_before(reference, style="Normal")
        format_reference(paragraph)

    document.save(DOCUMENT_PATH)

    verify = Document(DOCUMENT_PATH)
    texts = [p.text.strip() for p in verify.paragraphs]
    if not texts[0].startswith("PILOT:"):
        raise RuntimeError("标题校验失败")
    if "A. 应用背景" in texts or "C. 方法概述" in texts:
        raise RuntimeError("旧引言小标题未被移除")
    if not any(text.startswith("为此，本文提出PILOT") for text in texts):
        raise RuntimeError("新引言校验失败")
    if sum(1 for text in texts if text == "参考文献") != 1:
        raise RuntimeError("参考文献节校验失败")
    if not any("10.1109/CVPR.2018.00387" in text for text in texts):
        raise RuntimeError("R2R参考文献DOI校验失败")
    print(DOCUMENT_PATH)


if __name__ == "__main__":
    main()
