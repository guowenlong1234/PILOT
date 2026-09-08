# 论文资料目录

该目录集中存放 ETP-R1 / PILOT / TopoForesight 论文相关内容，避免论文初稿、图片和补充材料继续散落在工程源码与通用 `docs/` 目录中。

## 目录结构

- `drafts/`：论文初稿、章节草稿和不同版本的 Markdown 文档。
- `figures/`：论文插图、框架图、定性案例图及其可编辑源文件。
- `tables/`：表格数据、统计结果和制表源文件。
- `references/`：参考文献、BibTeX 和引用核对材料。
- `notes/`：写作笔记、审稿意见和修改清单。

## 当前内容

- `drafts/TopoForesight_TASE_初稿骨架.before-related-work.md`：由 `docs/TopoForesight_TASE_初稿骨架.before-related-work.docx` 整理得到的 Markdown 版本。
- [本地 TASE 与 VLN 写作参考](notes/local-tase-vln-writing-reference-20260905.md)：本地论文的格式、论述和图文组织观察，以及 HNR 等相近工作的比较要点。
- [相关工作 A：联网检索与写作逻辑](notes/related-work-a-research-20260905.md)：任务、航点、地图与拓扑规划的原始来源、正文组织及引用取舍。
- [世界模型技术图谱与 B 小节分析](notes/world-model-landscape-and-section-b-20260905.md)：世界模型路线、NWM/RAE-NWM 技术路径、当前模型核对及 B 小节写作建议。
- [相关工作 C：前瞻规划与决策](notes/related-work-c-research-20260905.md)：16 项相关研究的检索范围、关键方法证据，以及前瞻决策与计算取舍的写作依据。

## 建议约定

- 图片正文引用优先使用相对路径，例如 `../figures/figure-01-overview.png`。
- 文件名建议包含主题和日期；需要保留多个版本时，可追加 `YYYYMMDD` 或版本号。
- 模型权重、训练日志和大体积数据仍按工程约定保存在实验目录，不放入本目录。
