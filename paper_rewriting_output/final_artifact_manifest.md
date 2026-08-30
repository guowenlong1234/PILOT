# 最终产物清单

工作流：`build_from_materials`；层级：`pro`；输出语言：中文。

| 类别 | 文件 | 用途与状态 |
|---|---|---|
| required | `paper_spine_config.json`、`paper_spine_config.md` | 已确认的工作流配置、方案 A 与实验范围。 |
| required | `source_map.md`、`source_inventory.md`、`source_inventory.json` | 当前工程、论文材料和代码证据索引。 |
| required | `reference_materials/source_index.md`、`research_dossier.md`、`exemplar_learning_dossier.md`、`style_profile.md`、`sota_gap_map.md` | 本地优先的研究、范例结构和近邻工作差异分析。 |
| required | `motivation_options_after_research.md`、`confirmed_motivation.md`、`confirmed_contribution.md` | 用户选择的方案 A、控制性动机与贡献边界。 |
| required | `citation_support_bank.md`、`citation_verification_en.md`、`citation_quality_audit.md` | 引用候选、逐条核验和质量审计。 |
| required | `section_blueprints.md`、`writing_rationale_matrix.md` | 章节蓝图和逐单元写作依据。 |
| required | `evidence_bank.md`、`claim_register.md`、`figure_asset_map.md` | 证据、主张和图形资产映射。 |
| required | `final_paper/main.tex`、`final_paper/references.bib` | 中文期刊论文逻辑母稿与 BibTeX 文献库。 |
| required | `final_paper/figures/method_overview.png`、`final_paper/figures/mechanism_analysis.png` | 方法总图与机制分析图。 |
| required | `final_paper/paper.zh.docx`、`word_report.zh.md` | 可编辑中文 Word 初稿与结构/字体检查报告。 |
| required | `final_paper/paper.pdf`、`latex_report.md` | 经无界面逐页检查的阅读版 PDF 与生成说明；PDF 来自 Word/LibreOffice，不是 TeX 编译。 |
| required | `integrity_audit.md`、`artifact_check.md`、`progress.md` | 完整性、产物链与流程关卡记录。 |
| pro-extra | `structured_review.md`、`structured_review_clarity.md`、`structured_review_contribution.md`、`structured_review_methods.md` | 清晰性、贡献和方法三类审稿意见及汇总。 |
| pro-extra | `reviewer_audit.md`、`reviewer_audit_check.md` | 审稿人价值图、主要异议和审计结果。 |
| pro-extra | `results_validation.md`、`results_validation_check.md`、`contribution_check.md` | 贡献—实验验证映射与硬性关卡。 |
| pro-extra | `humanize_matrix.md`、`humanize_report.md` | 中文稿的语言自然度与写作模式检查。 |
| pro-extra | `results_placeholder_manifest.md` | 所有待补结果键、元数据和原始来源约束。 |
| pro-extra | `tools/make_word_tex.py`、`tools/style_paper_docx.py` | 可复现的 Word 转换兼容层和样式脚本。 |

## 不适用的可选类别

- optional-translation：不适用；正文直接以中文撰写，不是英文稿的翻译包。
- optional-submission：未请求；结果尚未回填，当前不生成投稿材料。
- optional-review-response：未请求。

## 校验摘要

- `final_paper/paper.zh.docx` SHA-256：`aab0cb2c905c05590960b0bcf9114eaba180437910505c64af09d851c51d3fd4`
- `final_paper/paper.pdf` SHA-256：`c238202cdb0d65b38f0592fb0f279b57f30483dbcc5a15ef92584288084e0b8b`
- `final_paper/main.tex` SHA-256：`4148e9c15506ee60bcc1d09dea2434334b1787915599f533b4809c564347b655`
- `final_paper/references.bib` SHA-256：`e33a923f0fae4be83bb8f5da5fd66c6530c83c2606f83abec3237442166d31e3`

当前版本定位为中文初稿。任何结果键被替换、方法实现发生变化或目标期刊确定后，都应重新生成 Word/PDF、更新哈希并重新运行最终审计。
