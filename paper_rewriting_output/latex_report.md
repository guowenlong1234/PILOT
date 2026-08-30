# LaTeX、Word 与 PDF 生成报告

- Status: PASS
- 输出语言：中文（`output_language=zh`）
- 文稿来源：`build_from_materials`
- LaTeX 源文件：`final_paper/main.tex`
- 参考文献：`final_paper/references.bib`
- Word 文件：`final_paper/paper.zh.docx`
- PDF 文件：`final_paper/paper.pdf`

## 生成说明

本机没有可用的 TeX 引擎，因此未执行 XeLaTeX/CTeX 编译，也没有生成 `main.pdf`。`main.tex` 已通过 PaperSpine 的 LaTeX 结构与引用检查。为提供可阅读、可逐页检查的 PDF，先由 Pandoc 将同一 LaTeX 正文和 BibTeX 文献转换为中文 Word，再用 LibreOffice 7.3 无界面渲染 Word；最终 PDF 因此是 Word 版的版式等价导出，不冒充 LaTeX 编译产物。

## Word 检查

- 中文字体：宋体；拉丁字符：Times New Roman。
- 数字引用由 `\cite{...}`、`references.bib` 与 IEEE 数字 CSL 解析，不使用手写引用序号。
- `word_guard.py`：PASS。
- 无界面渲染：15 页 A4。
- 已逐页检查全部 PNG：未发现文字裁切、对象重叠、缺字、表格越界或未解析的 LaTeX 表格代码。

## PDF 检查

- 页数：15。
- 页面尺寸：A4，595.304 × 841.890 pt。
- PDF 版本：1.6；未加密；无 JavaScript；无表单。
- 已用 Poppler 重新栅格化为 15 张 PNG，页面数量与 Word 渲染一致。
- 最终 PDF 与已检查的 LibreOffice 导出文件 SHA-256 完全一致：`c238202cdb0d65b38f0592fb0f279b57f30483dbcc5a15ef92584288084e0b8b`。

## 文稿状态边界

本次交付是用于实验回填和后续英文定稿的中文逻辑初稿。文中保留具名结果键；R2R/RxR 主结果、主动前视 GRPO、候选级真实渲染诊断以及向前第 2、3 步实验完成后，需要回填原始结果并重新运行全部检查。当前文件不应被标记为投稿就绪版本。
