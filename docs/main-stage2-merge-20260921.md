# 主工作区合并二阶段（2026-09-21）

用户要求主工作区成为包含两条开发线的最新入口。当前笔记本主目录仍使用`feature/e24-joint-sft`，不是历史`main`分支。

- `b10afdc`：主工作区快照，保存计划、分析和工作区归档记录；论文与绘图内容改为本地保留、Git忽略。
- `ffff1ef`：提交二阶段原有两份未提交记录，包括四环境重试和已备份检查点清理说明。
- `0903fcc`：将二阶段`feature/stage2-e24-offline`合入主工作区，保留主线RxR优化与恢复改动，同时纳入二阶段采集、离线训练、在线评价和实验编排。

冲突仅涉及`.gitignore`、实施计划、`research.md`；忽略规则及研究记录保留双方内容，计划采用二阶段实施记录并注明历史阶段与最新入口。代码文件自动合并，无额外运行逻辑改写。

## 本地文件保留

`paper/`、`paper_rewriting_output/`、`docs/TopoForesight_*`、`docs/paper-mainline.md`和`pelican-bicycle.html`已忽略；75个既有跟踪文件只从索引移除。269个本地文件在操作前后SHA256一致。历史提交仍保留此前已提交的论文内容，未改写Git历史。已有诊断产物忽略规则及二阶段`stage2_assets/`规则均保留。

## 验证

提交前完成变化Python文件语法解析、`git diff --cached --check`及文件大小检查。真实运行环境验证在测评机专用容器`gwl-etpr1-rae`、`etpr1_rae`环境内进行，临时工作区固定检出`0903fcc`；CPU线程数限制为1，`CUDA_VISIBLE_DEVICES`为空。没有启动GPU测试。

首次直接调用环境Python因未加载项目Habitat路径而在测试收集阶段退出2；改用现有主目录`etpr1_rae_runtime_exec.sh`加载只读依赖，Python当前目录保持临时工作区，运行前断言`vlnce_baselines.__file__`来自临时合并代码。

实际版本：Python 3.11.15、PyTorch 2.2.2+cu121、Transformers 4.49.0、CUDA构建12.1、Habitat/Habitat-Sim均0.3.3。

测试选择（通过项目运行环境入口调用`pytest.main`）：

```text
-q tests/test_stage2* tests/test_panorama* tests/test_ghost_concat* tests/test_resume_environment_resize.py tests/test_rxr_parallel_teacher.py
```

结果：退出0，`165 passed, 1 skipped, 1 warning in 1.15s`。测试覆盖二阶段数据、训练恢复、在线评分、未来信息对照、全景上下文、候选融合及主线RxR相关逻辑；未重新运行真实导航完整测评或长训练。

两条分支已推送到训练机中央裸仓库。远端运行中的主目录和stage2目录没有pull或切换提交，测评机临时验证工作区已移除，原采集进程仍存活。二阶段独立目录仍服务当前采集，不因本次合入而自动归档。
