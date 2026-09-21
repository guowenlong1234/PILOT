# Project Research

2026-09-21工作区整理完成：最终归档训练机ETPNav59历史验证目录，累计归档15个，三机现各保留主目录、stage2-e24、recursive-future-rollout，共9个。用户明确保留递归前瞻；后续明确被主线包含或明确不采用的历史工作区可核对依赖并保存内容后自主归档，采用意向不明且无法判断的才询问。最终清单、归档位置与恢复说明见`docs/git-workspace-inventory-20260921.md`顶部。RxR训练及stage2采集进程仍存活。

2026-09-21第四批归档：经用户确认，笔记本`ETP-R1-world-model-migration`归档完成。原分支和准确提交保留；32项额外文件及未提交research.md全文/补丁均保存并验证。当前剩10个工作区（笔记本3、训练机4、测评机3），远端未做修改。恢复入口见`docs/git-workspace-inventory-20260921.md`顶部。

2026-09-21第三批归档：经用户逐项确认，三机`ETP-R1-grpo-stop-fix`归档完成，各机准确提交、原分支、非Git文件及独立运行环境均保留并核验。当前剩11个工作区（笔记本4、训练机4、测评机3），RxR训练与stage2采集进程仍存活。详细清单及恢复入口见`docs/git-workspace-inventory-20260921.md`顶部。

2026-09-21第二批归档：经用户逐项确认，笔记本和训练机`ETP-R1-perf`归档完成，原性能分支、7个主线未包含的提交及训练机独立运行环境均保留。累计归档10个工作区，当前剩14个（笔记本5、训练机5、测评机4）；详情与恢复入口见`docs/git-workspace-inventory-20260921.md`。

2026-09-21，经用户授权归档已被主线完整包含的五类8个工作区，剩余笔记本6、训练机6、测评机4，共16个。原分支及精确提交归档引用保留；非Git文件与链接保存在各机器工程父目录的`ETP-R1-workspace-archives/20260921/`。归档清单、恢复方法和验证结果见`docs/git-workspace-inventory-20260921.md`顶部；其余历史线等待逐个讨论，未处理。

2026-09-21 Git工作区盘点：三机共24个已注册工作区（笔记本9、训练机11、测评机4）。当前实际主线为`feature/e24-joint-sft`，二阶段独立开发线为`feature/stage2-e24-offline`；训练机主目录正在运行RxR，测评机stage2目录正在采集。已合入、历史独立分支、未提交内容与同步差异见`docs/git-workspace-inventory-20260921.md`。本次未删除或同步工作区，未改变任务进程。

## Project Goal

ETP-R1 是一个 VLN-CE 项目：让智能体在连续三维环境里，根据自然语言指令导航到目标位置。代码包含预训练、在线 SFT、在线 RFT/GRPO、R2R-CE 和 RxR-CE 评测流程。

## Quick Start And Environment

2026-09-21，RxR已在训练机以性能优化版从2200步完整续训，目标仍30000步。实验名rxr_dino_baseline_20260920，产物改写入该实验根的 `train_fast_resume_20260921/`，原训练与测评产物保留。两卡各12环境×累积1次，总批量24；已核对视觉编译、导航重算等优化开启，模型/优化器/调度器/缩放与主进程随机状态恢复。通过显式allow_env_count_change_on_resume仅重建6→12的环境采样队列，教师概率及衰减曲线不变；95项相关CPU检查通过。当前任务正在运行，操作记录见 `docs/rxr-fast-resume-20260921.md`。

2026-09-21核查：RxR 2200步双卡全量val_unseen测评已于9月20日19:24结束，退出0，用时约84分42秒。两卡各5503条，合并11006条无重复、无遗漏，逐条重算汇总一致。SR51.3538%、SPL42.7064%、nDTW60.1102%、SDTW42.7691%、OSR58.8043%，距目标6.4125米；四语言SR50.35%—52.39%。完整结果见 `docs/rxr-full-evaluation-20260920.md`。目前只是2200步早期点，不能据此判断最终收敛或相对基线收益。

2026-09-20晚，已在训练机双卡启动最新RxR基座2200步的完整val_unseen测评，11006条guide路线、四语言，两卡各5503条、每卡6环境。采用原RxR无滑动及control回退规则，模型贪心选动作；首次双卡设备类型错误已修复为torch.device。18:01监控累计完成293条，两卡正常、无新报错；初估19:20左右完成，预留至19:50。入口 `scripts/run_rxr_full_eval_server.sh`，日志与结果位置见 `docs/rxr-full-evaluation-20260920.md`；训练仍停止。

2026-09-20，完成训练机RxR吞吐优化：原长训练已停止，2200步模型/恢复状态保留。最终每卡12环境×累积1次、总批量24，采用冻结视觉编译、并行教师查询、异步有限值检查及导航中间结果重算；教师动作采样概率和原衰减曲线不变。双卡50步同条件复测动作吞吐45.986→54.739/秒（+19.03%），更新时间6.832→5.813秒（-14.91%），峰值显存约25.8GiB；79项回归通过、2项跳过。代码已合入主工作区，入口 `scripts/manage_rae_rxr_sft_fast_server.sh`，完整证据及使用边界见 `docs/rxr-throughput-optimization-20260920.md`。没有自动恢复长训练。

2026-09-20，按用户要求准备并启动训练机上的 RxR DINO 导航基座，不使用世界模型或二阶段。已有基座链路及465000步DINO预训练权重齐全；本次显式关闭RAENWM/ACTIVE_LOOKAHEAD并补齐版本日志。训练机126项测试、真实小规模HDF5、单环境训练及恢复、总批量24的双卡短训练、R2R/RxR单回合检查均通过。正式参数沿用原RxR脚本的30000步、1.5e-5学习率、1000步预热和每200步保存；4卡×6环境改为2卡×6环境×2次累积。10:28首份200步模型和恢复状态验收通过，损失2.362，优化器/调度器均200步、无AMP溢出；任务继续运行。命令、证据与验收见 `docs/rxr-dino-baseline-training-20260920.md`。同日训练机已重启，默认驱动恢复为一致的580.178.04，不再使用旧580.173.02隔离库；此前记录为历史状态。

2026-09-17，完成E24后续优化分析：见 `docs/stage2-e24-optimization-analysis-20260917.md`。复核独立stage2工作区首轮结果与真实代码，最佳4750步教师future有效组净+79、缺失组净-55；发现当前主要排序监督依赖教师future有效，以及候选存在/future有效mask耦合。完整dev CPU诊断得到现有±1边界下1008个错误的标签知情乐观纠正上界。只读对照ETPNav历史提交a197196，建议依次做有限校准、future增量对照、缺失监督、mask解耦、小模型与定向重采。本次仅分析与文档，没有启动新训练；历史一阶段与旧实验记录保留。

2026-09-16，按用户要求形成完整二阶段实施计划：`docs/plans/2026-09-16-stage2-e24-training-plan.md`。明确步骤0—11的代码职责、全景q0快照、q1独立随机流、完整stage1冻结、预测future离线数据、容量估算、训练/恢复、有限选点与同机完整导航验证。默认建议6400基座，新增命令均标为待实现；本次仅交付计划，未实现或启动训练。

2026-09-16，核查已完成的持久候选一阶段如何接入 ETPNav E24 二阶段，分析见 `docs/stage2-e24-migration-analysis-20260916.md`。两机持久组测评总表均 completed，各25点；SR最佳9200，SPL及SR+SPL最佳6400。当前 ghost_concat、冻结导航及 world_exact_select 与旧E24存在显式互斥；全景查询也没有旧E24需要的源快照接口，不能只开配置。建议冻结完整stage1、用当前预测future重新采集、复用E24网络和移动候选离线损失单独训练；旧Oracle分片、固定avg3加载、含STOP的旧native联合CE及优化器恢复合同不能直接沿用。仅更新分析文档，未启动训练或修改运行代码。

2026-09-16，按用户要求删除笔记本、训练机、测评机的 `ETP-R1-prediction-adapter` 独立工作区及三机本地/中央仓库的 `feature/prediction-cls-adapter` 分支，未合入主工作区。该实验测试和训练完成，但同机适配前后SR为64.0566%/63.8390%、SPL为54.3754%/54.2389%，未证实适配器收益。产物与报告保留：训练机 `/mnt/data2tb/ETP-R1_data/experiments/prediction_cls_adapter_20260916/`；测评机 `/home/a6000/gwl/ETP-R1/data/logs/prediction_cls_adapter_20260916/`（从已删除工作区迁入，模型在 `adapter_inputs/`）。两端的 `workspace_archive/report.md` 和 `results.json` 保存最终实验记录；历史命令中的工作区路径保持原样。源码历史提交为 `ed948f649042467f404ddd5b9708a139f74026ff`，对应分支已删除。

2026-09-16 只读核查：持久状态组 `ghost_concat_persistent_compiled_10k_20260911` 双机全量测评已完成，训练机于9月15日10:18:48、测评机于10:50:40结束（北京时间）。两份总表均为 completed；合并覆盖200至10000步全部50点，逐点episode JSON均为1839条，manifest均为completed且退出码0。成功率最佳9200步：SR64.0566%、SPL54.3754%、nDTW66.7336%、距离目标4.0757m、路径12.2549m；SPL最佳6400步：SR63.6215%、SPL55.2559%；最后10000步：SR62.3709%、SPL53.5604%。50点平均SR62.8070%、SPL54.0251%；仅1点SR超过历史14200基线，所有点SPL均低于该基线，尚无稳定提升证据。两机均未发现本组运行进程，GPU空闲；训练机默认NVML仍版本不匹配，使用既有隔离580.173.02库只读查询成功。证据为两机实验根的eval_summary.json、50份episode结果及manifest、eval_supervisor.log；未修改远端任务或产物。下方9月14日记录为历史状态。

2026-09-14，按用户要求启动持久状态组双机全量测评：训练机GPU0负责400、800……10000，测评机负责200、600……9800，各25点、每点1839条路线。两边已实际推进。训练机因系统驱动升级未重启导致EGL失败，最终用项目内解压的匹配580.173.02用户态库恢复，未修改系统；下次重启后须重新核对。精确命令、交付和状态路径见 `docs/persistent-parallel-evaluation-20260914.md`。下方“尚无全量结果”是启动前的核查结论。

2026-09-14 只读核查：训练机 `ghost_concat_persistent_compiled_10k_20260911` 已于9月12日15:13左右完成10000步，50份模型齐全，但尚无全量测评结果。测评机最新完成的是 `ghost_concat_direct_compiled_10k_20260909`，9月11日20:25完成50点、每点1839条R2R val_unseen；成功率最佳4600步为63.9478%/SPL54.1082%，最后10000步为62.2621%/53.6175%。相对历史基线的成功率峰值仅净多4条且SPL下降，不能称为稳定提升。证据与基线比较见 `docs/latest-evaluation-status-20260914.md`。

2026-09-11 14:31，真实GPU验证通过后，持久候选状态以 `1093ee8` 合入当前 `feature/e24-joint-sft` 并同步训练机主工作区，启动 `ghost_concat_persistent_compiled_10k_20260911`。从原始14200基座重新训练10000步，每200步保存，双卡总批量8；除持久状态模式和输出路径外，参数与9月9日实验一致。产物位于训练机数据盘；测评机继续旧实验队列。启动记录见 `docs/persistent-ghost-compiled-10k-operations-20260911.md`。

2026-09-11，持久候选状态已在训练机独立工作区通过真实单卡更新、双卡总批量8更新、第2步恢复至第4步、冻结权重审计及R2R四路线/RxR单路线验证。实际覆盖跨步无预测保留；展开配置核对，长训相对9月9日实验仅改变候选记忆模式，其他计算参数一致。详细证据见 `docs/persistent-ghost-gpu-validation-20260911.md`。下方9月10日GPU待验收记录是历史状态。

2026-09-10，本独立工作区分支 `feature/persistent-ghost-state` 新增可选的融合节点状态持久化：`ghost_concat_memory_mode=persistent_node_state`，按真实观测次数递推合并旧融合状态与新观测，当前预测再融合写回；整段 rollout 保留跨步梯度。旧模式为缺省 `current_step_only`。训练机独立工作区 `/home/gwl/project/etpr1/ETP-R1-persistent-ghost`，使用 `etpnav_unified`，运行时独立复制。已完成训练机 119 项不同 CPU 测试（117 项回归＋2 项审计故障注入）；两张 GPU 正被既有训练占用，真实单卡/双卡/恢复/短评测仍待验收。入口、检查点契约、测试与产物位置见 `docs/persistent-ghost-state-implementation-20260910.md`。本分支不改论文、不启动长训练。


2026-09-09 晚间已重新启动 `ghost_concat_direct_compiled_10k_20260909` 长训练：主工作区选定合并版本，14200 基座重新初始化、双卡总批量 8、10000 步、每 200 步测评；直接渲染/FP16/64批次上限/Inductor，未包含已放弃的静态条件缓存和导航 SDPA。训练写入数据盘，单份流式同步到测评机。测评机离线补齐专用容器编译工具、修复数字 UID 的编译缓存路径，并通过 8 环境真实预检。操作记录见 `docs/ghost-concat-direct-compiled-10k-operations-20260909.md`。

2026-09-09，按用户决定，将静态条件缓存和导航SDPA之前的 `75cbe1d` 合并到当前 `feature/e24-joint-sft`，合并提交 `372ba21`，保留直接渲染和世界模型编译。当前笔记本工作区为 `/home/sia/project/ETP-R1`，训练机对应 `/home/gwl/project/etpr1/ETP-R1`，使用既有 `etpnav_unified`；后面的性能分支测试数字属于历史试验。静态条件缓存和SDPA已被用户放弃，不作为待实施或待启用方案，相关实现未合入。

本性能分支已验证可选世界模型编译：在 `direct_context_fast.yaml` 后合并 `configs/nwm/direct_context_compiled.yaml`，或为入口添加 `--compile-model --compile-backend inductor`。同轮 16 步双卡短测（剔除四步预热）从 11.296 降至 9.265 秒/更新，吞吐提高 21.92%；11 场景固定质量对照中，复核集 CLS/图块余弦变化仅约 -0.000090/-0.000063。首次编译有启动成本，含预热的 16 步总耗时仍比未编译略长。上次数值失败已复现并确认实际 BF16 中间舍入差异；原生算子 CUDA Graph 可保持零差异但训练更慢。详见 `docs/nwm-compile-validation-20260909.md`。编译仍需显式开启，正式长训练未启动。

未启用编译时的已验证配置为 `configs/nwm/direct_context_fast.yaml`：直接历史位姿定向渲染、DINO/世界模型批次上限 64、FP16。上一轮训练机 12 步双卡复测 12.105 秒/更新，较当轮全景优化版下降 32.27%，较同轮旧 front 10.637 秒仍慢 13.79%。567 项回归及额外 13 项 GPU 检查通过；11 场景固定目标质量对照未见下降。详见 `docs/direct-context-fast-validation-20260909.md`。长训练仍停止，未合并主工作区。

性能优化此前在 2026-09-09 建立的 `perf/panorama-training` 独立分支开发：笔记本 `/home/sia/project/ETP-R1-perf`、训练机 `/home/gwl/project/etpr1/ETP-R1-perf`。当前已选取 `75cbe1d` 合入主工作区，后续被放弃的试验仍留在性能分支历史中。原 10000 步任务及配套等待队列已按用户要求停止。基准与优化记录见 `docs/panorama-training-performance-20260909.md`，历史产物保存在训练机性能工作区 `data/logs/panorama_perf_20260909/`（数据盘软链接）。两工作区各自使用已有资产及隔离运行时，未放宽所有权检查。

README 原始说明要求创建 `etpr1` conda 环境，核心环境为 Python 3.6.12、PyTorch 1.9.1+cu111，并使用 Habitat-Sim 0.1.7 和 Habitat-Lab 0.1.7。该说明和本机已有 `etpnav` 环境只用于了解旧 CLIP 链路，不作为本次 RAE/DINOv2 工作的运行方案。

已检查本机状态：

- `etpr1` 已按用户要求删除。
- `/home/gwl/miniconda3/envs/etpnav` 是旧链路参考环境，Python 3.7.16。
- `etpnav` 已能导入 Habitat-Lab 0.1.7、Habitat-Sim 0.1.7、PyTorch 1.9.1+cu111、TorchVision 0.10.1+cu111、TorchAudio 0.9.1。
- Habitat-Lab 0.1.7 来自 `/home/gwl/project/DGNav/habitat-lab`，通过 develop/egg-link 方式接入 `etpnav`。
- 运行项目入口需要 `torch.utils.tensorboard`，因此 `tensorboard` 使用 1.15.0。它和 `tensorflow 1.13.1` 的声明版本范围不完全一致，但实测两者都能导入，项目入口也能导入。

本次 RAE/DINOv2 替换的正式运行位置是测评机，不是本机：

- 测评机当前没有独立互联网或 Tailscale 入口。笔记本统一经训练机跳板访问：`ssh -J server a6000@10.10.10.2`；训练机通过 2.5 GbE 专线直接访问 `a6000@10.10.10.2`。2026-09-02 更换平台后实测 GPU 为 RTX 3090 24GB，工作根为 `/home/a6000/gwl`。
- 专用容器：`gwl-etpr1-rae`，复用镜像 `gwl-etpnav:etpnav-runtime-20260701185256`，但不复用或修改现有 `gwl-etpnav` 容器。
- 专用环境：`/home/a6000/gwl/miniconda3/envs/etpr1_rae`，从远端 `raenwm` 只读克隆后清除继承的 ETPNav `.pth` 路径绑定。
- 远端 `raenwm` 已核验为 Python 3.11.15、PyTorch 2.2.2+cu121、Transformers 4.49.0；它本身保持只读。
- 远端项目目录固定为 `/home/a6000/gwl/ETP-R1`，Habitat 依赖放在本项目自己的 `.runtime/etpr1_habitat/` 中。
- 所有环境搭建、依赖安装、测试、特征生成、训练和评测均在专用容器与专用环境中执行。本机只负责阅读、编辑、评审、Git 和文档工作。
- 详细执行边界见根目录 `AGENTS.md`，完整技术设计见 `docs/superpowers/specs/2026-07-10-rae-dinov2-visual-encoder-design.md`。

## Project Structure

- `run.py`: 训练、评测入口。
- `run_r2r/`, `run_rxr/`: R2R-CE 和 RxR-CE 的运行脚本与配置。
- `vlnce_baselines/`: 主要训练器、模型、环境封装和配置。
- `habitat_extensions/`: 对 Habitat 任务、传感器、度量、地图和模拟器封装的扩展。
- `pretrain_src/`: 预训练数据、模型和脚本。
- `precompute_img_features/`: 图像/深度特征预计算脚本。
- `paper/`: 论文专用工作区；`drafts/` 存放初稿，`figures/`、`tables/`、`references/` 和 `notes/` 分别存放图片、表格、引用材料与写作笔记。
- `docs/paper-mainline.md`: 2026-08-31 确认的论文主线，说明双阶段候选前瞻、预测状态引导查询、受约束计算，以及已实现方法与计划机制的边界。当前指定写作稿为 `paper/drafts/TopoForesight_TASE_初稿骨架.before-related-work.md`；标题、摘要和引言使用 PILOT；2026-09-14 已完成 III/IV 方法与训练初稿并采用 PILOT，实验及后文仍有 TopoForesight 旧名和占位。独立方法版本为 `paper/drafts/PILOT_方法初稿_v1_20260914.md`，不能仅凭原文件名判断正文版本。
- `copy_extra_files.py`: 将额外数据、checkpoint 等资源复制到项目目录；内含可选的 Habitat 数据软链接逻辑。

## Core Scripts And Entry Points

- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash pretrain_src/run_pt/run_mix_server.bash 2333`: 联合预训练。
- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash dagger 2333`: R2R 在线 SFT。
- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash grpo 2333`: R2R 在线 RFT/GRPO。
- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_r2r/main_server.bash eval 2333`: R2R 评测。
- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main_server.bash dagger 2333`: RxR 在线 SFT。
- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main_server.bash grpo 2333`: RxR 在线 RFT/GRPO。
- `CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_rxr/main_server.bash eval 2333`: RxR 评测。
- `scripts/manage_rae_pretrain_host.sh start|resume|status|tail|stop`：在 4090 宿主机检查 ETPNav/GPU 后，通过专用容器内的 tmux 托管 RAE/DINOv2 完整联合预训练。完整命令见 `docs/rae-dinov2-pretrain-operations.md`。
- `scripts/stress_pretrain_dataloader.py`：不构造模型，直接使用正式 MLM/SAP 数据集和整理 batch 的代码，对 `n_workers`、`pin_memory`、`fork/spawn/forkserver`、CUDA 初始化顺序和 CUDA 搬运做分组压力测试；正式长训练运行时不得并行执行其大规模或 CUDA 模式。
- `scripts/manage_r2r_sft_checkpoint_sync_server.sh start|status|tail|stop`：训练机持续扫描完整的 R2R SFT 模型/训练状态对，经 2.5 GbE 直连把模型 checkpoint 原子同步到测评机。
- `scripts/manage_rae_r2r_eval_watch_host.sh start|status|tail|stop`：测评机宿主机持续监控同步完成的 checkpoint，在确认 ETPNav 未占用 GPU 后，通过 `gwl-etpr1-rae` 和 `etpr1_rae` 串行完成 R2R `val_unseen` 全量评测。
- `scripts/manage_legacy452500_r2r_sft_eval_watch_host.sh start|status|tail|stop`：本次旧最佳预训练迁移 SFT 的专用测评入口；固定 checkpoint/结果目录，并等待测评机 `pretrain_resume_source_250000` 训练进程退出且 GPU 空闲后，再按迭代正序启动完整 R2R `val_unseen` 评测。
- `scripts/manage_eval_best_pretrain_sft_followup_server.sh start|status|tail|stop`：训练机上的 SFT 自动接力入口。它要求当前 15,000 次 SFT 正常退出且最终模型/训练状态成对存在，同时要求测评机预训练正常完成第 500,000 步；随后读取最终 `best_metrics.json`，经 2.5 GbE 直连复制最佳模型并核对大小与 SHA-256，再以相同双卡 SFT 参数在新目录启动下一轮训练。
- `scripts/manage_eval_best465000_r2r_sft_eval_watch_host.sh start|status|tail|stop`：测评机第二轮 SFT 的专用评测入口，固定使用第 465,000 步预训练基座、单卡、8 环境、完整 1,839 个 `val_unseen` episode，并按 checkpoint 迭代正序评测。
- `scripts/manage_second_sft_eval_handoff_host.sh start|status|tail|stop`：测评机两轮 SFT 评测自动接力入口。它逐个验证第一轮的 75 份 JSON 结果，等第一轮全部完成且 GPU 显存低于 1 GiB 后，只启动一次第二轮专用 watcher。

## Important Modules And Functions

- 2026-09-11，持久候选状态已由 `1093ee8` 合入主工作区 `feature/e24-joint-sft`。`ghost_concat_fusion.py` 与 `GraphMap.write_ghost_concat_state` 支持把融合结果写回候选图状态：先按真实观测次数聚合旧融合状态与新观测，再加预测拼接残差，预测不增加观测次数；无预测时保留聚合状态，跨步保留梯度、每段 rollout 新建图。这不同于历史临时融合实验，也不同于第二阶段的原始 q0 预测记录。
- 持久状态配置为 `run_r2r/iter_train_rae_dino_ghost_concat_persistent.yaml`，第二阶段关闭、GRPO 仍拒绝 ghost_concat。真实 GPU 更新、恢复和短导航验证记录见 `docs/persistent-ghost-gpu-validation-20260911.md`，启动记录见 `docs/persistent-ghost-compiled-10k-operations-20260911.md`。论文按完整双层结构组织，但不能以这一单阶段配置宣称完整组合已联合验证。
- 当前原生 CLS 世界模型入口配置为 `configs/nwm/raenwm_mp3d_fresh_cls.yaml`：冻结 DINOv2 表征上的 `CDiT-B/2` 条件生成，联合预测 CLS+256 patch，采用线性路径速度场/流匹配和欧拉采样。`runtime.py` 禁用 RGB 解码，但不能因此将其归类为直接回归式 JEPA；技术定位与原始 RAE-NWM 论文差异见 `paper/notes/world-model-landscape-and-section-b-20260905.md`。
- `vlnce_baselines/ss_trainer_ETP_R1.py`: SFT/监督训练相关 trainer。
- `vlnce_baselines/GRPO_trainer_ETP_R1.py`: GRPO/RFT 相关 trainer。
- `vlnce_baselines/models/R1Policy.py`: ETP-R1 策略网络封装。
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py`: 核心多模态模型。
- `vlnce_baselines/common/env_utils.py`: Habitat 环境创建和并行环境工具。
- `habitat_extensions/task.py`: VLN-CE/RxR 数据集和任务扩展。
- `habitat_extensions/habitat_simulator.py`: 对 Habitat-Sim simulator 的项目定制封装。

## Data, Configs, And Artifacts

论文主图当前以 `paper/figures/pilot-main-v10-editable.drawio` 为唯一编辑主版本，包含用户手工修改；不可用旧 HTML/SVG 构建脚本覆盖。2026-09-14 已整理 Input 栏的输入、编码器和特征输出，直接用本机 draw.io 导出验证；版本变更记录见 `paper/notes/pilot-svg-redraw-v10-20260914.md`，备份在 `paper/figures/archive/`。

2026-09-20，核对主工作区评分器及独立 `ETP-R1-stage2-e24` 工作区的预测、评分和部署代码，整理右侧第二阶段主图布局：上部画patch驱动的后继查询与共享世界模型，下部突出三轮“指令—未来融合／跨候选比较”，底部有界分数回加。详见 `paper/notes/pilot-stage2-main-figure-layout-20260920.md`；本轮仅分析，未修改draw.io。独立工作区新增的 `all_present` 候选上下文是可选消融，当前默认部署仍采用future有效掩码。

同日后续按用户确认的简化方案，已将Stage 2核心直接加入上述draw.io主文件：三行候选内融合、共享语言输入、联合候选比较、整组反馈及×3 rounds，均为可编辑原生元素。当前仅画核心，未接入查询生成及最终分数支路。脚本 `paper/figures/add_stage2_core_native.py` 拒绝重复添加，备份位于archive；精修及验证见 `paper/notes/pilot-stage2-native-core-20260920.md`。

同日17:25，按用户手工重画的最新版完成润色：当前Stage2采用紧凑三行特征包→融合→共用竖向Joint attention→三路输出及×N，无旧版反馈回路。统一字体、纯色填充和绑定端点的水平箭头，保留用户布局；具体备份及预览见上述原生核心记录。后续应以最新draw.io为准，不运行旧生成脚本重建。

同日后续模块名称已改为Instruction–Future Fusion与Cross-Candidate Attention；17:45在重复框外接入Residual Scoring、有界δ、基础分数直通加号及最终s′=s+δ，保留用户的紧凑布局。最新细节见 `paper/notes/pilot-stage2-native-core-20260920.md`。

2026-09-21，基于用户最新图稿（评分模块已手工改为Action Head），已原生补绘DINO-CWP、方向距离网格、q1查询和世界模型返回的未来特征分发支路；原717个元素未修改。备份、连接语义及导出验证见 `paper/notes/pilot-dino-cwp-native-20260921.md`。最新唯一编辑源仍为draw.io，不能用旧生成脚本覆盖。

同日按用户要求表现递归前瞻深度，基于最新手绘连线增加D-step轨迹、预测patches到CWP的回路、终端z_{d*}标记；下方融合层数改标L。只读核对独立recursive-future-rollout工作区确认终端/最深有效层聚合，但其本地深度检查仅允许1/2/3，图以D表示而未写1–5已验证。详见 `paper/notes/pilot-lookahead-depth-20260921.md`。

README 要求准备 Matterport3D 数据，目标结构是 `data/scene_datasets/mp3d/{scene}/{scene}.glb`。当前项目内 `data/scene_datasets/mp3d` 是软链接，指向本机已有数据 `/home/gwl/project/dataset/mp3d_unzipped/mp3d`，跟随软链接可看到 90 个 `.glb` 场景。

当前训练机和测评机的 `data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/` 均包含 `train`、`val_seen`、`val_unseen` 和 `test`。其中 `test/test.json.gz` 有 3,408 个 episode、覆盖 18 个场景，但没有 `test/test_gt.json.gz`，episode 也不含 `goals` 或 `reference_path`；因此只能生成官方格式的测试轨迹，不能像 `val_unseen` 一样在本地计算完整导航指标。EvalAI 的 VLN-CE Challenge 719 已于 2026-01-31 结束并冻结，官方公告建议今后的方法比较报告可在本地完整评测的 `val_unseen`；当前不能依赖该服务器为新模型产生 test 指标。笔记本工作区当前没有 `data/datasets/`。

`extra_files.zip` 的内容此前已解压并合并复制到项目根目录。2026-07-11 使用校验和模式确认 `/home/gwl/project/etpr1/dataset/extra_files` 中的全部文件都已在正式工程中且内容一致，随后删除这份约 21GB 的重复目录。关键资源继续保存在正式工程：

- `pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt`
- `data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth`
- `data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth`
- `data/logs/checkpoints/release_rxr_dagger/store/ckpt.iter20600.pth`
- `data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth`
- `pretrain_src/img_features/CLIP-ViT-B-32-views-habitat.hdf5`
- `pretrain_src/img_features/ddppo_resnet50_depth_features.hdf5`

论文的联合预训练使用 5 类轨迹—指令数据。当前 `mix_pretrain_server.json` 已逐项加载对应的本地转换文件，数量与论文中的约数一致：

- R2R train：`R2R_train_enc_xlmr.jsonl`，14,039 条（论文写 14K）。
- Prevalent：`R2R_Prevalent_enc_xlmr.jsonl`，1,069,620 条（论文写 1M）。
- Prevalent Gemini Aug：`R2R_Prevalent_gemini_aug_enc_xlmr.jsonl`，1,046,280 条（论文写 1M）。
- RxR train：`rxr_train_guide_xlmr.jsonl`，79,467 条（论文写 80K）。
- RxR-Marky：`rxr_marky_enc_xlmr.jsonl`，1,001,331 条（论文写 1M）。

5 个训练文件合计 3,210,737 条。2026-07-10 已逐行检查：JSON 解析错误 0、关键字段缺失 0、任务类型编码错误 0。所有训练轨迹共引用 7,635 个不同视点，这些视点在 RGB 和深度 HDF5 特征中的缺失数均为 0。用于选择预训练 checkpoint 的 R2R val unseen（2,349 条）和 RxR val unseen（13,652 条）也都存在、可解析，视觉特征缺失均为 0。

预训练初始化资源也在本地：`bert_config/xlm-roberta-base/pytorch_model.bin` 是文本编码器权重；RGB 和深度输入已预计算为上述两个 HDF5，因此运行联合预训练时不需要现场重新下载 CLIP 或深度编码器权重。若要重新生成深度特征，本地另有 `data/ddppo-models/gibson-2plus-resnet50.pth`。

预训练图像资源链路：`precompute_img_features/save_img.py` 按 connectivity 中 90 个 MP3D scan 的 10567 个 included viewpoint，从 `data/scene_datasets/mp3d/{scan}/{scan}.glb` 渲染每个 viewpoint 的 36 个离散视角，生成原始 RGB/Depth HDF5；`extract_rgb_features.py` 再从 RGB HDF5 提取 `CLIP-ViT-B/32` 特征，`extract_depth_features.py` 从 Depth HDF5 提取 DDPPO ResNet50 depth 特征。`pretrain_src/run_pt/mix_pretrain_server.json` 实际读取的是 `CLIP-ViT-B-32-views-habitat.hdf5` 和 `ddppo_resnet50_depth_features.hdf5`，不是直接读取 `save_img.py` 输出的原始图像文件。2026-07-08 已验证这两个特征文件各有 10567 个 key，和 `pretrain_src/datasets/R2R/connectivity` 完全一致，缺失 0、额外 0。

当前 RGB 视觉链路分成两部分：离线预训练读取上述 512 维 CLIP HDF5；在线 SFT、GRPO 和评测则由 `R1Policy.py` 的 `CLIPEncoder` 对 Habitat 的 224×224 RGB 观测实时提取 512 维特征。两条链路必须使用相同的编码器语义和投影权重，否则预训练与在线阶段会出现视觉特征分布不一致。当前 `BinaryDistPredictor_TRM` 路点预测器的 RGB 分支已被注释，实际只使用 128×4×4 的深度特征，因此更换 RGB 编码器不需要修改或重训路点预测器。

## Tests And Verification

现代测评运行时的隔离检查位于 `tests/test_runtime_contract.py`、`tests/test_runtime_behavior.py` 和 `tests/test_modern_runtime_imports.py`。必须在测评机的专用容器与专用环境中通过 `scripts/etpr1_rae_runtime_exec.sh` 运行。本机不运行这些测试。下面两条命令是 2026-06-08 对本机旧 CLIP 环境做过的历史验证记录，不是新 RAE/DINOv2 方案的测试入口：

```bash
conda run -n etpnav python -c "import habitat_sim, habitat, habitat_baselines; print(habitat.__version__, habitat_sim.__version__)"
conda run -n etpnav python run.py --help
```

2026-06-08 已验证：

- `run.py --help` 正常。
- `import run; import vlnce_baselines.ss_trainer_ETP_R1; import vlnce_baselines.GRPO_trainer_ETP_R1; import habitat_extensions.task` 正常。
- `baseline_registry.get_trainer('SS-ETP-R1')` 和 `baseline_registry.get_trainer('GRPO-R1')` 都能找到 trainer。
- `run_r2r/iter_train.yaml` 和 `run_rxr/iter_train.yaml` 能正常解析到对应任务配置和数据路径。

RAE/DINOv2 分支的所有验证必须在测评机 `gwl-etpr1-rae` 容器和 `etpr1_rae` 环境中进行。正式实现时依次验证环境导入、RAE CLS 数值一致性、小规模与全量 HDF5、预训练 MLM/SAP、SFT、GRPO 冻结状态以及 R2R/RxR 单 episode；不得用本机旧环境的导入结果代替。

2026-07-15 已完成正式验收：完整测试为 `269 passed, 3 warnings`；RAE encoder-only 对照结果为 `max_abs=0`、`cosine=0.9999999404`；全量 HDF5 含 10,567 个 `[36,768] float32` 视点且完整性错误为 0；RAE 正式 smoke 的 15 个阶段全部通过。原 CLIP 的 `ckpt.iter25000.pth` 也在现代运行时完成一个 R2R `val_unseen` episode，权重无未处理 missing/extra layer。详细命令、性能和远端日志见 `docs/rae-dinov2-eval-host-validation.md`。

2026-07-15 又在完整 3,210,737 条真实联合预训练数据上完成单卡 RTX 4090 的 batch 实测。float32 下，`batch_size=32` 虽能完成 20 次更新，但峰值显存达到 23,684 MiB，距离整卡上限只剩约 880 MiB，不适合作为长训练配置；`batch_size=16` 的峰值为 17,692 MiB；`batch_size=16 + gradient_accumulation_steps=8` 能完成真实累积更新，峰值为 17,842 MiB，对应有效 batch 128。正式配置现已使用后者。持久化日志位于测评机 `data/logs/rae_dino_batch_test/`。

2026-07-15 已补齐联合预训练断点续训和 tmux 长任务托管：每个可恢复点同时原子写入模型与包含优化器、混合精度缩放器、全局步数、数据混合步数和随机状态的 `train_state`；恢复时会拒绝 batch、梯度累积、GPU 数或模型配置不一致的状态。默认保留最近 3 对完整状态，并每 25,000 步保留一个模型里程碑。真实模型已完成“第 1 步保存、由新进程恢复并完成第 2 步”，状态含 484 组优化器参数；全量测试为 `276 passed, 3 warnings`。操作手册见 `docs/rae-dinov2-pretrain-operations.md`。

2026-07-15 已增加预训练最佳模型：每次验证将 R2R/RxR 的 MLM 准确率等权平均、SAP 准确率等权平均，再把两个均值相加；仅当总分严格提高时，通过硬链接更新 `best/model_best_step_<step>.pt`，并原子更新包含四个原始准确率、两个均值、总分和步数的 `best_metrics.json`。真实一步 GPU 验证确认硬链接与指标文件正确，全量测试为 `280 passed, 3 warnings`。最佳模型用于下游选择，断点恢复仍使用最近 3 对完整 checkpoint。

2026-07-15 正式长训练从头运行约 4 小时后，在全局第 11,195 步附近停止。直接异常是 DataLoader 子进程输出 `free(): invalid size` 后被 `SIGABRT` 中止；主进程随后报 `DataLoader worker exited unexpectedly` 并以退出码 1 结束。宿主机无 OOM、NVIDIA Xid 或重启记录，容器共享内存、磁盘和内存均充足；RGB/深度 HDF5 的 10,567 个视点也已在同一环境完整顺序读取且错误为 0。因此当前故障域集中在 `n_workers=1` 的多进程数据加载与 HDF5/NumPy/PyTorch 本地库交互，精确到具体本地库函数的根因仍待复现或 core dump 确认。最近完整可恢复状态是第 10,000 步。

本地完整 checkpoint 评测记录见 `docs/ETP-R1_checkpoint_eval_comparison.md`，里面包含单卡评测命令、R2R/RxR 四个 checkpoint 的指标、结果文件路径和并行环境数量调整记录。ETP-R1 与原版 ETPNav 的代码差异分析见 `docs/ETP-R1_vs_ETPNav_diff_analysis.md`。

## Current Caveats And Open Questions

- 2026-09-09 已按用户要求把`world_exact_select`接为主工作区ghost_concat训练/评测默认：全局auto在低级native CLS ghost配置解析为新模式，当前ghost YAML与CLI明确默认新模式；环境按已访问历史位姿提供全景，runtime维护与reset/pause一致的历史并调用目标对齐预测。旧E24保持front并拒绝新源快照合同。旧front权重仅允许非续训迁移，新/旧上下文跨模式恢复被拒绝；新检查点保存panorama元数据。87项相关测试、双环境19查询（10倒序端点）及2步联合训练通过，479策略张量与CLS映射更新、冻结视觉未变。说明见`docs/ghost-concat-fusion-implementation-20260908.md`，训练机日志`data/logs/panorama_default_20260909/`；无长任务或导航性能结论。
- 2026-09-09 全景上下文预测质量实现已完成：11场景22短片段、288实际查询；6场景162目标比较7方案，预先锁定`world_exact_select`，另5场景126目标×3噪声复核，场景等权CLS0.6950→0.8267、图块0.5707→0.7049、RMSE下降24.46%，5场景均提升、108/126目标改善。>90度组CLS0.6191→0.8413。44项测试和真实可复用接口/缓存检查通过，推荐配置`configs/nwm/panorama_context_quality_best.yaml`，完整报告`docs/nwm-context-quality-results-20260909.md`。世界模型与策略权重未训练，原导航默认链路未切换，本次没有导航性能结论。
- 2026-09-09 该实现最初位于用户授权的独立全景上下文实现分支`feature/nwm-panorama-context`，两端目录后缀`ETP-R1-wm-context`。新增`panorama_context.py`实现全景取视图、固定目标位姿及虚拟源条件，`panorama_runtime.py`提供真实已观测全景的缓存与批量预测，独立质量基准按11场景中的6/5场景选择/复核，规则见`docs/nwm-context-quality-plan-20260909.md`。世界模型4帧没有独立时间顺序编码，重排本身近似不改变输出；可变因素是方向图像和参考源条件。正式产物保留训练机新目录`data/logs/nwm_context_quality_20260909/formal_v3/`，已完成的预测质量结论见结果报告，不测试导航SR/SPL。
- 2026-09-09 进一步核查虚拟时间倒序发现关键架构事实：75k模型4帧使用完全相同的冻结空间位置编码，没有帧顺序/每帧时间标记，拼接后作无时间掩码交叉注意力。真实样本固定条件噪声仅倒序，预测余弦0.9999549；单次FP32前向最大差5.72e-6、余弦1.0。应修正“倒序本身让现有模型识别前进运动”的解释；主要可变因素是选视角、新参考源及目标条件。新算法推导见`docs/nwm-virtual-trajectory-algorithm-20260909.md`，顺序诊断退出0，未改生产模型或训练。
- 2026-09-09 用户进一步建议身后目标倒序上下文：已补充到全景方案，需同时倒序图像/位姿并将源改为原最早帧，重新计算目标条件；仅倒序前置图不能降低转角。样本0选侧后视角后倒序可使历史相机前向位移由-0.125变+0.125米；样本19含0度锚点，直接倒序会将小转角重新变成132.46度，应拒绝不连续序列。均为几何分析，未实测新预测。
- 2026-09-09 已分析按查询方向选择全景历史：建议低级4个时间点保存12方向图像，每个目标选择末帧最近的相机偏角并在4帧保持同一相对偏角，固定目标位姿、同时变换源平移和转角。113查询仅几何换算可将剩余角度压至≤15度、均值7.95度；不是预测验证。需处理真实多视角采集/按需编码成本、向后观看的历史运动分布、teleport分段及检查点上下文格式，方案见`docs/nwm-panorama-context-proposal-20260909.md`，未实现或运行新实验。
- 2026-09-09 补查大转角查询来源：同4条路线全量113个有效查询中50个>90度，31个与选中动作匹配的有效查询中仅1个>90度；162度床头候选未被选中，合并前已162度、合并后162.46度。源位姿差0；主要为360度候选共享前置4帧上下文。teleport每次清空历史并设朝向0度，36个就绪上下文中3个含0度锚点与后续不同朝向；查询理想位姿与15度/0.25米离散控制终点存在差异，31对均值0.185米/3.21度，因果影响未验证。详见`docs/nwm-query-angle-analysis-20260909.md`及配套几何图、全部查询/动作JSON。训练机短评测退出0，无生产逻辑变更。
- 2026-09-09 已完成世界模型实际导航调用/可视化诊断：训练机1800步策略跑4条路线、34个真实查询（同一场景），原工程与接入实现对同一真实输入及噪声重放逐元素一致；训练侧坐标公式与导航条件最大差5.56e-8，全部目标可通行，上下文重渲染特征平均绝对差0.00129。预测CLS/复用源CLS对目标余弦为0.6104/0.5050，大转角查询质量较差；同噪声50步未优于10步（0.5892/0.6104）。未确认调用实现bug，不能将单场景结果推广到全量。可视化与证据见`docs/nwm-navigation-call-diagnostic-20260909.md`，新增采集/解码/重放脚本为`scripts/{diagnose,render,replay}_nwm_navigation*.py`，原始特征保留训练机`data/logs/nwm_call_diagnostic_20260909/`；23项相关测试通过，没有改生产调用或启动训练。
- 2026-09-09 已核验ghost concat联合batch8实验完成：训练9月8日17:26:59–23:25:13（5h58m），全部10份1839路线评测于9月9日05:24:12结束，从训练启动累计11h57m；实际评测累计11h21m，二者并行。最高SR在1800步为63.7847%，仅比历史基线63.7303%多1条成功，SPL53.8192%低1.7862点；2000步62.3165/53.5917，尚无可靠综合提升。源位姿414201有效查询错位0。1800/2000策略479张量更新、冻结视觉参数未变，双学习率正确；优化器实际1799/1999步与一次AMP跳步一致，不是未完成训练。报告`docs/ghost-concat-joint-results-20260909.md`，峰值模型和原始结果保留远端，本次未启动新实验。
- 2026-09-08 已按用户批准补齐并启动ghost concat联合微调：双卡各4、全局batch8、原始14200权重起步、策略lr2e-6/融合lr1e-5、E24关闭、共2000步。用户进一步要求所有检查点评测，测评机监听200到2000共10个检查点，每个1839条路线。实验根两机均为`data/logs/ghost_concat_joint_bs8_20260908/`，管理器`scripts/manage_ghost_concat_joint.py`支持start/status/resume。两机113项相关测试及训练机533项回归通过；双卡4步和从2恢复至4真实验证成功，策略479张量变化、冻结视觉参数未变，4组优化器LR与step/scheduler正确；预检2/4模型按序同步评测成功。启动时训练双卡进程存活并推进，评测等待200。参数、日志与恢复范围见实现文档联合训练小节，尚无完整性能结论。
- 2026-09-08 用户提出从14200做可训练策略的ghost concat联合微调、全局batch8。已检查当前代码并在训练机做两步CPU合成联合梯度验证：共享融合模块保留观测/策略/MLP梯度且原始输入不变；这不是完整联合导航验收。当前factory显式要求冻结、CLI也强制freeze=True，尚不能直接启动联合模式；batch是每rank，双卡全局8需各4。建议先补可选联合训练、策略/融合分组学习率（建议起点2e-6/1e-5）及真实双卡/恢复验证，再跑2000步固定400/1000/2000评测。细节见`docs/ghost-concat-joint-training-review-20260908.md`。本次仅审查与建议，未改生产训练逻辑或启动正式训练。
- 2026-09-08 已按用户最新确认实现批量ghost拼接残差融合。修改前快照为`a7ead28`；新模式`rgb_fusion_type=ghost_concat`，三层1536→1536→1536→768、最后一层零初始化，无显式特征相减、无门控、无全景复制。纯观测图更新后，将所有环境的当前有效ghost与预测raw CLS批量拼接、单次MLP前向并回填临时图输入，不跨步保存预测。新旧adapter合同隔离，默认旧模式不变；新模式仅支持SS-ETP-R1，GRPO明确拒绝。训练机CPU回归529通过2跳过、测评机相关119通过1跳过；真实单/双卡2步与冻结审计通过，测评4路线验证每步一次全景、最多一次MLP且原图不变。64ghost模块基准批量约0.155ms、逐行9.091ms，不代表整套导航提速。实现/命令见`docs/ghost-concat-fusion-implementation-20260908.md`；未启动正式长训练。
- 2026-09-08 用户要求分析改变RGB注入结构，已直接复查全景编码、candidate→ghost更新和图输入组装，设计见`docs/rgb-ghost-local-injection-design-20260908.md`。建议保留纯观测图，使用冻结全景编码器的单候选对照差值，在图输入处对当前有效ghost做零初始化、有界临时修正；第一版不跨步缓存预测，不改STOP规则、不启用E24。明确预测CLS不能因同为768维就直接加到多模态全景嵌入上，也不能承诺图注意力之后STOP不变。需先做容量对照与预测/真值信息价值诊断；本次仅写方案，没有修改生产模型或启动实验。
- 2026-09-08 15:16 已核验RGB-only三步实验全部完成：3组2000步训练、12份1839任务完整评测、4份冻结审计，19项验收重新通过；实际结束时间09:35。完整分析见`docs/rgb-only-optimization-analysis-20260908.md`。没有有效RGB配置超过历史SR63.7303或本轮基线63.5672。A1000/2000为SR61.6639/62.3709；B1000/2000均63.3496，B2000 SPL54.8832；C1000/2000为63.2409/63.1865。旧5200关闭增量alpha0最高SR63.7303但SPL53.9829，不算有效RGB提升；alpha1为63.5128/54.2061。源修复前后1839逐条结果完全相同，九份WM评测共370818有效查询错位计数均0，不能将此前退化归因于该边界错误。冻结保留能力但未取得净增益，简单对齐未验证有效；当前保留原始14200为使用参照，后续优先预测信息价值与ghost局部注入诊断。本次未启动新实验。
- 2026-09-07 用户改为要求全自动执行，停止人工持续检测，具体分析等再次发起。`scripts/manage_rgb_only_pipeline.py start/status`已部署两机，可接管现有训练、评测、发布和收集进程，重复启动验收保持同一组PID。最终收集器已切到`--collect-only`：仅在3组训练、12评测、4冻结审计全部通过后生成测评机实验根的`final_report.md`和`final_summary.json`，不提前给出原因分析或模型推荐。人工监控exec循环已停止；操作命令和结果位置见`docs/rgb-only-optimization-execution-20260907.md`。
- 2026-09-07 17:28 三步优化中途实测：本轮基线SR/SPL `63.5672/55.5275`，历史基线`63.7303/55.6054`，同SHA与1839 ID配对净差-3条；旧源5200完整复测`63.5128/54.2061`精确复现原指标。其41520次有效候选查询中源位姿错位为0，因此先前合成复现的边界错误不能解释这次旧5200退化。当前A已到约1104且1000模型同步，B约188，C等待；alpha0评测进行中。自动训练完成证据发布、4份冻结审计和19项产物验收后的最终报告已部署，尚未形成新模型性能结论。
- 2026-09-07 用户已授权执行 RGB-only 三步优化，执行记录见 `docs/rgb-only-optimization-execution-20260907.md`。源位姿修复、冻结底座、预测侧共享冻结CLS适配层及有限实验调度已通过Git同步；代码截至 `eca64f3`。两机源位姿46项及冻结相关97项测试通过，真实B/C单卡、B双卡和A/C单卡batch16预检通过；643个导航张量冻结审计均一致。正式产物位于两机 `data/logs/rgb_only_optimization_20260907/`：训练机GPU0 A→C、GPU1 B（各单卡batch16/2000步），测评机依次完成基线、旧5200桥接、四强度以及ABC固定1000/2000完整评测。启动不等于性能已提高，最终指标、源位姿错位比例和配对结论等待完整结果。既有论文及未提交修改保持原状。
- 2026-09-07 已补齐 RGB-only 三轮各 50 点的最终审计，见 `docs/rgb-only-navigation-diagnosis-20260907.md` 和 `docs/diagnostics/rgb-only-audit-20260907-metrics.json`。TensorBoard 的 `start_iter14200_no_rgb` 只是原始 14200 指标的水平参照线，不是同预算无注入续训。旧低级后补测 `iter9200` 达到 SR/SPL `64.0022/54.3409`；梯度修复后 `iter1000/5200` 并列最高 SR `63.5128`，5200 的 SPL 更好 `54.2061`。三轮均值 SR 几乎相同，主要问题是大量修好与破坏任务同时发生、路径效率下降。新发现低级碰撞转向后缓存末帧朝向与查询当前朝向可能不一致，测评机 CPU 合成位姿复现已确认，真实发生率与性能影响未测，生产代码尚未修复。优先源位姿合同核验、已有模型 alpha 消融、同预算无注入续训与冻结底座训练，再考虑对齐及全景编码后 ghost 局部注入；本轮没有启动新训练或完整导航评测。
- 2026-09-05 已完成 RGB-only 跨工程与远端结果审计，详见 `docs/rgb-injection-navigation-audit-20260905.md`。核查时训练机修复后低级实验已保存并同步 `iter5000`，测评机已完整评完 `iter200`–`iter3800`，正在评 `iter4000`。同一 DINO 无 WM SFT 基线为 SR/SPL `63.7303/55.6054`；修复后当前最佳 `iter1000` 为 `63.5128/53.7450`。旧高层 native RGB 最佳 `64.2197/53.9265`，逐 episode 对照仅净增 9 条（修好 128、损失 119）。低级训练门均值在 `iter5000` 达到 `0.994`，候选注入覆盖率约 81%；`back_algo=teleport` 使低级上下文在每个高层动作边界清空，不能等同于旧高层跨决策历史。当前融合的观测输入是经过 residual MLP 的 `nav_cls`，预测输入是反归一化后的原始 DINO CLS；直接相减符合现有设计，但对齐效果仍待验证。优先建议已有 checkpoint 的注入强度消融、同预算无注入继续训练对照、冻结底座仅训 adapter；本次未启动新实验或改动远端任务。
- 2026-09-04 提交 `ea860aa` 已修复低级上下文 SFT 的导航 CLS residual MLP 梯度丢失。根因是 `LowLevelContextSynchronizer` 在整个 rollout 的 CUDA autocast 区域内，先以 `torch.no_grad()` 调用同时包含 raw CLS/patch 和 navigation CLS 的接口；这会让 autocast 缓存无梯度版本的 residual MLP 权重，随后 waypoint 前向复用该缓存并失去梯度。编码器现提供完全不经过 residual MLP 的 raw-only 接口，低级同步器只能调用该接口；trainer 还会在每次优化器更新前拒绝 residual MLP 只有零锚、没有任何真实非零梯度的更新。训练机针对性测试 `59 passed`；完整测试除 3 个只能在测评机固定 Habitat 容器路径运行的 runtime-behavior 用例外为 `546 passed`。真实单卡低级两步 probe 的六项梯度全部非零；真实双卡 DDP 单步中两个 rank 的六项梯度全部非零且归约后一致，没有保存 checkpoint。旧低级 checkpoint 仍属于受影响产物，不能视为满足该梯度合同。
- 2026-09-04 正式修复后重训的 `iter200` 到 `iter1400` 已完成参数级复核。七个相邻 200 步区间中，CLS residual MLP 的三个 bias 每次均为 `768/768` 元素变化，三个 weight 每次也有至少 `589818/589824` 元素变化；相对导航基座的整体 L2 漂移从 `iter200` 的 `0.619808%` 单调增至 `iter1400` 的 `1.461174%`。旧错误实验同两点只有 `0.002045%` 和 `0.014300%`，且三个 bias 始终逐位不变；旧 weight 的缩放系数约 `0.999859--0.999865`，与学习率 `1e-5`、权重衰减 `0.01` 在 1,400 步下的纯衰减系数 `0.999860` 一致。新 weight 变化的非缩放分量占更新范数约 `97.2%--99.9%`。结合正式训练连续超过 1,400 步且逐步非零梯度硬检查从未触发，可确认修复在各 checkpoint 区间持续生效，而非只在冒烟或首步生效。
- 2026-09-04 测评机工作区的 `origin` 已从旧的不可达 Tailscale 地址校正为项目固定的训练机专线中央仓库 `ssh://gwl@10.10.10.1/home/gwl/git/ETP-R1.git`；确认工作树干净后，已通过 `fetch` 和 `ff-only pull` 快进到 `ea860aa`。
- 2026-08-14 已确认测评机单卡续训 `pretrain_resume_source_250000` 的停止原因是宿主机 CPU 内存耗尽，不是显存溢出。任务使用提交 `94f4372`，配置为 `n_workers=2`、`pin_mem=true`、`thread_prefetch=false`；它从第 250,000 步运行到第 305,000 步，在 RxR MLM 验证开始后由内核 OOM killer 以 `SIGKILL` 结束。内核现场有 7 个约 18 GiB RSS 的 Python 进程，正好对应 1 个训练主进程、MLM/SAP 共 4 个长期训练 worker 和当前验证的 2 个 worker；容器累计内存峰值为 65,468,919,808 字节，2 GiB swap 已耗尽。`R2RTextPathData(in_memory=True)` 会让每个 worker 独立、单调填充 RGB/深度视点缓存，完整缓存约 1.36 GB（十进制），同时 worker 通过 Linux `fork` 继承装有 321 万条 Python 记录的约 18 GiB 主进程，长期访问会增加写时复制的私有页。它是有上界的缓存和进程复制，不是计算图无界泄漏，但实际表现为缓慢增内存并最终 OOM；验证额外 worker 构成最后峰值。最近完整恢复点是第 302,500 步。此前已稳定长跑的 `n_workers=0`、`pin_mem=false`、`thread_prefetch=true` 路径没有这些子进程；本次只读诊断未恢复任务、未修改训练代码或远端产物。
- 2026-08-14 已实现保持 `n_workers=2` 的有界内存路径：321 万条 JSONL 改为约 24.5 MiB 的 mmap 行索引并按需解析；worker 使用 `spawn`；RGB+深度特征采用 256 MiB/worker 的按字节 LRU；预取降为 1；`pin_mem=false`；验证使用 `val_n_workers=0` 和零特征缓存。真实数据在 CUDA 先初始化的顺序下完成 2,000 micro-batch，500--2,000 batch 的 cgroup 内存稳定在约 22.82--22.88 GB，进程私有内存为主进程约 1.36 GiB、四个训练 worker 各约 1.38--1.43 GiB，退出正常。惰性/eager 的真实 R2R 首中尾样本和完整输入逐项一致，针对性测试 `44 passed`。测试中还发现 PyTorch `persistent_workers=true` 会在提前关闭时触发本地库 `SIGABRT`，最终配置已关闭。正式预训练现已在测评机从第 302,500 步恢复；2026-08-15 本次只读核验时已到约第 462,000 步，训练进程正常。当前最佳是第 452,500 步：MLM 平均准确率 `0.8666867801`、SAP 平均准确率 `0.8011502767`、联合选择分数 `1.6678370568`；对应 R2R/RxR 的 MLM 为 `0.8199916701`/`0.9133818901`，SAP 为 `0.8063005534`/`0.796`。新配置的每轮验证稳定约 166--172 秒（R2R 约 35 秒，RxR 约 131--136 秒）；旧配置共用 `n_workers=2` 时约 49 秒。差异主要来自新设置显式使用 `val_n_workers=0`，使验证的 HDF5 读取、样本构造和 batch 整理与 GPU 前向串行；这是为避免旧路径在验证 worker 创建时再次触发内存峰值的保守配置，不是训练 worker 卡死或验证逐轮退化。
- 2026-08-15 对比当前 raw-CLS 实验与上一次完整 `rae_dinov2_cls_mlp` 实验：两者不是同配置复跑。旧实验使用经 `stat.pt` 归一化的 DINO CLS，再经可学三层 `768→768→768→512` 投影和 `512→768` 映射；当前实验为对齐 ETPNav，改用未应用 `stat.pt` 的 raw CLS 和单层 `768→768` 映射。旧实验有效 batch 为 128；当前实验前 10,000 步为 128，之后主动允许改为 64，因此同为 500,000 次参数更新时约只看到旧实验一半的训练样本。分数差距在前半程已存在：旧实验第 250,000 步最佳 `1.666969`，当前实验第 252,500 步为 `1.646896`；故后续转移到测评机、OOM 和 `val_n_workers=0` 不是主因。验证子集由固定 seed 选取，但 MLM mask 和 SAP 终点类型仍是每轮随机生成，因此单点最佳分数也包含一定验证噪声。现有证据能说明主要差异来自视觉链路和有效 batch，但没有单变量实验能精确分解两者各自的贡献。
- 2026-08-15 当前训练机 R2R SFT 正式入口为 `scripts/run_rae_r2r_sft_server_job.sh`，使用双卡 DDP、每卡 8 个 Habitat 环境、每卡 batch 8、梯度累积 1，即每次优化器更新全局约 16 条轨迹。学习率 `1e-5`，预热 500 次；`min_lr_ratio=1.0` 使预热后实际保持恒定学习率。DAgger 教师采样初始比例 `0.75`，每 3,000 次按幂衰减；最长轨迹 15，最长文本 150，waypoint augmentation 开启。DINO 主干、深度编码器和路点预测器冻结，零初始化的 CLS residual MLP 与导航策略其余部分参与训练。上一次 `r2r_sft_formal` 实际使用 15,000 次；当前脚本默认已改为 2,000 次的 `panorama_order` 检查实验，且默认预训练路径仍是 raw-CLS 实验的 `model_best_step_220000.pt`，启动新正式 SFT 前必须明确覆盖为本轮选定的预训练 checkpoint 和所需总迭代数。上一次 50 万步 `rae_dinov2_cls_mlp/model_best_step_452500.pt` 属于已退役的旧视觉接口：其经预训练的 CLS MLP 是 `768→768→768→512`，后接 `img_linear 512→768`。当前 SFT 接口则是 raw CLS 上的残差 MLP `768→768→768→768`，再接 `img_linear 768→768`；当前代码会直接拒绝加载旧 checkpoint。若使用旧的 nonvisual-transfer 转换，旧 `rgb_projection` 和形状不匹配的 `img_linear` 都被丢弃；新 residual MLP 与 `img_linear 768→768` 均未经联合预训练，视觉桥接层需从 SFT 开始学习。
- 原版 ETP-R1 的 R2R SFT 发布入口 `run_r2r/main_server.bash` 使用 4 卡、每卡 8 个 Habitat 环境、无梯度累积和 30,000 次优化器更新，因此每次更新的全局有效 batch 是 32 条轨迹；发布的下游 checkpoint 是从该预算内选择的 `ckpt.iter25000.pth`。原始训练循环每个 iteration 只执行一次 `optimizer.step()`，`IL.batch_size: 1` 并不代表全局 batch 为 1。当前双卡、每卡 8 环境、15,000 次配置的有效 batch 为 16、总轨迹预算约为原版的四分之一。RxR 原版则是 4 卡、每卡 6 环境、30,000 次，有效 batch 24。
- 2026-08-15 18:12（Asia/Shanghai），已在训练机从上述旧最佳的转换产物 `rae_dinov2_etpnav_cls_768_legacy_base_transfer/model_step_452500_nonvisual_transfer.pt` 启动新的双卡 R2R SFT。实验名为 `rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft`，输出目录为 `data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815`，总迭代数 15,000；checkpoint 同步目标也已隔离到测评机对应的新目录。启动提交为训练机分支 `feature/world-model-migration` 的 `97d397f`。两个 rank 均成功加载转换 checkpoint，连续运行到第 200 次更新后已成对写出约 1.53 GB 的 `ckpt.iter200.pth` 和约 3.00 GB 的 `train_state.iter200.pth`，随后继续进入下一轮更新；模型 checkpoint 也已通过直连同步到测评机的新目录。保存后两张 A6000 显存约 21.7/20.1 GiB、利用率约 62%/61%，未见缺失键、形状冲突、异常退出或显存溢出。启动日志为 `supervisor_server/start_20260815T181211.log`。
- 2026-08-15 18:33（Asia/Shanghai），测评机已启动本次 SFT 的专用评测 watcher，日志为 `data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815/eval_watch_val_unseen/watch.log`。启动时已收到 2 个 SFT checkpoint、结果数为 0；日志明确以 `reason=blocking_project_task` 等待仍在运行的预训练，没有创建评测 `run.py` 进程。18:44 收紧进程匹配条件以排除长期 TensorBoard 后重启，当前 PID 为 `1489031`；预训练退出后还会检查 4090 显存不超过 1 GiB，才按迭代正序逐个进行完整 R2R `val_unseen` 评测。
- 2026-08-15 18:45（Asia/Shanghai），训练机已启动“当前 SFT 完成后使用测评机最终预训练最佳模型再训一次”的接力 watcher，PID 为 `983219`，日志为 `data/logs/rae_dinov2_etpnav_cls_768/eval_best_sft_followup_monitor/watch.log`。部署时当前 SFT 仍正常运行，watcher 状态为 `reason=current_sft_running`。测评机预训练当时的临时最佳为第 465,000 步、联合分数 `1.6718887749`，但 watcher 不提前固定该点；它只会在测评机第 500,000 步模型和训练状态存在且 supervisor 为 `exit_code=0` 后读取最终最佳。预训练进程判定同时匹配 `train_r2r.py` 与输出根目录，不会把长期 TensorBoard 误认为训练进程。下一轮固定沿用双卡、每卡 8 环境、每卡 batch 8、梯度累积 1、15,000 次及当前全部 SFT 调度参数，并使用新的实验与同步目录。
- 2026-08-16 10:21（Asia/Shanghai），两个自动接力均已生效。测评机预训练于 01:11 正常完成第 500,000 步，最终最佳仍为第 465,000 步、联合分数 `1.6718887749`。第一轮 legacy452500 SFT 于 10:09 正常完成 15,000 次并以 `exit_code=0` 退出；训练机接力 watcher 随后复制并校验最终最佳模型，于 10:10 启动同参数第二轮 `rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft`，核验时约到第 227 次且第 200 次 checkpoint 已同步测评机。测评机 watcher 已完成第一轮 SFT 的 60/75 个完整 R2R `val_unseen` 评测，正在评估第 61 个 `ckpt.iter12200.pth`；两台机器均无运行时错误。
- 2026-08-16 10:30（Asia/Shanghai），测评机已启动两轮评测接力 watcher，PID 为 `1623716`，日志为 `data/logs/rae_dinov2_etpnav_cls_768/second_sft_eval_handoff/watch.log`。启动后确认第一轮已有 61/75 份有效结果，日志明确为 `reason=first_eval_incomplete`；第一轮 watcher PID `1489031` 继续单独评测 `ckpt.iter12400.pth`，第二轮 watcher 仍停止，已有 1 个同步 checkpoint、0 份结果，没有提前占用 GPU。第一轮达到 75 份有效 JSON 且 GPU 低于 1 GiB 后，接力脚本会以同样的单卡、8 环境和完整 1,839 episode 参数按正序启动第二轮评测。
- 2026-08-13 已修复离线 RAE/DINOv2 全景生成的俯仰相机漂移；生成器现在使用 ETPNav 的零传感器偏移方案。训练机当前使用的 10,567 视点 `RAE-DINOv2-B-14-RAW-CLS-views-habitat.hdf5` 仍是修复前旧逻辑采集的，本次按用户要求不重新生成。
- 2026-08-13 SFT 检查点保存支持通过 `IL.checkpoint_sync_enabled` 和 `IL.checkpoint_sync_destination` 异步原子同步；两机间使用 `10.10.10.1/10.10.10.2` 的 2.5 GbE 直连。持续监控和批量评测通过 `EVAL.checkpoint_order` 选择正序或倒序，并从同一配置的同步目标推导本地监控目录。
- 2026-08-13 运行状态：训练机的双卡 R2R SFT 已按用户要求正常停止，checkpoint 同步守护进程也已停止；最后一对完整模型/训练状态为第 14,200 次迭代。测评机的 R2R `val_unseen` 监控和正在执行的第 14,200 次迭代 checkpoint 评测也已停止，已完成的 70 份评测结果保持不变。停止后两台机器均无训练/测评计算进程：训练机两张 A6000 的计算显存占用为空，测评机 4090 仅保留约 130 MiB 桌面基础占用。训练机 TensorBoard 和测评机日志查看器不是计算任务，仍保持运行。
- 2026-08-12 训练机实时状态：新的双卡 A6000 联合预训练实验 `rae_dinov2_etpnav_cls_768_raw_cls_20260810` 正在 `/home/gwl/project/etpr1/ETP-R1` 运行，产物位于 `/mnt/data2tb/ETP-R1_data/pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_raw_cls_20260810`。任务从第 10,000 步恢复，目标 500,000 步；只读核验时 TensorBoard 已到第 224,346 步，最近完整恢复点为第 222,500 步，当前最佳模型为第 220,000 步。配置为双卡、每卡 batch 32、梯度累积 1、`n_workers=0`、`pin_mem=false`、`thread_prefetch=true`。两个训练进程持续存活约 48 小时，日志未发现报错，输出盘尚余约 1.4 TB。两卡温度为 85--86°C，但核验时没有处于软/硬件热降频状态。
- 上一次 `rae_dinov2_cls_mlp` 联合预训练已于 2026-07-29 完成 500,000 步；按“R2R/RxR 的 MLM 准确率均值 + SAP 准确率均值”选择出的最佳点为第 452,500 步，总分 `1.6833923785864027`（MLM 均值 `0.8731406970197786`，SAP 均值 `0.8102516815666241`）。原始 `best_metrics.json` 保存在测评机对应实验目录，训练机保留了最佳模型本体。
- `pip check` 会报告 `tensorflow 1.13.1` 声明要求 `tensorboard<1.14`，但 PyTorch 1.9 的 tensorboard 接口要求 `tensorboard>=1.15`。当前选择 `tensorboard==1.15.0`，因为这是项目入口能导入的最低可用折中。
- RAE/DINOv2 已完成一步预训练、单环境 SFT/GRPO 和 R2R/RxR 单 episode 冒烟。正式长训练先后发生三次本地内存层崩溃：第 11,195 步的 DataLoader 子进程出现 `free(): invalid size`/`SIGABRT`；第 80,204 步的 DataLoader 子进程出现段错误；从 80,000 步恢复后又在第 117,624 步由训练 rank 0 主进程直接收到 `SIGSEGV`。第三次宿主机内核同时记录 Python 崩在 `libc.so.6`，没有 OOM、NVIDIA Xid、容器重启或主机重启。三次共同指向 Python 之外的本地内存破坏，故障域优先集中在原 `n_workers=1`、`pin_mem=true` 的多进程数据加载、HDF5/NumPy 读取、跨进程张量传输与锁页内存路径；但由于系统没有保存 core dump，现有证据仍不能精确到某一个本地库函数。2026-07-23 已把正式配置调整为 `n_workers=0`、`pin_mem=false`，从第 117,500 步恢复继续观察。
- 原多进程路径的启动顺序现已进一步确认：`main()` 先初始化 CUDA、构造并搬运模型，再创建训练 DataLoader；`MetaLoader.__init__()` 随即对 MLM、SAP 两个 DataLoader 分别调用 `iter()`，在 Linux 默认 `fork` 模式下产生两个 worker。worker 因而从一个已经初始化 CUDA、载入 HDF5 且约有 80 个线程的大进程中派生。当前单进程训练的 rank 0 常驻内存约 20 GiB，五份 JSONL 原始文件合计约 2.6 GiB，RGB/深度 HDF5 合计约 1.3 GiB。这个顺序是当前最有证据的结构性风险：它同时解释 worker 本地内存崩溃、主进程锁页/跨进程搬运路径崩溃以及问题的间歇性；但在完成分组长压测前仍标记为高概率判断，不当作已精确证明的单一根因。
- 为保留 `n_workers=0` 的稳定性并补回吞吐，已正式启用单进程后台线程预取 `ThreadPrefetchLoader`：后台线程只准备 CPU batch，主线程负责搬到 CUDA，不创建子进程。真实模型从 `train_state_117500.pt` 短测推进到第 117,703 步，没有 DataLoader 子进程或异常；同步单进程路径约 1.48 秒/步，线程预取约 1.27 秒/步，短测提速约 14%。修复后完整测试为 `290 passed, 3 warnings`。正式训练已从第 117,500 步重新恢复，当前配置为 `n_workers=0`、`pin_mem=false`、`thread_prefetch=true`。
- 2026-07-23 已在专用容器中用 CPU 完整读取最新 `train_state_117500.pt`：状态步数为 117,500、数据混合步数为 940,000，含 484 组优化器状态和 2 个参数组，文件可正常反序列化。训练停止时最新进度约为 117,624，因此最多损失 124 个全局步。冒烟此前使用 `n_workers=0`，没有覆盖长跑的多进程数据加载路径。
- 完整数据 batch、断点续训、checkpoint 保留和 tmux 托管均已验证。当前每 2,500 步保存一次，异常断电最多损失最近一个保存间隔；恢复会继续正确的模型、优化器和全局步数，但数据随机采样流不承诺逐样本、逐位复现。
- 旧版 TensorBoard 日志中的 `loss/*` 和 `valid*` 使用进程内从 0 开始的计数器，而 `lr`、`grad_norm` 使用真实全局步数，因此每次断点恢复后 loss 曲线会在 TensorBoard 中错误地回到 0；这只影响展示，不影响训练、优化器或 checkpoint。当前运行中的训练没有重启，由实时校正进程把旧事件映射回真实步数，并让 6006 端口的 TensorBoard 读取校正目录。2026-07-24 已把实时校正从“每 10 秒完整重读全部历史事件”改为“每个事件文件只读一次、后续只消费新增记录”，避免重复解析约 70 万条标量。
- 当前 R2R 与 RxR 验证调用都使用 `setname='_unseen'`，原始事件会把两套 MLM/SAP 指标写入相同标签和相同步数。2026-07-24 已修复校正器：根据训练固定的验证顺序，把同一步第一次出现的指标输出为 `valid_r2r_unseen_*`，第二次输出为 `valid_rxr_unseen_*`，并重建完整历史日志。6006 当前读取 `tensorboard_normalized/live_20260724_split`，不再展示混合标签；原始训练事件和模型不受影响。
- 默认旧配置里还有 `habitat_extensions/config/vlnce_task.yaml` 这类历史路径，但 README 的实际脚本使用 `run_r2r/iter_train.yaml` 和 `run_rxr/iter_train.yaml`，这两个路径已验证可解析。
- 联合预训练配置将 `max_txt_len` 设为 250，`dataset.py` 会截断更长的指令。现有数据中 RxR-Marky 有 38,456 条、RxR train 有 3,097 条超过 250 个词元；这是训练配置造成的截断，不是数据文件缺失。
- 5 类数据的训练就绪 JSONL 都完整，但转换脚本引用的部分原始源文件和 Gemini 标注中间文件未按原路径保存在当前仓库中。因此可以直接运行联合预训练，但若要从原始指令和 Gemini API 输出开始重新生成全部 JSONL，还需要另行补齐源数据。
- 本机 `etpnav` 环境是 Python 3.7、PyTorch 1.9.1、Transformers 4.12.5，不包含 RAE-NWM 所用的 `Dinov2WithRegistersModel`，因此只能参考旧 CLIP 链路。测评机现已建立独立容器 `gwl-etpr1-rae`、独立 `etpr1_rae` 环境和 ETP-R1 自有 Habitat 0.3.3 依赖目录。现代导入链还需要仓库原环境固定的 `boto3==1.20.31`；它只安装在 `etpr1_rae` 中。
- 测评机只有一张 RTX 3090 24GB。现有 ETPNav 任务占用 GPU 时，不得并行启动全量特征生成、预训练、SFT、GRPO 或完整评测，也不得擅自中断 ETPNav。

## Last Reviewed

2026-09-20，为RxR吞吐优化复核训练机进程、原采样曲线、真实双卡性能及显存；主要检查 `ss_trainer_ETP_R1.py`、`R1Policy.py`、视觉编码器、快速配置及基准脚本。最终验证和所有候选结果见 `docs/rxr-throughput-optimization-20260920.md`。

2026-09-15，为用户提供的主图补绘淡紫色 Stage 1，直接核对 `vlnce_baselines/nwm/ghost_concat_fusion.py` 和 `vlnce_baselines/nwm/active_lookahead/candidate_q0.py`：查询来自更新后的 ghost 位置，原节点特征与预测 raw CLS 拼接，经 MLP 形成残差后回填图节点输入；patch 不进入该融合模块。以 `paper/figures/pilot-stage1-inplace-v2.reference.png` 为底图使用 ImageGen，提示词为同名前缀 `.prompt.txt`。仅制作论文视觉稿，不涉及远端实验。

2026-09-14，依据用户提供的网页版 GPT 参考图重绘可编辑论文主图：`paper/figures/pilot-main-v10-editable.html` 内嵌 SVG，提供原图对照、缩放、文字编辑与导出；独立源图为同名 `.svg`，构建脚本为 `paper/figures/pilot-main-v10-build.py`，参考图保存在 `paper/figures/pilot-main-v10.reference.png`。保留参考图的共享顶部和左右双色分区，按方法初稿校正共享冻结 W、q0/z0 与 q1/z1 接口、CLS 融合、patches 复用、Top-K 和有界评分残差。除观测缩略图外均为矢量元素；图示完整方法，不表示完整双层已联合验证。本轮仅在笔记本验证图稿页面，不涉及训练或远端资源。

2026-09-14，按用户要求完成指定论文草稿 III 方法与 IV 训练方法的 v1 正文，原稿存入 `paper/drafts/archive/`，独立版本为 `paper/drafts/PILOT_方法初稿_v1_20260914.md`。核对持久融合、共享世界模型、q0/q1接口、交错候选比较、原生CLS调整监督、STOP隔离及冻结GRPO。写作边界和逐项检查见 `paper/notes/pilot-method-v1-writing-review-20260914.md`。额外发现在线评分头和native重算的基础分数归一化分别使用ghost集合与完整动作集合，记录为待核对项，未改算法。其他论文章节未改，未生成图片或运行实验。

2026-09-11，用户认可 v4 网络图视觉风格，要求暂停生成，按双层全启用的完整形态梳理模块。核对当前主工作区 `21d8feb` 的感知/记忆、持久融合、共享世界模型、基础策略、旧第二阶段查询与评分、原生CLS适配和动作接口，形成 `paper/notes/pilot-module-boundaries-and-dataflow-20260911.md`。统一七个一级功能模块，细分第二层三个子模块，区分图状态、真实历史和q0预测记录；当前开关与上下文兼容差异仅放附录。未生成图片或运行实验。

2026-09-10，为重新设计可解释的论文主图，逐段核对主工作区 `e4aa771` 的完整 rollout、建图、语言条件图评分、到达查询、拼接融合、后继查询与多轮候选比较，并只读核对持久状态分支 `ef7e046`。详细因果链与图示边界见 `paper/notes/pilot-architecture-reading-for-main-figure-20260910.md`。补充两点：第一层当前只为本步关联的有效 ghost 去重预测，不是每步重算整个图；当前 E24 实际动作路径通过 `stop_isolated_e24_actions` 保留基础 STOP，移动时在全部修正后 ghost 中选择。论文图应区分当前位置、ghost 到达 q0 与仅供评估的 q1，突出表示增强和指令条件候选比较。未运行实验或修改算法。

2026-09-10，为论文双层主图核对指定草稿、`docs/paper-mainline.md`、ghost 拼接融合和第二阶段查询/评分入口，整理绘图目标至 `paper/notes/pilot-two-level-main-figure-brief-20260910.md`。图以“评分前改善候选表示、评分后选择性深入比较”为主线；按用户最新要求先由 ImageGen 生成样式图，认可后再制作可编辑源图。未修改论文正文、训练代码或远端任务。

2026-09-10，补充按用户指定的 `feature/persistent-ghost-state` 核对最新实现：读取独立分支的融合模块、配置、第二阶段和GRPO入口及持久状态验证记录，使用 `git ls-remote` 确认中央分支与本地同为 `ef7e046`。旧记录中的“仅临时增强、不写回图”只适用于旧模式和此前2000步实验；最新可选模式写回融合状态。论文正文未改，未启动训练或测试；GPU验收状态取自仓库记录，未查询远端运行现场。

2026-09-10，任务上下文：为继续论文写作熟悉本地工程。通读指定 Markdown 草稿、`docs/paper-mainline.md`、论文目录说明与学术写作要求，并核对 `ghost_concat_fusion.py`、独立配置及9月8—9日融合实现、结果和新长训练记录。相关工作 A/B/C 已有正文和32条参考文献；方法、训练、实验仍含待办与占位，PILOT/TopoForesight 名称尚未统一。后续写作须区分主线的双阶段方法与最新 ghost_concat 单阶段实验：后者仅临时增强导航图输入，不将预测写入原始图历史，独立配置关闭第二阶段，新结构尚未接入 GRPO，不能直接继承旧融合路径的缓存或训练描述。已完成2000步实验尚未支持稳定导航提升；9月9日晚直接渲染/编译版10000步任务仅有启动验收记录，本轮未连接远端确认后续进度。未修改论文正文或运行实验。

2026-09-09 晚间，重启 `ghost_concat_direct_compiled_10k_20260909`：14200 基座、双卡总批量8、10000步、每200步同步测评，直接渲染/FP16/64批次上限/Inductor；保留选定合并版本，未引入静态条件缓存或导航SDPA。测评机专用容器离线安装G++并修复编译缓存目录后，7项相关测试和8环境8路线预检通过。正式第200步模型及恢复状态已保存，双端SHA一致，测评机已开始1839路线评测，训练继续推进。见 `docs/ghost-concat-direct-compiled-10k-operations-20260909.md`。

2026-09-09，按用户决定排除静态条件缓存与SDPA，从实施前的 `75cbe1d` 合并到主工作区 `feature/e24-joint-sft`，合并提交 `372ba21`，Git tree与选定提交完全一致。保留用户 `.gitignore` 未提交修改，通过Git同步训练机主目录并完成568项CPU回归、11项GPU检查和双卡8次真实更新，全部退出0，两卡冻结/参数更新审计通过。源码未做额外改写，之后仅更新合并状态和测试文档。详见 `docs/perf-merge-validation-20260909.md`；正式长训练未启动。

2026-09-09，实现并验证世界模型编译。复现旧最大差 0.789076，AOT 原生执行零差异，真实 BF16 门控中间值证实 Inductor 改变舍入位置；按用户要求以固定目标质量验收，Inductor 指标变化很小。新增动态形状编译后端、可选原生 CUDA Graph 对照、CLI/覆盖配置和诊断测试。CPU 回归 568 passed，GPU 编译合同 2 passed，11 场景质量与三组 16 步双卡训练通过；推荐 Inductor，预热后 11.296→9.265 秒/更新，首次编译开销单独报告。主要检查 `nwm/compile_runtime.py`、`predictor.py`、`runtime.py`、`raenwm_core/models.py`、`model_utils.py` 及三份编译/质量/训练验证脚本。详见 `docs/nwm-compile-validation-20260909.md`。

2026-09-09，针对 `perf/panorama-training` 的后续提速做源码审查，核对 SFT 更新/保存循环、直接渲染管线、DINO 编码、CDiT 条件计算与 Euler 采样器、导航注意力。建议优先试验采样内静态条件缓存、导航合并注意力，再评估渲染/编码重叠、融合梯度合桶与异步保存；当前短基准不包含保存成本。未执行新基准或修改训练实现，收益均待实测。详见 `docs/panorama-training-next-performance-review-20260909.md`。

2026-09-09，完成直接渲染、大批次、混合精度最终验收。选定 direct/FP16/64/64，通过原生图像零像素差校准，开发集选定后在五场景126目标×三噪声复核；CLS 余弦 0.826675→0.829386，图块余弦 0.704895→0.707941。双卡12步吞吐和冻结权重审计通过，最终567项回归、13项GPU检查通过，CLI和覆盖配置已提供，所有基准已退出。报告 `docs/direct-context-fast-validation-20260909.md`。

2026-09-09，按用户新授权实现按需直接渲染、更大 DINO/世界模型合批和混合精度。新增历史编号白名单与 reset 失效机制，批量按需渲染；随机噪声按固定八行分组，独立于执行批次。六步同规格基准从全景优化版 16.641 降至直接 FP32 13.244，再经世界模型合批和 FP16 降至 10.842 秒/更新。11 场景288目标质量对照中，复核五场景×三噪声的选定 FP16 64/64 方案无下降；推荐覆盖配置 `configs/nwm/direct_context_fast.yaml`，详情见 `docs/direct-context-fast-validation-20260909.md`。

2026-09-09，补跑同规格旧 front 六步基准：9.969 秒/更新，对比优化后全景 16.641 秒。核实旧 runtime 将有效查询整批预测，新路径按 8 拆批；新目标定向视图还失去了原 front 特征的跨候选共享，并显式用 FP32 编码。六步 rank 0 世界模型累计 41.44→50.77 秒、原始 DINO 编码 3.81→10.10 秒，另有目标图像准备 8.01 秒。说明见性能报告补充对照小节；分批成本不能视为朝向修正的数学必然成本。

2026-09-09，在独立性能工作区建立固定真实输入回放和双卡短训练基准。保留精确几何缓存、GPU 视觉缓存、跳过未使用 front 编码、GPU 双精度批量插值四项优化；逐项 6 步基准从 21.807 降至 16.641 秒/更新，固定输入预测保持逐元素相同。最终关闭分段同步的 12 步对照为 24.567→17.871 秒/更新，吞吐提升 37.47%；562 项回归通过，GPU 相关 25 项通过，两卡冻结视觉/路点/世界模型权重哈希不变，CLS 映射与融合层更新。编译与推理批量 16 试验因数值差异被拒绝。基准任务全部退出，原长训练保持停止；详见 `docs/panorama-training-performance-20260909.md`。

2026-09-09，复查新全景 10000 步训练的冻结与性能：实际优化器为 375,128,067 个导航参数加 5,902,080 个融合参数，视觉骨干/路点/世界模型冻结。45 次采样双卡平均利用率 65.0%/48.4%，前 25 步更新耗时约为旧任务 2.14 倍（非同轨迹严格对照）。确认全景额外渲染、CPU 重投影、FP32 编码、特征往返与遗留 front 编码开销，未定量归因各阶段；详见 `docs/ghost-concat-panorama-performance-review-20260909.md`。

2026-09-09，按旧 2000 步任务参数启动 `ghost_concat_panorama_bs8_10k_20260909`：14200 基座、双卡总批量 8、学习率 2e-6/1e-5、新全景上下文、10000 步、每 200 步测评。训练产物通过实验目录软链接写入 `/mnt/data2tb/ETP-R1_data/experiments/`；独立传送程序逐份同步，测评成功且双端校验后释放测评副本，原件与结果保留。操作与启动验证见 `docs/ghost-concat-panorama-10k-operations-20260909.md`。

2026-09-09，复查朝向上下文合并后的 `84e6081`。训练机当前代码完成 559 项回归测试（2 skipped）和双环境 19 个真实查询（10 个倒序端点），退出码均为 0；默认目标对齐已生效。发现旧 ghost 联合实验管理器未固定 front，新增 CLI 参数使旧 manifest 命令比较不一致，影响未完成旧任务重启。详见 `docs/nwm-context-merge-review-20260909.md`。此前“原导航默认链路未切换”仅指独立预测质量实验阶段，当前默认已经切换。

2026-09-09，按用户要求先提交主工作区快照，再将`feature/nwm-panorama-context`的世界模型视角实现、配置、测试与验证工具合并到当前`feature/e24-joint-sft`。保留两侧研究记录，解决的冲突仅为本文件；运行代码与已通过44项相关测试的来源分支一致。合并不自动切换导航默认上下文，独立预测质量产物仍保留原实验位置。

2026-09-09，继续核查联合训练及用户询问耗时：确认训练和全部10次测评正常结束，重核1839任务ID、配对指标、1800/2000实际权重哈希、参数更新、优化器/调度器与AMP缩放状态，整理耗时和结果报告。没有进行额外GPU实验。

2026-09-08，按用户要求先提交当前工作区快照，再实现三层、无中间压缩的批量ghost拼接残差。核对trainer、全景与图输入、检查点、冻结优化器和两机运行入口；通过Git同步，在实际环境完成CPU回归、单/双卡短更新、冻结审计、4路线原图保护检查及融合模块性能基准。仅短程实现验证，无正式性能实验。

2026-09-08，用户要求查看自动实验结果。读取两机训练/评测清单、自动汇总、四份冻结审计、源位姿计数及B/C训练日志；在测评机专用环境重新执行19项只读验收，核对源修复前后逐任务字典完全一致，保存轻量结果快照并编写完整分析。三组训练与12份完整评测均正常完成；未获得超基线的有效RGB模型，未开展额外训练或评测。

2026-09-07，任务上下文：分析三轮 native CLS RGB-only SFT 修复后未稳定超过原始 14200 基线的原因。核对两机配置、完整聚合与逐 episode 结果、TensorBoard 基线、训练诊断和真实 checkpoint 中 CLS 适配层参数，检查融合、全景编码、图历史、低级上下文和 NWM 条件构造，并对照 ETPNav 历史 57→59 实验及 NWM 训练数据合同。在测评机专用环境用 CPU 复现源朝向错位，保存本地诊断与指标快照；未改远端实现、环境、权重或任务。

2026-09-06，任务上下文：参考本地 VLN 论文完成方法 III-A 初稿。重读 ETPNav、ETP-R1、HNR、NavMorph、DGNav 的方法开头，结合当前 `graph_utils.py`、SFT 图输入及候选评分掩码核对基础接口。删除 A 下两个编号小标题，以三个自然段串联任务/输入、在线拓扑候选和动作接口，按需定义符号并明确基础评分所在阶段。复用 [2]、[7]、[9]；缓存及残差细节保留在后文，A 之外正文不变。阅读依据追加在 `paper/notes/local-tase-vln-writing-reference-20260905.md`。

2026-09-06，任务上下文：按用户确认的结构将问题定义并入方法章节。指定论文初稿的原 III、IV 合并为“III. 方法”，包含 A 问题定义与基础导航框架、B 方法总体框架、C 候选到达预测与表示增强、D 预测状态引导的后继查询与候选校正、E 前瞻干预与计算约束；候选预算和展开符号移至总体框架，评分/查询/残差以及回退/触发/成本分别归组。明确基础评分处于第一阶段增强之后、第二阶段校正之前；后续训练、实验、讨论、结论顺延为 IV–VII。参考 `paper/要求/学术写作的基本逻辑与要求.md` 的方法组织要求，保留现有公式、图表与未完成写作提示，本轮仅调整结构及衔接，不进行远端实验。

2026-09-05，任务上下文：广泛检索后完成相关工作 C。查阅 16 项相关研究的原始摘要或全文，重点核对 Look Before You Leap、Active VLN、Dreamwalker、HNR、NWM 与 Metacontrol 的方法。C 改为“前瞻规划与决策”，围绕预测辅助策略、导航分支评估、额外信息获取和本文定位写成四段，新增书目 [29]–[32]；区分真实探索与模型预测，保留第二阶段触发待集成的状态。检索证据及取舍见 `paper/notes/related-work-c-research-20260905.md`。A、B 和其他原有正文及书目保持不变，未操作远端实验。

2026-09-05，任务上下文：按用户确认的世界模型技术路径完成相关工作 B。标题改为“世界模型与导航动态建模”，以四段、1,168 字符连接潜在动态、可视生成、直接表征预测与 RAE-NWM；新增书目 [19]–[28]，复用 HNR、Dreamwalker、NavMorph，重点说明稠密表征上的动作条件流匹配及本文冻结使用方式。新增书目由上一轮保存的 arXiv 原始元数据核对。A、C 和其他正文及已有书目未改，同步记录于世界模型研究笔记。

2026-09-05，任务上下文：广泛检索世界模型流派并分析相关工作 B。联网读取 23 篇原始文献页面，重点核对本地 NWM、RAE-NWM、DINO-WM 方法及当前原生 CLS 配置、速度场损失和采样代码。确认当前接入的是稠密 DINOv2 表征上的动作条件流匹配生成模型，关闭图像解码并不改变其生成式属性；原论文 256 patch 与项目 CLS+patch 输出分开说明。新增技术图谱与 B 写作建议，未改论文正文或运行远端实验。

2026-09-05，任务上下文：用户认为相关工作 A 的显式分类行文过于报告化，要求融合前两版。保留大模型导航与任务专用策略的区别，删除分类维度说明和重复的“决策核心”句式，改为任务及大模型方法、专用策略与航点、空间记忆与在线拓扑、后续改进及本文定位四段连续论述。全部引用和 A 节外论文内容保持不变，同步更新写作逻辑笔记。

2026-09-05，任务上下文：按用户意见重组相关工作 A。将分类依据明确为视觉语言大模型与任务专用跨模态策略两种决策核心，按任务与分类、大模型路线、专用策略及航点、地图与在线拓扑、本文定位组织为五段。保留全部 13 项引用和既有书目，同步更新 `paper/notes/related-work-a-research-20260905.md`；检查确认 A 节外论文内容保持原样。

2026-09-05，任务上下文：广泛联网检索并完成相关工作 A。核对 R2R、RxR、VLN-CE、WPN、候选航点、跨模态地图、DUET、GridMM、BEVBert、ETPNav、ETP-R1、DGNav、NaVid、StreamVLN 共 14 项原始研究来源，补读 4 份论文前 4 页。II-A 改为四段正式正文，按任务演进、动作与观测路线、地图/在线拓扑、后续改进与本文定位组织，追加参考文献 [13]–[18]；引言及其他章节保持原样。来源与书目版本差异记录在 `paper/notes/related-work-a-research-20260905.md`。发现跨模态地图已有未观测语义预测，故不泛称所有地图方法只记录历史。未改变已暂存内容或运行远端任务。

2026-09-05，任务上下文：依据本地 TASE/VLN 参考再次优化指定稿的引言。用同一个走廊指令例子串联观测不足、到达预测和后继查询，明确承认 HNR 已使用预测信息生成更远航点，具体介绍 Dreamwalker、HNR、NavMorph 的作用；突出本稿基础评分前后两阶段的分工和有界修正。引言含图注从 2,535 字符精简至 1,911 字符，图 1 改为概念图图注，证明、符号和绘图细节留在方法或 `paper/notes/local-tase-vln-writing-reference-20260905.md`。保留结果及机制状态括注，核对引用和正文修改范围，引言外论文内容保持原样。

2026-09-05，任务上下文：参考本地 TASE 与 VLN 论文继续写作准备。浏览 TASE 目录 8 篇 PDF，重点读 HEIGHT、ViTeC，并阅读 ETPNav、ETP-R1、HNR、NavMorph、DreamNav、DGNav 的开篇与相关工作，检查代表页版式；新增 `paper/notes/local-tase-vln-writing-reference-20260905.md`，记录实际阅读范围和 14 篇来源。HNR §3.2.4 已使用候选预测深度生成更远航点，§3.3 使用前瞻图评估分支，因此“预测信息产生后继位置”不能直接当作独有创新，后续须具体比较潜在查询、双阶段评分接口和干预约束。本轮未修改论文正文，未重新核查各工作区实现或远端实验。

2026-09-05，任务上下文：按用户确认的结构优化论文引言。修改 `paper/drafts/TopoForesight_TASE_初稿骨架.before-related-work.md` 的引言及其框架图图注：合并重复拓扑背景，明确候选方向可见与到达后视角未知的区别，补充候选对齐、后继查询、误差与成本三个研究问题，重组双阶段前瞻、预测状态引导查询、受约束选择性前瞻三项贡献。沿用引言中的 PILOT 名称；按用户偏好保留完整结果句并括注“待实验验证”，未集成机制另作括注。摘要和相关工作及之后章节保持原样；此前记录的引言贡献重复问题已在本次修改中处理。

2026-09-05，任务上下文：开始逐节论文写作，暂不讨论最终实验结果。通读用户指定的 `paper/drafts/TopoForesight_TASE_初稿骨架.before-related-work.md`、`paper/README.md` 和 `docs/paper-mainline.md`，并查看旧主线候选、调查文档及前瞻模块代码。当前稿的摘要和引言已有正文，相关工作仍为提纲，方法和训练部分混合公式与作者待办；引言贡献前两点存在重叠，主线中的预测状态引导查询未单独突出。补充上述论文入口索引，保留原稿和既有未提交修改；后续可先逐段写引言，实验结论留待结果完成后补充。

2026-09-05，任务上下文：整理 `docs/TopoForesight_TASE_初稿骨架.before-related-work.docx`。核对 Word 原稿为 19 页、636 个段落、10 个表格，无批注、修订痕迹或内嵌图片；新建根目录 `paper/` 作为论文资料工作区，并将原稿按实际标题层级、列表、公式、表格、图注、占位符和 12 条参考文献整理为 `paper/drafts/TopoForesight_TASE_初稿骨架.before-related-work.md`。原始 docx 未修改。

2026-09-05，读取训练机/测评机的运行进程、保存 YAML、训练诊断和历史完整测评 JSON，核验 1,839 条 episode 的结果配对；对照笔记本 ETPNav 的历史 57.8032%→59.1082% 实验与实际 RGB adapter、ETP-R1 的注入、上下文、候选查询和优化器代码。新增 `docs/rgb-injection-navigation-audit-20260905.md`，区分已确认事实与待验证原因。远端操作仅只读检查，测评环境导入退出 0，未重跑训练或完整测评。

2026-09-04 17:01（Asia/Shanghai），对修复后正式训练的
`base_iter14200 → iter200/400/600/800/1000/1200/1400` 做 CLS residual MLP
参数级审计，并用同区间旧错误实验作对照。逐张量核对了三层 weight/bias 的 L2、
最大绝对变化、变化元素数以及 weight 对前一检查点的最佳纯缩放残差；结果确认七个
区间六个张量都在真实更新，新实验到 `iter1400` 的整体漂移约为旧实验 102 倍，
三个 bias 持续变化，weight 更新几乎都不能由 AdamW 纯衰减解释。训练日志同时未
出现 anchor-only、梯度合同、DDP、NCCL 或显存错误。检查使用 CPU mmap 顺序读取
模型 checkpoint，没有占用或中断训练 GPU。

2026-09-04 15:09（Asia/Shanghai），已把修复后低级上下文 RGB-fusion SFT 的
完整评测结果接入现有联合 TensorBoard。测评机容器内新增独立 tmux 会话
`etpr1-rgb-fusion-lowlevel-bs16-gradfix-eval-normalizer`，每 10 秒扫描
`native_cls_eval_lowlevel_bs16_gradfix_ea860aa_20260904` 的有效 JSON，并以 run
`native_cls_rgb_fusion_sft_lowlevel_bs16_gradfix_ea860aa` 写入既有目录
`data/logs/active_lookahead/native_cls_e24_joint_eval/metrics_tensorboard_20260828/`。
首次写入 `iter200/400/600` 共 33 个标量；TensorBoard 数据接口已显示新 run，
SR/SPL 的 step 和数值与原始 JSON 一致。本机原有 `6008` 隧道和测评机 6009
TensorBoard 均保持运行，用户刷新当前页面即可看到，后续结果会自动追加。

2026-09-04 10:45（Asia/Shanghai），测评机已启动修复后新训练的独立升序评测
watcher。启动前确认 ETPNav 未运行、GPU 空闲、工程工作树干净；把失效的 Git
`origin` 校正到训练机专线中央仓库并以 `ff-only` 快进到 `ea860aa`，随后在
`gwl-etpr1-rae`/`etpr1_rae` 中完成 59 项定向测试，结果为
`59 passed, 3 warnings`。新 watcher PID 为 `38957`，监控训练同步目录
`data/logs/raenwm_rgb_fusion/native_cls_sft_lowlevel_bs16_gradfix_ea860aa_20260904/checkpoints/etpr1_native_cls_rgb_fusion_sft_lowlevel_bs16_gradfix_ea860aa/`，
结果写入
`data/logs/raenwm_rgb_fusion/native_cls_eval_lowlevel_bs16_gradfix_ea860aa_20260904/`；
固定单卡、8 环境、完整 1,839 个 R2R `val_unseen` episode，并按 checkpoint
升序执行。启动时训练约到第 60 次更新，尚无 `iter200`，watcher 正常记录
`reason=no_checkpoints`，没有占用 GPU。修复前旧 watcher 和 `iter7200` 评测
进程已停止，旧结果与 checkpoint 未删除。

2026-09-04 10:28（Asia/Shanghai），按用户要求在训练机从干净导航基座重新启动
低级上下文原生 CLS RGB-fusion SFT。源码固定为梯度修复提交 `ea860aa`，使用
`start`/weights-only 路径加载 `base_iter14200.pth`，基座 SHA-256 仍为
`1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61`，没有恢复
旧低级实验的模型或优化器状态。除实验名、输出/同步目录和基座路径由相对路径改为
同一文件的绝对路径外，保存配置与
`native_cls_rgb_fusion_sft_lowlevel_bs16` 的有效训练参数一致：双卡、每 rank
8 环境、每 rank batch 8、梯度累积 1、全局 batch 16、10,000 次更新、每 200 次
保存、学习率 `1e-5`、无 warmup、DAgger 偏移 14,200、低级上下文和 RGB fusion
参数不变。新实验名为
`etpr1_native_cls_rgb_fusion_sft_lowlevel_bs16_gradfix_ea860aa`，输出根为训练机
`data/logs/raenwm_rgb_fusion/native_cls_sft_lowlevel_bs16_gradfix_ea860aa_20260904/`，
supervisor PID 为 `42131`，启动日志为
`supervisor_server/start_20260904T102830.log`。启动核验时两个 rank、torchrun 和
supervisor 均存活，两张 A6000 满载，NWM 两侧均 `314/314` 匹配，至少前两次真实
优化更新已跨过 residual MLP 非零梯度硬检查，无 traceback、DDP、NCCL 或显存
错误。checkpoint 同步使用测评机同名隔离目录；修复前旧实验评测随后已按用户
要求停止，已有文件未覆盖。

2026-09-04，完成低级上下文导航 CLS residual MLP 梯度修复并提交 `ea860aa`：新增 raw-only 编码接口、低级同步器强制接口和 anchor-only 更新硬检查；训练机完成 59 项针对性测试、546 项其余完整测试、真实单卡低级两步和真实双卡 DDP 单步回归。测评机当时正在执行完整评测，未抢占 GPU；另发现其 Git origin 仍为旧不可达入口，待下一次源码同步前按项目固定专线地址校正。

2026-09-03，断电重启后恢复当前低级上下文 RGB-fusion SFT 的联合测评
TensorBoard。测评机专用容器 `gwl-etpr1-rae` 内重新启动 TensorBoard 2.21.0
和 `native_cls_rgb_fusion_sft_lowlevel_bs16` 结果归一化器；归一化器继续扫描
`native_cls_eval_lowlevel_bs16_20260902` 的完整 JSON 结果，并写入既有联合目录
`data/logs/active_lookahead/native_cls_e24_joint_eval/metrics_tensorboard_20260828/`。
本机用户服务 `etpr1-native-cls-e24-eval-tb-tunnel.service` 已改用训练机跳板和
测评机固定专线地址，并在每次启动时动态查询 ETP-R1 容器地址，不再依赖旧
`eval` 入口或写死 Docker 地址。本机访问地址为 `http://127.0.0.1:6008/`；
HTTP、TensorBoard 数据接口和浏览器页面均已验证，当前 run
`native_cls_rgb_fusion_sft_lowlevel_bs16` 显示到第 2,800 次，后续完整结果会
每 10 秒自动追加。

2026-09-02 16:02（训练机时间），只读核验正在运行的 R2R 原生 CLS
RGB-fusion 低级上下文 SFT。训练机提交为 `e46832d`，supervisor、torchrun 和
两个 rank 均存活；实时进度为 `1606/10000`，最新完整模型/训练状态对为
`iter1600`。两张 A6000 均为 100% 利用率，显存约 23.2/24.1 GiB，温度
85/86°C；日志未发现 traceback、CUDA OOM、NCCL 或 RuntimeError。换平台期间
`iter200`--`iter800` 曾因专线不可达和 SSH 主机指纹变化同步失败；网络恢复后
`iter1000`、`1200`、`1400`、`1600` 均成功发布到测评机。随后按用户要求用
项目原子同步工具补传 `iter200`、`400`、`600`、`800`，已有文件按大小跳过；
两端现有 `iter200`--`iter1600` 共 8 个模型 checkpoint 的 SHA-256 逐一一致，
测评机 `.incoming` 无残留。随后在测评机重新启动本实验的升序持续评测 watcher，
PID 为 `14673`，输出根为
`data/logs/raenwm_rgb_fusion/native_cls_eval_lowlevel_bs16_20260902/`；它固定使用
单卡、8 环境和完整 1,839 个 R2R `val_unseen` episode，已领取 `iter200` 并
推进到 44/1,839，3090 显存约 12.8 GiB、利用率 100%，无加载或 CUDA 错误。
训练进程自身继续为后续每 200 步的新 checkpoint 启动原子同步；watcher 会按
升序跳过已有结果并持续领取新文件。更换平台后测评机系统时钟一度比训练机慢约
2 小时 53 分，已在
本地授权窗口中一次性校准到约 5 秒内；因测评机没有可用公网时间源，
`timedatectl` 仍显示未持续 NTP 同步，后续应继续以文件大小和 SHA-256 判断同步
完整性，而不是只比较两机本地时间戳。

2026-09-02，测评机更换平台并改用 RTX 3090 24GB 后完成推理回归。恢复专用
`gwl-etpr1-rae` 容器，在 `etpr1_rae` 环境确认 Python 3.11.15、PyTorch
2.2.2+cu121、Transformers 4.49.0、CUDA 可用、Habitat/Habitat-Sim 0.3.3。
低级上下文 E24 joint `iter200` 单 episode 推理退出 0，NWM `314/314` 匹配，
SR/SPL 为 `1/1`。随后使用旧平台已有完整结果的 RGB-fusion `iter10000`
checkpoint，以原高层上下文语义、单卡、8 环境完整复跑 1,839 个 R2R
`val_unseen` episode，耗时 53 分 29 秒且无 traceback/CUDA/OOM 错误。新旧
SR 为 `0.6329526917/0.6318651441`，SPL 为
`0.5417617617/0.5415429860`，分别偏移 `+0.1088/+0.0219` 个百分点；nDTW、
SDTW 分别偏移 `+0.0754/+0.0898` 个百分点，未见严重平台偏移。新结果和日志
位于测评机 `data/logs/platform3090_validation/`。旧 4090 同一检查点评测耗时
29 分 27 秒；新 3090 本次耗时 53 分 29 秒，表面上慢约 81.6%。但旧结果运行
时测评机仍停在提交 `0e82d27`，新结果使用提交 `e46832d` 并显式把后来新增的
上下文参数恢复为旧值 `high_level_nav_latent`。因此 checkpoint 和评测语义相同，
但可执行代码并非同一提交；这组耗时不能作为纯 4090/3090 硬件对照。若要严格
归因，需要在 3090 上使用 `0e82d27` 的代码原样复跑。

2026-09-02，测评机更换平台但保留原硬盘后，重新核验三机网络。新平台只检测到
RTL8125 2.5GbE 网卡 `enp6s0`，固定为 `10.10.10.2/24`，没有独立默认路由、
互联网或可用的 Tailscale 入口。训练机 `enp5s0` 保持 `10.10.10.1/24`；双端
协商为 2.5 Gbps 全双工，专线无丢包，笔记本通过 `server` 跳板登录测评机成功。
笔记本到测评机的标准入口改为 `ssh -J server a6000@10.10.10.2`。测评机受限
部署密钥已只读验证可通过 `10.10.10.1` 读取中央裸仓库；全局和项目
`AGENTS.md` 已同步移除旧 `ssh eval`/Tailscale 入口并更新 Docker、GPU、管理员
授权和 Git 同步示例。

2026-09-02，按用户要求启动新的 R2R 原生 CLS RGB-only SFT 与升序完整测评，
使用最新低级世界模型上下文代码，但不启用前瞻。训练源提交为 `e46832d`，配置
`run_r2r/iter_train_rae_dino_native_cls_rgb_fusion.yaml` 明确使用上下文合同
`r1_low_level_move_rgb_anchor_v1`、`rgb_fusion_enabled=True`、
`rgb_fusion_trainable=True` 和 `ACTIVE_LOOKAHEAD.enabled=False`。任务从
`base_iter14200.pth` weights-only 启动，双卡、每 rank 8 个 Habitat 环境、
梯度累积 1，全局 batch 16，共 10,000 次更新，每 200 次保存并通过 2.5 GbE
直连原子同步到测评机。实验名为
`etpr1_native_cls_rgb_fusion_sft_lowlevel_bs16`，训练输出根为
`data/logs/raenwm_rgb_fusion/native_cls_sft_lowlevel_bs16_20260902/`，训练机
supervisor PID `3364601`。

启动前训练机和测评机的 RGB-only/低级上下文针对性测试分别为
`28 passed, 2 warnings` 和 `28 passed, 3 warnings`。启动后两个 rank 的 NWM
均 `314/314` 匹配，已稳定跨过第 3 次更新，无 traceback、DDP 或显存错误；
两张 A6000 均为 100% 利用率，显存约 21.1/19.6 GiB。测评机 watcher PID
`1123911`，输出根为
`data/logs/raenwm_rgb_fusion/native_cls_eval_lowlevel_bs16_20260902/`，实验名为
`etpr1_native_cls_rgb_fusion_eval_lowlevel_bs16_watch`，固定单卡 8 环境、完整
1,839 个 `val_unseen` episode、按 checkpoint 升序执行。交接时尚未到第 200
次保存点，watcher 正常处于 `reason=no_checkpoints`；受保护的 ETPNav 容器未
运行，4090 空闲。旧 RGB-only 完成态 watcher 仅保留 30 秒空轮询且无评测子进程。

2026-09-02，按用户要求停止低级移动上下文 batch-16 原生 CLS E24 joint SFT，
并同时停止测评机对应的升序 watcher 和当时正在执行的 `iter1600` 全量评测。
训练机 supervisor、torchrun 和两个训练 rank 均已退出，两张 A6000 显存降至
45/16 MiB、利用率为 0%；最后完整模型/训练状态对均为 `iter1600`。测评机
watcher 退出后，Docker 内评测因位于独立进程组而未随 watcher 的 TERM 信号
退出，随后按完整实验名精确终止该评测进程组；4090 降至 142 MiB、利用率为
0%。两个 TensorBoard 和低级上下文指标归一化进程按用户范围保留运行。

2026-09-01，将本轮低级上下文 batch-16 joint SFT 的增量评测接入笔记本
`http://127.0.0.1:6008/`。该端口继续由用户级 SSH 隧道映射到测评容器 6009，
TensorBoard 读取联合目录
`data/logs/active_lookahead/native_cls_e24_joint_eval/metrics_tensorboard_20260828/`。
容器内新增 tmux 会话 `etpr1-lowlevel-bs16-eval-tb-normalizer`，每 10 秒扫描
`native_cls_e24_joint_eval_lowlevel_bs16_20260901` 的完整结果，并以独立 run
`native_cls_e24_lowlevel_bs16` 写入相同的 `eval_*` 标签，后续 checkpoint 会自动
追加。首个 `iter200` 已完成全部 1,839 个 `val_unseen` episode，11 项指标已由
TensorBoard 数据接口和浏览器页面共同确认；其中 SR 为 `0.6275149584`，SPL
为 `0.5338301659`。页面刷新后新 run 已出现在 run 列表并默认勾选，旧 joint、
RGB-fusion 和起点曲线均保留，可直接叠图比较。

2026-09-01，使用当前低级移动上下文、延迟渲染和 Q0 缓存逻辑启动正式
R2R 原生 CLS E24 joint SFT。目标为 10,000 次优化器更新，双卡、每 rank 8 个
Habitat 环境、梯度累积 1，因此全局有效 batch 为 16；每 200 次保存并经训练机
到测评机的 2.5 GbE 直连原子同步。首次同配置冒烟在第 2 次更新前发现两个 rank
均有 6 个 `rgb_encoder.cls_residual_mlp` 参数未进入首轮损失，显存并未不足。
提交 `e46832d` 为这组动态参数增加数值为零的 DDP 梯度锚点，保持损失和真实
梯度不变；训练机针对性测试为 `26 passed, 2 warnings`。修复后的 8×2、无累积
两次更新冒烟退出 0，NWM `314/314`，Q0 缓存与 Q1 成功率均为 1，设备显存
占用约 21 GiB，日志位于
`data/logs/active_lookahead/native_cls_e24_joint_smoke/20260901_bs16_ddp_anchor/`。

正式任务源提交为 `e46832d`，从 `base_iter14200.pth` weights-only 启动，不热启
旧高层上下文 joint 权重；实验名为
`etpr1_native_cls_e24_joint_sft_lowlevel_bs16`，输出目录为训练机
`data/logs/active_lookahead/native_cls_e24_joint_sft_lowlevel_bs16_20260901/`，
启动 supervisor PID 为 `2835202`。交接检查时已跨过第 2 次更新，两张 A6000
持续计算、没有 traceback 或 DDP 错误。测评机同时启动独立升序 watcher，PID
`256545`，等待 50 个 checkpoint 并逐个完成全部 1,839 个 R2R `val_unseen`
episode；输出根为
`data/logs/active_lookahead/native_cls_e24_joint_eval_lowlevel_bs16_20260901/`。
启动时受保护的 `gwl-etpnav` 容器未运行，4090 仅有约 142 MiB 桌面占用；尚无
第 200 次 checkpoint，因此 watcher 正常处于 `reason=no_checkpoints`。

2026-09-01，任务上下文：在不改变 `r1_low_level_move_rgb_anchor_v1`、checkpoint
格式和 trainer 事件接口的前提下，把低级上下文从“每个成功前进小步立即渲染”
优化为延迟渲染。Habitat worker 现在只记录成功移动后的完整位姿，每次 drain
最多保留最后 4 个；碰撞静止帧在渲染前丢弃。高层动作结束后，最后一帧复用
完整导航观测中的正前方 RGB，其余最多 3 帧才用单 RGB 传感器回放。teleport
清空旧队列并记录落点位姿；视频模式复用已经生成的逐步 RGB。回放会恢复最终
agent 位姿、`_prev_sim_obs` 和碰撞状态，不改变下一步导航与测量。实现提交为
`f2c1d58`，测试替身修正为 `273ee4b`。

训练机正式环境版本仍为 Python 3.10.14、PyTorch 2.2.2+cu121、Transformers
4.49.0、CUDA 12.1、Habitat/Habitat-Sim 0.3.3。延迟渲染相关定向测试为
`70 passed, 2 warnings`；排除明确只适用于测评机专用容器的三个运行时测试文件
后，可收集完整测试为 `495 passed, 2 warnings`。未排除时为
`542 passed, 3 failed, 2 warnings`，三个失败均是训练机不具备测评机固定的
`/home/a6000/...` runtime 与包装器，不涉及本次改动。

R2R/RxR 两次 SFT 更新和一次冻结 R2R GRPO 更新均退出 0。R2R 每个高层边界
的有效位姿均值保持旧基线 `4.529`，实际 frame/encoded frame 降为 `3.235`，
其中单传感器回放 `2.235`、复用已有 RGB `1.000`、裁剪 `1.294`；上下文就绪
比例仍为 `0.706`。RxR 有效位姿均值保持 `6.083`，实际 frame 降为 `3.458`，
其中回放 `2.458`、复用 `1.000`、裁剪 `2.625`；就绪比例仍为 `0.792`，碰撞
静止去重仍为 `0.188`。两者 NWM 均 `314/314` 匹配，Q0/Q1 成功率均为 1，
Oracle q1 为 0。冻结 GRPO 的 29 次 drain 共记录 177 个有效位姿，只输出 102
帧：回放 73、复用 29、裁剪 75；future-valid 为 104，无批次/行失败。保存后的
E24 135 个张量和 RGB fusion 10 个张量与源 SFT checkpoint 逐项完全一致。
产物位于训练机
`data/logs/active_lookahead/native_cls_e24_joint_single_smoke/20260901_delayed_r2r/`、
`data/logs/active_lookahead/rxr_native_cls_e24_joint_smoke/20260901_delayed_rxr/`
和 `data/logs/active_lookahead/r2r_delayed_frozen_grpo_smoke/20260901/`。

R2R/RxR 单 episode 评测均退出 0，NWM `314/314`、Oracle q1 为 0且无失败。
R2R 的 SR/SPL 为 `1/1`、NDTW/SDTW 均为 `0.900898`；旧即时链路记录并编码
66 帧，新链路记录同样 66 个有效位姿，但只编码 42 帧，其中回放 30、末帧复用
12、裁剪 24。DINO 总编码时间从 `0.331270` 秒降为 `0.258732` 秒，回放渲染
耗时 `0.020720` 秒；单 episode 墙钟为旧 `17.9803` 秒、新 `18.0569` 秒。
RxR 新链路记录 15 个有效位姿并编码 15 帧，其中回放 9、复用 6；新旧全部导航
统计逐项相同，墙钟为旧 `9.3739` 秒、新 `9.4104` 秒。单样本墙钟只记录、不设
硬性能门槛；确定性验收以渲染次数下降、导航结果相同和单元测试中的最终四帧
RGB/latent 完全一致为准。结果位于训练机
`data/logs/active_lookahead/native_cls_e24_single_episode_eval/20260901_delayed_r2r/`
和 `data/logs/active_lookahead/rxr_delayed_single_episode_eval/20260901/`。

同日完成测评机条件步骤。旧 RGB-fusion 全量评测自然完成全部 50 个 checkpoint；
最后 `iter10000` 的 1,839 个 episode 完整退出并写入结果。其常驻 watcher 仍按
设计每 30 秒空轮询，但没有 `run.py` 子进程，GPU 已释放。测评机工作树干净后
以 `ff-only` 更新到 `5520dd1`，专用 `gwl-etpr1-rae` 容器和 `etpr1_rae`
环境版本为 Python 3.11.15、PyTorch 2.2.2+cu121、Transformers 4.49.0、
CUDA 12.1、Habitat/Habitat-Sim 0.3.3；定向测试为 `70 passed, 3 warnings`。
训练机到测评机只通过 2.5 GbE 直连传输两份单回合所需 checkpoint，大小和
SHA-256 均逐项一致，没有传输训练状态或日志。

测评机 R2R/RxR 单 episode 均退出 0，NWM `314/314`、Oracle q1 为 0且无
批次/行失败。R2R 导航和上下文计数与训练机一致：SR/SPL `1/1`、NDTW/SDTW
`0.900898`，66 个有效位姿输出 42 帧，其中回放 30、复用 12、裁剪 24；评测
耗时 `12.1103` 秒。RxR 的全部导航统计也与训练机逐项一致，15 个有效位姿输出
15 帧，其中回放 9、复用 6；评测耗时 `7.9883` 秒。结果位于测评机
`data/logs/active_lookahead/native_cls_e24_single_episode_eval/20260901_delayed_r2r_eval4090/`
和 `data/logs/active_lookahead/rxr_delayed_single_episode_eval/20260901_eval4090_fix2/`。

2026-09-01，任务上下文：把原生 CLS 主链路的 RAE-NWM 上下文从“每个高层决策
写入当前全景正前方特征”改为低级移动观测。新合同为
`r1_low_level_move_rgb_anchor_v1`：episode reset 与 teleport 清空旧轨迹并记录
落点锚帧，之后只记录 `MOVE_FORWARD` 后的正前方 RGB，转向忽略、碰撞静止帧
去重。Habitat worker 不运行 DINO；SFT、评测和冻结 GRPO 在 `envs.step()` 后、
暂停环境前共用跨环境批量编码器，最多 64 帧一批，按原事件顺序写入原生
`[CLS+256 patch]` buffer，不足四帧不填充。R2R/RxR joint checkpoint 格式分别
升为 `etpr1-native-cls-e24-joint-q0-cache-v4` 与
`etpr1-rxr-native-cls-e24-joint-q0-cache-v3`，并在 source snapshot、RGB-only
metadata 和 joint provenance 中绑定上下文合同；低级模式拒绝旧高层断点。
全局默认仍是 `high_level_nav_latent`，用于历史复现；native 主配置显式切到低级
模式。启动器默认从导航基座、离线 E24、恒等 Top-5 adapter 和新 RGB fusion
开始；旧权重只允许在同时提供 checkpoint、SHA-256 和源上下文合同后做
weights-only 迁移，不恢复训练状态。实现主要涉及
`vlnce_baselines/nwm/low_level_context.py`、`common/environments.py`、
`nwm/runtime.py`、两个 trainer 与 `nwm/frozen_grpo.py`；本地静态配置/脚本测试
为 30 项通过。训练机正式环境为 Python 3.10.14、PyTorch 2.2.2+cu121、
Transformers 4.49.0、CUDA 12.1、Habitat/Habitat-Sim 0.3.3；初版定向测试
`73 passed`，可收集完整测试 `481 passed, 2 warnings`。RAE/ETPNav 同一输入的
数值一致性为 `max_abs=0`、cosine `0.9999999404`。

低级合同的真实短测均已完成。R2R 与 RxR 各做两次 SFT 更新，均退出 0；R2R
上下文就绪比例约 `0.706`、每个高层动作约 4.529 帧，RxR 分别约 `0.792` 和
6.083 帧，并实际观察到碰撞静止帧去重。两者 Q0/Q1 请求成功率均为 1，Oracle
q1 均为 0。冻结 R2R GRPO 一次更新退出 0，future-valid 为 104、上下文就绪
比例约 `0.793`；保存前后 E24/Top-5 的 135 个张量及 RGB fusion 的 10 个张量
与源 SFT checkpoint 逐张量完全相同。结束时 VectorEnv 打印过
`BrokenPipeError` 清理提示，但训练、checkpoint 保存和主进程退出码均正常。
对应产物在训练机
`data/logs/active_lookahead/native_cls_e24_joint_single_smoke/20260901_lowlevel_r2r/`、
`data/logs/active_lookahead/rxr_native_cls_e24_joint_smoke/20260901_lowlevel_rxr/`
和
`data/logs/active_lookahead/r2r_lowlevel_frozen_grpo_smoke/20260901/`。

单 episode 在线评测也已覆盖 R2R 与 RxR。RxR 首次评测发现活动候选未保存真实
位置，提交 `6792ac7` 修复后退出 0，NWM `314/314` 匹配且无批次/行失败，结果
在训练机
`data/logs/active_lookahead/rxr_lowlevel_single_episode_eval/20260901_fix1/`。
R2R 使用同一低级合同于提交 `41e9085` 退出 0，SR/SPL 为 `1/1`，NDTW/SDTW
均为 `0.900898`，NWM `314/314` 匹配、Oracle q1 为 0、无批次/行失败；新增的
`lookahead_ckpt_*.json` 低级上下文段记录 12 次 drain、12 个 reset、66 个
frame/encoded frame、就绪比例 `0.833333`，编码总耗时 `0.331270` 秒、约
`0.005019` 秒/帧。结果在训练机
`data/logs/active_lookahead/native_cls_e24_single_episode_eval/20260901_lowlevel_r2r_diag/`。
诊断追加提交的相邻定向测试为 `59 passed, 2 warnings`。测评机仍在执行旧
RGB-fusion 全量评测；2026-09-01 12:32 检查时已完成 42/50 个 checkpoint，
正在启动第 8,600 次迭代评测。故尚未更新到低级合同提交，也没有在该机并行
补跑。

2026-08-31，任务上下文：只依据代码与方法逻辑修订论文主线，不使用尚未完成的训练结果判断方法是否成立。统一以大写 \(K\) 表示 Top-\(K\) 候选预算，以小写 \(k\geq1\) 表示从 \(q_0\) 开始的前瞻展开步数；方法公式和统一描述采用一般 \(q_0\rightarrow\cdots\rightarrow q_k\) 形式，默认 \(k=1\)。一次 \(q_0\rightarrow q_1\) 已构成完整双阶段主方法，但正式投稿必须完成 \(k=1,2,3\) 的视野长度消融。论文同时把“世界模型”收紧为冻结的动作条件表征世界模型，把 STOP 表述限定为残差不直接修改停止分数，并按实际代码将决策不变性写为查询候选上界为 \(\Delta\)、未查询候选上界为 0 的候选级证明。该证明只用于计划中的第二阶段深层查询链跳过，不能推出第一阶段 \(q_0\) 可跳过或性能不会下降。决策不变性触发、\(k>1\) 实现和 50/10/4步正式对照仍未接入完整论文方法链路，必须保持计划时态。

2026-08-30，任务上下文：只读诊断刚完成的 R2R 原生 CLS E24 联合 SFT。训练与评测均使用提交 `116510e`；训练从 `iter14200` 基座运行 10,000 次、退出码 0，50 个 checkpoint 与 50 份完整 1,839 episode `val_unseen` 结果全部有效。按 `SR+SPL` 选择的最佳点是联合第 7,800 次，SR/SPL 为 `0.6318651441/0.5488278347`，仍低于起点的 `0.6373028820/0.5560538836`；50 个点没有任何一个超过起点的 SR 或 SPL。最佳点与起点逐 episode 对比为成功新增 136、丢失 146，差异没有显示稳定收益。

直接证据表明 E24 在线退化为近似 no-op：离线 oracle-future 验证曾有 `12.95%` 动作翻转率和 `+2.14` 个百分点决策准确率，但本轮完整在线评测除第 200 次为 `1.78%` 外，之后通常只有约 `0.1%--0.4%`；训练中的残差绝对均值从第 200 次 `0.592` 降到第 600 次 `0.045`，Top-5 CLS adapter 梯度范数从 `0.239` 降到最终 `0.004`。主要结构性原因是：离线 E24 使用 oracle future 且有效 future 覆盖率 `87.2%`，在线串联预测只有约 `47%`；DINO-CWP 自身 `val_unseen` 的 waypoint `top1_cwp_success` 只有 `27.06%`，且本轮使用的旧 Q0 合同与 R1 实际 `ghost_mean_pos` 不一致；adjusted CE 训练完整 logits，但部署的 `stop_isolated_e24_actions()` 固定基础 STOP/MOVE 边界，只允许 ghost 间重排；基础策略又同时用原始 logits 的 CE 继续训练，容易吸收同一教师信号并让残差收缩。次要因素是起点本身是 75 个验证点中的孤立峰值，联合阶段把全局 batch 从 16 降到 8、仍以恒定 `1e-5` 继续训练。现有结果足以判定本轮没有端到端增益；Q0 修复、预测 future 质量、STOP 口径和冻结基座/残差的单变量贡献仍需消融确认。本次未修改训练代码、远端任务或实验产物。

2026-08-30，任务上下文：把原生 CLS 前瞻链路统一到 R1 的 `ghost_mean_pos` 语义，并复用第一阶段 Q0 预测。`GraphMap` 现在先算候选并入后的最终均值，SFT 和冻结 GRPO 共用 `CandidateQ0` 缓存契约 `r1_post_update_ghost_mean_cached_v1`；每个存活 ghost 只保留最新 CPU FP16 patch 与共享四帧源上下文，再次观测会先使旧缓存失效。Top-5 缓存缺失时只让该槽无效，不再回退重算 Q0。新 checkpoint 格式为 R2R `etpr1-native-cls-e24-joint-q0-cache-v3`、RxR `etpr1-rxr-native-cls-e24-joint-q0-cache-v2`；旧 joint checkpoint 只按强制 SHA-256 热启动 RGB 融合层、E24 和 Top-5 CLS 适配层，不继承导航策略或任何训练状态。

训练机正式环境针对性测试 `80 passed`，除测评机专属运行时布局检查外的完整测试 `496 passed`。R2R 单卡 2 次、双卡 2 次、双卡从第 2 次恢复到第 4 次均退出 0；RxR 单卡 2 次退出 0；冻结 R2R GRPO 双卡 1 次更新退出 0，保存后的 E24/Top-5 共 135 个张量和 RGB 融合层 10 个张量与源 SFT checkpoint 逐张量完全相同。双卡 SFT 第 2 次诊断中共享 Q0/Q1 时间分别为 `9.022/9.125` 秒，Top-5 `q0_requested=0`、`q0_nwm_seconds=0`、缓存成功率 1、Oracle 0；冻结 GRPO 同样为 Top-5 Q0 请求 0、缓存成功率 1、Oracle 0。训练 smoke 产物位于训练机 `data/logs/active_lookahead/*q0_cache*`。测评机旧 joint SFT 的 50 个 checkpoint 已于 2026-08-30 13:55 全部评测完成，整体最佳为第 7,800 次但仍低于原基座；新 Q0 链路的正式热启动选择、测评机单 episode 和旧/新固定 episode 性能对照仍待完成。由于新链路只迁移 RGB 融合层、E24 和 Top-5 adapter，不迁移导航策略，不能仅按旧完整模型的 `SR+SPL` 直接认定第 7,800 次也是最佳热启动权重。

2026-08-30，任务上下文：只读审计“关闭前瞻、仅启用原生 CLS RGB 注入，并从 R2R SFT `iter14200` 权重开新阶段”的代码合同。核心前向、保存和双卡手工梯度同步可以独立工作；基础 checkpoint SHA256 为 `1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61`，只含导航权重/配置/iteration/RGB 元数据，不含融合层或训练状态，因此首次启动必须是 weights-only（`IL.is_requeue=False`），不能伪装成断点恢复。当前存在两个明确缺口：仓库没有 RGB-only 配置、训练托管入口或工作流测试，原生 CLS E24 启动器仍强制要求 E24 warm-start 并校验 E24/DINO-CWP 资产；同时 training-state 只有 `ACTIVE_LOOKAHEAD.enabled=True` 的原生联合模式才保存和恢复每 rank NWM generator，RGB-only 即使开启 `strict_rng_resume` 也会在恢复后重置 NWM 采样序列。另有三项高风险实验差异：RGB-only checkpoint 没有绑定基础 checkpoint/NWM 资产/融合超参/Q0 合同的严格 provenance；当前共享 Q0 builder 已把 RGB 查询从 ETPNav SR59 的“本步候选均值”改为“并入后的 ghost 历史均值”；融合代码丢弃 gate、余弦一致度和注入改变量，只留下候选计数，长跑无法判断模块是否有效工作。ETPNav 的 `iter17400`（SHA256 `be759e498027910ba8dfbc7005cd40ee3ce2f16a01c6039aeb0f7129a7a5c056`，SR/SPL `0.591082/0.496767`）实际同时训练导航策略、RGB 融合层及旧 patch-only NWM 的 token/confidence heads，不等价于当前冻结 75k 原生 CLS NWM。训练机针对性测试为 `31 passed, 2 warnings`；本次未修改训练代码、远端任务或实验产物。

上述审计的第 1、5 点已由提交 `c9c7cd3`、`cc11e01` 修复：新增独立的原生 CLS RGB-only 配置、作业脚本、托管入口和工作流测试，不再校验或加载 E24/DINO-CWP；训练日志按每次优化更新跨轨迹步、梯度累积和 rank 汇总查询成功率、候选覆盖率、gate/余弦/注入范数分布、NWM 耗时及去除 AMP scale 后的融合层梯度范数。本机配置/脚本测试 `12 passed`；训练机 `etpnav_unified` 针对性测试 `38 passed, 2 warnings`。训练机单卡单环境 2 次 smoke 退出 0，覆盖率 `0.521`、注入范数约 `0.001`、融合层梯度范数 `0.008`；双卡每 rank 4 环境 2 次 smoke 退出 0，覆盖率 `0.570`、注入范数约 `0.001`、梯度范数 `0.034`，checkpoint 含 10 个已更新的融合张量且 training-state 含两份 rank 状态。正式任务于 2026-08-30 16:57 启动，训练机 supervisor PID `1077372`，日志 `data/logs/raenwm_rgb_fusion/native_cls_sft/supervisor_server/start_20260830T165653.log`；源提交 `cc11e01`，从 SHA256 `1694b175...c61` 的 `base_iter14200.pth` weights-only 启动，10,000 次、每 200 次保存、双卡×每 rank 4 环境、全局 batch 8、LR `1e-5`、DAgger 时间偏移 14,200，前瞻关闭，RGB 注入开启。确认两个 rank 存活、两卡 100% 利用率并进入第 1 次更新；正式训练仍在进行，尚未启动测评。审计中未获授权的 generator 恢复、provenance 和 Q0 语义差异仍保持原状。

同日 20:13，提交 `0e82d27` 新增 RGB-only 专用 `val_unseen` watcher，固定匹配训练配置、同步目录、465k 预训练资产、升序 checkpoint 和独立结果根，并复用共享 GPU 锁及受保护 ETPNav 检查。测评机专用容器针对性测试 `23 passed, 3 warnings`；第 200 次 checkpoint 的单 episode 冒烟退出 0，NWM `314/314` 严格匹配，SR/SPL/ NDTW/SDTW 为 `1/1/0.900898/0.900898`，结果位于 `data/logs/raenwm_rgb_fusion/native_cls_eval_smoke/iter200_episode1/`。正式 watcher PID `3849794`，日志 `data/logs/raenwm_rgb_fusion/native_cls_eval/watch.log`；启动时已有 6 个 checkpoint，已领取第 200 次并开始完整 1,839 episode 评测，4090 约占 13.0 GiB、利用率 63%。受保护的 `gwl-etpnav` 容器未运行；正式结果目录为 `data/logs/raenwm_rgb_fusion/native_cls_eval/results/etpr1_native_cls_rgb_fusion_eval_watch/eval_results/`。

第 200 次完整结果于 20:44 落盘，SR/SPL 为 `0.6242523/0.5224532`。增量转换器 PID `3852518` 每 10 秒把 11 项正式指标写为联合面板中的 `native_cls_rgb_fusion_sft` run。笔记本当前打开的 `127.0.0.1:6008` 实际映射到测评容器 6009，读取目录 `data/logs/active_lookahead/native_cls_e24_joint_eval/metrics_tensorboard_20260828`；本机 6007 才映射容器 6008 的旧联合目录。已通过当前 6008 数据接口和页面 DOM 确认新 run、step 200 的 SR/SPL 与全部 11 个标签可见。

2026-08-26，任务上下文：将最新原生 CLS RAE-NWM、Top-5 CLS adapter 与 E24 v2 前瞻链路推广到 RxR。新增普通 RxR RAE/DINO SFT 与 joint SFT 两阶段配置：普通阶段为双卡、每 rank 6 环境、累积 2、有效 batch 24、30,000 次更新；joint 阶段为双卡、每 rank 2 环境、累积 2、有效 batch 8、10,000 次更新，并从经 SHA/iteration/任务身份校验的普通 RxR 基座清单启动。普通 SFT training-state 新增可选格式 5，严格保存每 rank Python/NumPy/PyTorch/CUDA RNG 和 episode 队列；RxR joint checkpoint 使用 `etpr1-rxr-native-cls-e24-joint-v1`，保存四组优化器、每 rank RNG/NWM generator，并记录 guide、四语言、HFOV63、基座清单和 DAgger 绝对时间轴。选优固定为 SDTW→NDTW→SPL→SR→iteration，完整评测以数据集与 NDTW GT 交集生成 episode 清单；训练机和测评机均核验为 11,006 个 episode、ID SHA256 `2c36ff390aed7d066107ea2964ec1f104b701ebdd606bcf02c4b7f15a73ff0b4`。同步增加 checkpoint SHA 就绪标记，两个测评 watcher 共用 GPU 文件锁且不碰受保护 ETPNav 任务。

训练机实际环境完成 133 项针对性测试；普通 RxR SFT 单卡 2 次更新后以新进程严格恢复到第 3 次，单 episode 评测退出 0；双卡短测退出 0且状态含两份 RNG/episode 队列。RxR joint 单卡 2 次更新并严格恢复到第 3 次、双卡 2 次更新和 joint 单 episode 评测均退出 0；75k NWM 每个 rank 严格匹配 314/314，单卡/双卡峰值显存约 10.33/11.43 GiB，Top-5 与 E24 均有非零梯度，q0/CWP/q1 batch/row failure 为 0，q1 Oracle 为 0。恢复后 joint 单 episode 有 q0 15/15、CWP 15 次、q1 14/14、future-valid 14，action-flip rate `0.111111`。测评机专用容器另完成 46 项配置/脚本/现代运行时测试。实现最终提交为 `4c57581`；正式 30,000/10,000 长训练、完整 val_unseen、正式选优和大 checkpoint 传输均未启动。

2026-08-26，任务上下文：完成原生 `[CLS + 256 patch]` RAE-NWM 导航前瞻替换。新模式严格加载 75k EMA 的 314 个键，输出 normalized/raw CLS 和 patch；在线 DINO 保留 raw CLS、nav CLS、raw patch，RGB 融合使用 CLS 余弦一致性，并通过 Top-5 CLS adapter 与 E24 的 adjusted full-logits CE 联合训练。模型 checkpoint 格式为 `etpr1-native-cls-e24-joint-v2`，training-state 为 v4，保存四个优化器组及每 rank RNG/NWM generator。训练机 10 步跨版本 parity 的完整 token cosine 为 `0.999984`、CLS cosine 为 `0.999981`、最大绝对误差为 `0.07958`；单卡保存/恢复和双卡两次更新均退出 0，双卡 Top-5/E24 梯度范数为 `0.329/2.542`。测评机固定单 episode 成功：75k EMA `314/314` 严格匹配，q0/CWP/q1 均 `33/33` 成功，批次/行失败为 0，context 不足的早期候选未请求 NWM；自带运行时版本记录的结果与诊断位于 `data/logs/active_lookahead/native_cls_e24_single_episode_eval/20260826_smoke_iter2_v2/`。主要入口为 `run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml`、`scripts/manage_rae_r2r_native_cls_e24_joint_server.sh` 和 `scripts/manage_native_cls_e24_single_episode_eval_host.sh`。

2026-08-26，任务上下文：只读核对 R2R-CE test 推理与论文报告口径。确认本仓库 `inference` 会遍历 3,408 个 test episode，生成以 episode ID 为键、轨迹点含 `position`、`heading`、`stop` 的官方格式 JSON，但不会本地计算指标；现成 `run_r2r/main_server.bash infer` 固定旧 CLIP 四卡配置，现代 RAE/DINO 模型应使用匹配配置和预训练基座单卡运行。EvalAI Challenge 719 的官方 API 显示 `is_active=false`、`is_frozen=true`、结束时间为 2026-01-31，页面公告建议后续论文报告完整 `val_unseen`。新模型应从 1,839 个 `val_unseen` episode 的结果读取 NE、OSR、SR、SPL；不得把验证集结果填入 Test Unseen 列。另确认现代配置默认 `IL.back_algo=teleport`，而原版论文评测/推理脚本显式使用 `control`；当前 GRPO 750 的已有 `val_unseen` 结果 NE/OSR/SR/SPL 为 `4.1112/0.6939/0.6411/0.5543`，但在与原论文进行最终同口径比较前必须以 `control` 重跑，不能直接当作论文最终值。历史 ETP-R1 论文的 Ours-GRPO Test Unseen 数字仍为 NE 4.19、OSR 69、SR 64、SPL 54，但不能复用于新模型。

2026-08-18，任务上下文：统计 GRPO 采样中“缓冲区记录非停止动作，但因达到最大轨迹长度或无剩余路点而实际执行停止”的频率。现有训练日志和状态未保存停止原因，无法事后精确反推；因此在实际训练机分支提交 `6a77c8a` 增加只采样、不反向、不更新、不保存 checkpoint 的计数模式，训练机专用环境针对性测试 `16 passed`。使用原训练设置（双卡、每卡 8 环境、每个 episode 8 次随机采样、dropout 与 waypoint augmentation 开启），分别从 SFT 第 14,200 次起点、GRPO 第 20/750/1000 次 checkpoint 各采样 5 batch、640 条轨迹。错记频率依次为 `86/640=13.4375%`、`144/640=22.5%`、`90/640=14.0625%`、`111/640=17.34375%`；对应全部高层动作占比为 `1.4515%`、`2.1951%`、`1.4898%`、`1.7860%`。四组共 2,560 条轨迹、24,741 个高层决策，错记 431 条轨迹；全部 431 次都来自第 15 个高层动作达到 `max_traj_len` 后强制停止，`no_vp_left` 强停和其他非显式停止结束均为 0。结果表明问题在轨迹级并不低频：SFT 起点和最佳 GRPO 约每 7 条轨迹一次，早期第 20 次约每 4.4 条一次，训练终点约每 5.8 条一次；第 20 次升高与此前观察到的早期轨迹变长、停止退化方向一致，但该复测只证明触发频率，尚未通过修复对照实验测定它对最终 SR/SPL 的因果贡献。原始日志在训练机 `data/logs/diagnostics/grpo_stop_accounting_20260818/`；诊断后两卡回到 45/16 MiB、无训练进程、未生成 `.pth`。

2026-08-18，任务上下文：只读分析“当前 SFT 最佳高于官方 SFT、但 GRPO 最佳低于官方 GRPO”的原因。当前 SFT 的优势不是所有指标全面领先：相对官方 SFT，SR/SPL、nDTW/SDTW 分别高 `0.60/1.41`、`3.34/2.57` 个百分点，路径短 `1.89 m`、少 `11.11` 步，但 OSR 低 `0.38` 个百分点、碰撞多 `0.114`；优势主要来自第二轮原生兼容 DINO/raw-CLS 基座带来的路径表示与效率，以及在两轮共 150 个 `val_unseen` checkpoint 中按 SR+SPL 取最大值。第二轮第 14,200 次峰值（SR/SPL `63.73/55.61`）相邻点明显波动，故不能把这点优势全部视为稳定泛化增益。GRPO 超参数表面沿用官方，但优化预算不等价：官方为 4 卡×8 环境、500 次，每次更新 32 个 episode；当前为 2 卡×8 环境、1,000 次，每次更新 16 个 episode，在相同学习率下用半批量、双倍优化器步数保持总 rollout 预算，带来更高梯度噪声和不同参数移动量。当前 GRPO 相对 SFT 起点使 OSR/SR/SDTW提高 `1.25/0.38/0.67` 个百分点，却使 SPL/nDTW下降 `0.18/0.37` 个百分点、平均步数增加 `4.17`，说明奖励推动了更多探索/到达，但没有有效转化为更好的停止与路径效率。采样代码确实会先记录采样动作，再在最大步数或无剩余路点时强制执行停止，导致终局奖励可能记到未执行动作；但初始提交 `69139a8` 的官方代码也有同一逻辑，因此它是共同风险，若无两边触发频率统计，不能单独解释当前模型低于官方。现有结论是：更强且经过 150 点选择的 SFT 起点减少了 GRPO 提升空间；官方为 CLIP/官方 SFT 调出的 GRPO 参数直接迁移到 DINO 基座，加上半 batch、双倍更新的优化差异，是更有证据的差距来源。要严格分解需在独立选择集/测试集复评，并做相同 4 卡×8 环境×500 次或等效学习率缩放的对照实验。

2026-08-18，任务上下文：只读对比两轮 R2R SFT 基座及其影响。第一轮加载 `model_step_452500_nonvisual_transfer.pt`：原 452,500 步基座使用 `stat.pt` 归一化 CLS、三层 `768→768→768→512` 投影和 `512→768` 映射；转成当前 raw-CLS 接口时删除了 6 个旧投影参数，并用从未接受预训练更新的确定性随机 `768→768` `img_linear` 替换旧映射，只保留非视觉权重。第二轮加载原生兼容的 `model_best_step_465000.pt`，其 raw CLS `768→768` `img_linear` 已在联合预训练中更新。两轮在线 CLS residual MLP 都以零初始化残差、即恒等映射开始，并在 SFT 中训练。实际启动日志确认两轮均为双卡、每卡 8 环境、梯度累积 1、15,000 次及相同 SFT 调度；启动提交 `97d397f` 与 `abed04d` 之间只新增自动接力监控脚本，没有模型或训练逻辑差异。完整评测曲线显示第二轮主要优势是更快起步：SR 首次达到 50% 为第 600 次、第一轮为第 2,200 次；SPL 首次达到 50% 为第 2,000/3,600 次；`SR+SPL≥1.10` 为第 2,600/4,200 次。到第 8,000 次后差距明显缩小且部分点互有胜负；最终两轮最佳分别为第一轮第 14,000 次（SR/SPL `0.6302338227`/`0.5421013044`）和第二轮第 14,200 次（`0.6373028820`/`0.5560538836`）。因此现有证据较强地支持“原生兼容预训练视觉映射提高样本效率和早期收敛”，但单次随机训练与 75 点取最大值不足以把最终约 `0.71/1.40` 个百分点的 SR/SPL 优势全部归因于更高模型上限。

2026-08-18，任务上下文：只读核对两轮 R2R SFT 与一轮 GRPO 的最终最佳 checkpoint，并与官方同阶段发布 checkpoint 做同口径对比。训练机接力状态确认 GRPO 已正常完成 1,000 次更新，100/100 个 checkpoint 均已完成测评；两轮 SFT 的 150/150 个结果也完整。统一在完整 1,839 个 R2R-CE `val_unseen` episode 上按 `success + spl` 选择：第一轮 SFT 最佳为 `ckpt.iter14000.pth`（SR `0.6302338227`、SPL `0.5421013044`），第二轮及两轮总最佳为 `ckpt.iter14200.pth`（SR `0.6373028820`、SPL `0.5560538836`）；GRPO 最佳为 `ckpt.iter750.pth`（SR `0.6411092985`、SPL `0.5542831948`）。官方 SFT `ckpt.iter25000.pth` 的 SR/SPL 为 `0.6313213706`/`0.5419954062`，官方 GRPO `ckpt.iter270.pth` 为 `0.6541598439`/`0.5586563945`。因此当前最佳 SFT 的 SR、SPL 分别高官方 `0.5982`、`1.4058` 个百分点；当前最佳 GRPO 分别低官方 `1.3051`、`0.4373` 个百分点。GRPO 相对其 SFT 起点的 SR 增加 `0.3806` 个百分点，但 SPL 减少 `0.1771` 个百分点，`SR+SPL` 仅净增 `0.2036` 个百分点。原始结果保存在测评机对应 SFT/GRPO `eval_watch_val_unseen/.../eval_results/` 目录；本次未修改训练、评测进程或远端产物。

2026-08-17，任务上下文：只读核查 GRPO step 20 继续退化是否由此前的在线全景视图顺序问题复发。训练机和测评机当前提交都包含 `a676fc2`；SFT、GRPO 训练和评测均通过同一个 `R1Policy.pack_panoramic_observations()` 按数值角度排序，再构造 `[0, 330, ..., 30]` 的顺时针输入，SFT/GRPO 的 `_vp_feature_variable()` 也逐行一致。两台实际运行环境的三项全景排序回归测试分别为 `3 passed`，因此现有证据排除旧全景顺序 bug。step 20 在 1,839 个 episode 上 success/SPL 为 `0.6095704192`/`0.5026693046`，oracle success 却升至 `0.6878738445`；“到过成功范围但最终失败”的 episode 从 SFT 起点的 81 个增至 step 10 的 111 个、step 20 的 144 个，更指向停止决策退化。另发现原始 GRPO 采样代码先保存采样动作，随后在达到最大步数或无剩余路点时可能强制执行停止，却没有把缓冲区动作同步改为停止；这会把最终奖励记到未实际执行的非停止动作上，是与当前现象吻合的高风险点，但现有日志未记录触发次数，尚不能确认为唯一根因。本次未修改或停止训练与评测。

2026-08-17，任务上下文：只读诊断 GRPO 第 10 次更新的完整 R2R `val_unseen` 指标低于起始 SFT。相同的 1,839 个 episode 上，success 从 `0.6373028820` 降到 `0.6247960848`，SPL 从 `0.5560538836` 降到 `0.5294519878`；但 oracle success 从 `0.6813485590` 略升到 `0.6851549755`，同时平均步数、路径长度、最终目标距离和高层步数均上升，表现更像早期更新使路径效率与停止决策退化，而不是导航能力整体失效。启动日志确认 SFT 权重完整加载、无未处理缺失层、无 NaN/异常；前 10 次训练的原始梯度范数均值为 `13.834`，实际按 `2.0` 裁剪，KL 为 `0.072`，学习率仍约 `2e-5`。训练采样启用 dropout 和 waypoint augmentation，每次更新仅基于双卡各 8 个环境、每个 episode 8 次随机采样；而起点又是从同一验证集 150 个 SFT checkpoint 中选出的最高点，因此单个 step 10 结果不足以判定 GRPO 最终无效。第 20 次完整评测仍在进行，本次未修改或停止训练与评测。

2026-08-17，任务上下文：核对当前训练机 GRPO 的实际 SFT 起点。运行中 `torchrun` 命令、GRPO supervisor 启动日志和 `best_sft_grpo_followup_monitor/launched.env` 三者一致，均加载第二轮 `rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft` 的 `ckpt.iter14200.pth`；该 SFT 以预训练最佳 `model_best_step_465000.pt` 为基座。自动接力在两轮共 150 份完整 R2R `val_unseen` 结果中按 `success + spl` 选择，所选 checkpoint 的 success 为 `0.6373028820`、SPL 为 `0.5560538836`、合计 `1.1933567656`。

2026-08-17，任务上下文：把测评机当前 GRPO checkpoint 的完整 R2R `val_unseen` 测评持续接入现有联合 TensorBoard。`scripts/normalize_r2r_eval_tensorboard.py` 新增可配置标签前缀，GRPO 使用独立的 `grpo_eval_*` 图组，避免其 `10--1000` 横轴与 SFT 的 `200--15000` 横轴互相压缩；测评机专用环境针对性测试为 `7 passed`。容器内 `etpr1-grpo-eval-tb-normalizer` tmux 会话每 10 秒把新结果追加到 `current_grpo_best_sft`。官方 CLIP-GRPO `ckpt_270` 的 11 项指标以 `official_clip_grpo_ckpt270` 运行写在 step 10 和 1000，形成水平线。首个当前模型结果 step 10 已自动显示，其中 success 为 `0.624796093`、SPL 为 `0.529451966`；官方水平线对应 success `0.654159869`、SPL `0.558656373`。服务仍通过笔记本 `http://127.0.0.1:6007/` 访问，原有 SFT 转换器和曲线未改动。

2026-08-16，任务上下文：为测评机两轮 R2R SFT `val_unseen` 结果启动联合 TensorBoard，并修复原始事件文件合并后第一轮 step 从 6,401 回跳到 201 的折返曲线。`scripts/normalize_r2r_eval_tensorboard.py` 现在直接按结果文件中的 checkpoint 迭代号重建严格递增事件，并每 10 秒追加第二轮新结果；重启时按每个指标已落盘的最大 step 续写。服务运行在专用容器 `gwl-etpr1-rae` 的 6008 端口，显示为 `first_legacy452500` 和 `second_best465000`；笔记本通过用户级服务 `etpr1-sft-eval-tb-tunnel-ordered.service` 映射到 `http://127.0.0.1:6007/`。验证时第一轮 75 个点、第二轮 49 个点的 11 项指标均完整且 step 回跳数为 0。原始 JSON 和事件文件未修改，旧的测评机 6006 预训练 TensorBoard与原始双轮 6007 TensorBoard 均已停止。

2026-08-16，任务上下文：核对原版 ETP-R1 在线 SFT 的有效 batch 与训练步数。直接检查初始提交 `69139a8` 的 R2R/RxR 发布脚本和 SFT 训练循环，确认 R2R 为 4 卡 × 每卡 8 环境、30,000 次单次优化器更新，RxR 为 4 卡 × 每卡 6 环境、30,000 次；没有修改训练代码或运行任务。

2026-08-16，任务上下文：为测评机部署两轮 SFT checkpoint 评测自动接力。检查并验证了通用评测 watcher、第二轮固定参数入口和结果完整性/GPU 空闲交接脚本；本地与测评机容器针对性测试均为 `8 passed`。测评机现场确认第一轮只有一个 `run.py`，接力 watcher 正等待 61/75，第二轮没有提前启动。

2026-08-14，任务上下文：在保持每个 MLM/SAP DataLoader 两个 worker 的前提下修复预训练 CPU OOM。实现 JSONL 惰性 mmap 索引、有界特征 LRU、干净 `spawn`、训练/验证 worker 分离、预取和锁页内存约束，并补齐恢复脚本的显式参数。测评机专用容器完成真实样本数值一致性、2,000/5,000 micro-batch 内存平台、CUDA 先初始化顺序和退出清理验证；没有启动或恢复正式训练。

2026-08-14，任务上下文：只读排查最近一次联合预训练的缓慢内存增长与崩溃。核对了本地预训练入口、数据集缓存、训练/验证 DataLoader 生命周期、线程预取、训练循环、验证、日志与 checkpoint 保存；并检查训练机和测评机身份、Git 状态、进程、GPU、容器、supervisor 日志、内核 OOM 现场和容器 cgroup 峰值。确认训练机第 250,000 步状态完整，测评机从该点单卡恢复后在第 305,000 步验证阶段触发整机 OOM，最近完整状态为第 302,500 步。测评机专用环境的线程预取与 DDP 反向针对性测试为 `7 passed in 0.74s`。未启动、恢复或停止任务，未修改远端代码、环境或产物。

2026-08-13，任务上下文：按用户要求停止训练机和测评机的训练、评测任务。先确认训练机 `gwl-sever/gwl` 与测评机 `a6000/a6000` 的身份、工程分支和进程父链，再通过项目管理脚本停止训练机 R2R SFT、checkpoint 同步和测评机评测监控；对未随 `docker exec` 退出的当前评测进程组单独发送正常终止信号。最终跨宿主机和测评机全部运行中容器复核均未发现训练或测评进程，GPU 计算进程为空。代码、checkpoint、训练状态和已有评测结果均未删除。

2026-08-12，任务上下文：只读核验训练机本工程的实时训练进度。确认主机为 `gwl-sever`、用户为 `gwl`、工程分支 `main` 与 `origin/main` 一致且工作树干净；双卡 A6000 联合预训练已到 TensorBoard 第 224,346 步，最近完整断点第 222,500 步，最佳 checkpoint 第 220,000 步，训练进程、GPU、验证、日志和磁盘状态正常。未修改训练机代码、进程、环境或产物。

2026-07-29，任务上下文：为三机 Git 协作补充项目级规则。核对了训练机主仓库、现有工作树、未提交源码与文档、大文件边界、笔记本和测评机工作目录；项目的中央裸仓库固定为 `/home/gwl/git/ETP-R1.git`，连接与安全同步方式见根目录 `AGENTS.md`。

2026-07-24，修复 TensorBoard 第 180,000 步三项 MLM 指标突变。`normalize_pretrain_tensorboard.py` 现为每个原始事件文件持续记录同一验证标签、同一步的出现次数，将第一组映射为 R2R、第二组映射为 RxR；该状态跨实时刷新周期保留，因此 RxR 晚几十秒到达时不会再被当作重复项丢弃。测评机针对性测试为 `5 passed, 2 deselected, 1 warning`。新校正目录 `tensorboard_normalized/live_20260724_split` 已重建 724,421 条历史标量并持续追加；核对第 175,000、177,500、180,000 步的 R2R/RxR 三项 MLM 指标均与原始事件一致。6006 已切换到新目录，TensorBoard PID 为 1056522，旧校正器已关闭，训练未中断。

2026-07-24，诊断 TensorBoard 在第 180,000 步的三项 MLM 验证指标突变。原始事件显示第 177,500 到 180,000 步期间，R2R 的准确率 `0.8016 -> 0.8039`、损失 `0.8234 -> 0.8220`，RxR 的准确率 `0.9044 -> 0.9050`、损失 `0.4057 -> 0.3973`，两者都平稳。突变来自两套验证共用 `valid_unseen_mlm/*` 标签：历史校正保留同一步较晚的 RxR，新增量校正从第 180,000 步开始保留先到的 R2R并跳过同一步 RxR。只读诊断期间训练和 TensorBoard 校正进程均保持运行，未实施修复。

2026-07-24，修复测评机 6006 TensorBoard 停止更新。训练原始事件文件一直增长，TensorBoard 网页进程也存活；实际停止的是实时校正进程。宿主机内核日志显示该 Python 进程于 2026-07-23 14:49:31 发生 general protection fault，时间与校正日志停在第 121,082 步完全一致。旧校正实现每 10 秒通过 `EventAccumulator` 完整重读约 48 万条历史标量，现改用 `LegacyEventFileLoader` 保持文件游标并只读取新增事件。测评机专用环境针对性测试结果为 `3 passed, 2 deselected, 1 warning`。新校正器在容器内 `etpr1-tensorboard-normalizer` tmux 会话运行，输出目录为 `tensorboard_normalized/live_20260724_incremental`，日志为 `supervisor/tensorboard_normalizer_incremental.log`；宿主机 TensorBoard PID 为 1054224，HTTP 返回 200。接口实测 `loss/mlm` 从 178,106 推进到 178,164，校正器推进到 178,165，训练进程全程未停止。

2026-07-23，修复断点续训后 TensorBoard 的 loss 曲线只显示到局部步数的问题。根因是 `TensorboardLogger` 每个新进程都从 0 计数，且 loss 使用该内部计数，学习率和梯度却使用 checkpoint 恢复出的全局步数。新增 `set_step()`、在恢复后同步日志步数、调整 loss 写入顺序，并增加 `scripts/normalize_pretrain_tensorboard.py` 对已有事件进行实时校正；测评机临时副本的针对性测试为 `4 passed, 1 warning`。为避免中断正式训练，本次没有替换运行中进程加载的源码，也没有重启训练；校正进程由容器内独立 tmux 会话持续读取原始日志，TensorBoard 仅重载到校正目录。持续验证时校正日志已追加到 120,233 步，TensorBoard 的 `loss/mlm` 已显示到 120,231 步、HTTP 返回 200，训练进程始终存活。

2026-07-23 13:12--13:33（Asia/Shanghai），经用户明确授权先正常停止当时运行到约 118,775 步的任务；因未到保存点，正式恢复点仍为 117,500。使用新增压力工具在真实 321 万条数据上分别完成 `fork + worker=1 + pin_memory` 和关闭 `pin_memory` 的 5,000 micro-batch 测试：两者都没有在短测窗口复现低概率段错误，吞吐约 29.7/29.4 batch/s；但 MLM、SAP 实际各产生一个 worker，每个 worker RSS 从约 14GB 增长到约 15GB，说明关闭锁页内存没有消除大进程 fork 风险。随后正式启用 `n_workers=0`、`pin_mem=false`、`thread_prefetch=true`，用 `train_state_117500.pt` 完成真实模型短测至第 117,703 步，平均约 1.27 秒/步，相比同步单进程约 1.48 秒/步提速约 14%，且无子进程或异常。修复后完整测试为 `290 passed, 3 warnings`。最终通过托管脚本重新恢复正式训练，source identity 为 `tree-cd2160f02558c646fd64abbbd001def981806b4b6daddacb68cd632a014159a2`，日志为测评机 `/home/a6000/gwl/ETP-R1/pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/supervisor/resume_20260723T053137.log`；任务已推进超过恢复点，GPU 利用率约 90%，没有 DataLoader 子进程。

2026-07-23，为复现和修复预训练间歇性本地内存崩溃而检查 `train_r2r.py`、`data/loader.py`、`data/dataset.py`、`data/tasks.py` 和实际进程内存。确认旧路径会在 CUDA/模型初始化之后，以 Linux 默认 `fork` 为 MLM、SAP 各启动一个 worker；主进程约 80 个线程、20 GiB RSS，且 HDF5 已在 fork 前初始化。新增实际数据管线压力工具 `scripts/stress_pretrain_dataloader.py`，可对 worker 数、锁页内存、进程启动方式和 CUDA 初始化顺序做分组测试；另实现可选的单进程后台线程预取以避免 fork 并尝试恢复吞吐。新代码只同步到测评机临时目录 `/home/a6000/gwl/etpr1-prefetch-test.O6nOAt` 做 CPU 轻量测试，结果为 `10 passed in 0.96s`；没有覆盖正在运行的正式工程源码，也没有并行启动真实数据长压测或 GPU 测试。正式训练继续使用 `n_workers=0`、`pin_mem=false`。

2026-07-23 12:39（Asia/Shanghai），确认 ETPNav 无训练/评测进程且 GPU 空闲后，把正式预训练配置从 `n_workers=1`、`pin_mem=true` 调整为 `n_workers=0`、`pin_mem=false`，同步到测评机并通过 JSON 解析与校验和核对。随后使用 `scripts/manage_rae_pretrain_host.sh resume` 从完整的 `train_state_117500.pt` 恢复；日志明确打印恢复步数 117,500、batch 16、梯度累积 8 次。进程树只有 torchrun 和 rank 0 训练主进程，没有 DataLoader 子进程；训练已推进到至少第 117,520 步，GPU 显存约 17.2 GiB、利用率约 69%，任务仍在运行。当前 supervisor 日志为测评机 `/home/a6000/gwl/ETP-R1/pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/supervisor/resume_20260723T043910.log`。

2026-07-23，为诊断从第 80,000 步恢复后的正式预训练再次意外停止而只读复查。确认任务在第 117,624 步附近由 rank 0 Python 主进程收到 `SIGSEGV`，宿主机内核在同一时刻记录其崩在 `libc.so.6`；没有 OOM、NVIDIA Xid、磁盘不足、容器重启或主机重启。结合此前第 11,195 步的 DataLoader worker 堆内存错误和第 80,204 步的 DataLoader worker 段错误，当前判断为多进程数据管线相关的间歇性本地内存破坏，但缺少 core dump，不能精确定位到单个本地库函数。最新完整恢复点 `train_state_117500.pt` 已用 CPU 成功加载，含 step 117,500、meta loader step 940,000、484 组优化器状态。未恢复训练，未修改测评机代码、环境或产物。

2026-07-22 14:29--15:33（Asia/Shanghai），在确认 ETPNav 无任务、GPU 空闲后，通过 `scripts/manage_rae_pretrain_host.sh resume` 从 `train_state_80000.pt` 恢复正式预训练。恢复日志确认全局步数 80,000、数据混合步数 640,000 和 484 组优化器状态；随后连续监控一小时，tmux、训练主进程和两个数据加载子进程始终存活，GPU 利用率为 87%--96%，未出现数据加载、显存或分布式启动错误。训练推进到至少第 82,969 步，并成功写入 `train_state_82500.pt`；结束检查时任务仍在运行。当前 supervisor 日志为测评机 `/home/a6000/gwl/ETP-R1/pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/supervisor/resume_20260722T062918.log`。

2026-07-22，为诊断测评机正式预训练的停止原因而只读复查。直接检查了两份 supervisor 日志、断点目录、实际预训练配置和数据加载源码；确认从 10,000 步恢复的任务在 80,204 步再次由 DataLoader worker 段错误终止，`train_state_80000.pt` 完整存在。未修改测评机代码、环境、训练进程或产物。

2026-07-15 23:35（Asia/Shanghai），正式长训练已通过 `scripts/manage_rae_pretrain_host.sh resume` 从 `train_state_10000.pt` 恢复，配置保持 `n_workers=1`、batch 16、梯度累积 8 次。日志明确打印恢复全局步数 10,000，现场已推进到约 10,037 步；tmux、torchrun、训练进程和两个数据子进程均存活，GPU 利用率连续为 89%–95%。本次 supervisor 日志为测评机 `/home/a6000/gwl/ETP-R1/pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/supervisor/resume_20260715T153346.log`。

2026-07-15，为排查测评机正式长训练停止而复查。核对了测评机身份、容器/tmux/进程、GPU、系统与内核日志、训练 supervisor 日志、checkpoint 时间线、DataLoader/HDF5 代码、共享内存和实际 Python/PyTorch/NumPy/h5py/HDF5 版本；并在同一专用容器和环境中只读扫描两份完整 HDF5，确认数据集本身无坏视点。未修改训练代码，未重启任务。

2026-07-15，为完成 RAE/DINOv2 视觉编码器计划而复查。核对了隔离运行时、全量特征、真实 RAE 数值一致性、离线/在线投影、checkpoint 过滤、三阶段冻结、CLIP 旧 checkpoint 兼容、完整测试和正式 smoke。最终代码验证提交为 `5436799`，正式 smoke manifest 为 `f4503e77b2e338cc4f8efab5a5193ca8223f7afa1ee6e39509f0d16c371e5c76`；随后补做完整真实数据的 batch 32、batch 16 和 batch 16 加 8 次梯度累积测试，并把正式配置改为 batch 16 加 8 次累积。又检查并修改 `train_r2r.py`、`utils/save.py`、`data/loader.py`、预训练 parser/配置/启动脚本和新增的宿主机/容器托管脚本，完成真实两步跨进程恢复；随后增加 MLM+SAP 联合准确率最佳模型保存并完成真实 GPU 落盘验证，最终全量测试为 280 项通过。旧长跑在首个 checkpoint 前按用户要求停止，准备以新逻辑从头启动。

2026-07-11，为清理本机重复存储而复查。使用 `rsync -anrc --itemize-changes` 确认 `dataset/extra_files/` 的全部文件已完整存在于正式工程，只有目录时间戳差异；随后删除重复目录，释放约 21GB。删除说明见 `/home/gwl/project/etpr1/dataset/README.extra_files_removed_20260711.md`。

2026-06-08，为把 ETP-R1 建立在 `etpnav` 环境上并组织本地资源而检查。主要检查了 `README.md`、`environment.yaml`、运行脚本、配置文件、本机 conda 环境、本地 Habitat 目录、`extra_files.zip` 和 MP3D 数据。

2026-07-08，为项目熟悉和代码导览而复查。主要检查了 `README.md`、`research.md`、`run.py`、`run_r2r/main_server.bash`、`run_rxr/main_server.bash`、`run_r2r/iter_train.yaml`、`run_rxr/iter_train.yaml`、`vlnce_baselines/ss_trainer_ETP_R1.py`、`vlnce_baselines/GRPO_trainer_ETP_R1.py`、`vlnce_baselines/models/R1Policy.py`、`vlnce_baselines/models/graph_utils.py`、`vlnce_baselines/common/env_utils.py`、`habitat_extensions/task.py`、`pretrain_src/run_pt/run_mix_server.bash`、`pretrain_src/run_pt/mix_pretrain_server.json` 和 `pretrain_src/pretrain_src/train_r2r.py`。

2026-07-08，为对比 ETP-R1 与原版 ETPNav 差异而检查。对比基准为本地 ETPNav 仓库早期提交 `85bed52`，临时 worktree 位于 `/tmp/etpnav-baseline-85bed52`。主要对照了 `run.py`、`run_r2r/main.bash`、`run_r2r/main_server.bash`、`run_r2r/iter_train.yaml`、`run_rxr/iter_train.yaml`、`vlnce_baselines/ss_trainer_ETP.py`、`vlnce_baselines/ss_trainer_ETP_R1.py`、`vlnce_baselines/GRPO_trainer_ETP_R1.py`、`vlnce_baselines/models/Policy_ViewSelection_ETP.py`、`vlnce_baselines/models/R1Policy.py`、`vlnce_baselines/models/etp/vilmodel_cmt.py`、`vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py`、`pretrain_src/run_pt/r2r_pretrain_habitat.json`、`pretrain_src/run_pt/mix_pretrain_server.json`、`pretrain_src/run_pt/r2r_model_config_dep.json` 和 `pretrain_src/run_pt/mix_model_config_dep.json`。

2026-07-08，为确认更换视觉编码器前的预训练图像采集和特征文件覆盖范围而检查。主要检查了 `precompute_img_features/save_img.py`、`precompute_img_features/extract_rgb_features.py`、`precompute_img_features/extract_depth_features.py`、`precompute_img_features/run.bash`、`pretrain_src/run_pt/mix_pretrain_server.json`、`pretrain_src/run_pt/mix_model_config_dep.json` 和 `pretrain_src/pretrain_src/data/dataset.py`；并用 `etpnav` 环境读取 HDF5 key 数和 shape。

2026-07-10，为核对论文列出的全部预训练数据而检查。对照本地论文 `/home/gwl/桌面/论文/ETP_r1.pdf` 的 Offline Joint Pretraining 小节，检查了 5 个训练 JSONL、2 个验证 JSONL、RGB/深度 HDF5、Matterport3D 场景、connectivity、`mix_pretrain_server.json`、`train_r2r.py` 和文本编码器权重；完成约 322 万条样本的逐行解析及轨迹视点覆盖校验。

2026-07-10，为分析将 CLIP 更换为本地 RAE-NWM 的 RAE/DINOv2-B 编码器而检查。主要检查了离线特征生成脚本、`R1Policy.py`、`CLIPEncoder`、预训练与在线 `ImageEmbeddings`、预训练 checkpoint 键名、路点预测器，以及 `/home/gwl/project/RAE-NWM/raenwm` 中的 DINOv2-with-registers-base、RAE 编码与 native 224 预处理链路。

2026-07-10，为把 RAE/DINOv2 的全部运行工作迁移到测评机而复查。只读核验了 `ssh 4090` 对应主机的 GPU、磁盘、Docker 容器、远端 `raenwm` 包版本、`etpnav-local-deps.pth` 和 RAE-NWM 权重位置；确定使用独立容器 `gwl-etpr1-rae`、独立环境 `etpr1_rae` 和 ETP-R1 自有 Habitat 依赖目录。本轮只更新约定与设计，没有创建远端环境或运行实验。用户随后明确授权停止当时正在运行的一个 ETPNav 评测任务；已向其 `torchrun` 主进程发送正常终止信号，任务进程树退出、GPU 释放，`gwl-etpnav` 容器未停止。该授权只适用于这个具体任务，后续仍默认禁止停止 ETPNav 进程。

2026-07-10，在设计获批后编写 RAE/DINOv2 实施计划。计划位于 `docs/superpowers/plans/2026-07-10-rae-dinov2-visual-encoder.md`，依次覆盖测评机隔离环境、现代 Habitat 兼容、三层投影、冻结编码器、离线预训练、在线 SFT/GRPO、checkpoint 过滤、全量 HDF5、CLIP 回归和最终冒烟。本阶段只形成计划，尚未创建 `gwl-etpr1-rae`、`etpr1_rae` 或开始模型实现。

2026-07-10，为 Task2 现代运行兼容复查。检查了 Python 3.11.15、NumPy 1.26.4、Habitat/Habitat-Sim/Habitat-Baselines 0.3.3、Transformers 4.49.0 的导入链，补齐旧配置与入口兼容，并确认 `run.py`、两个 R1 trainer、`habitat_extensions.task` 和 trainer 注册可用。代码审查后又补充了 R2R/RxR 旧配置到现代 OmegaConf 的桥接、原生配置入口幂等保护、动作编号保护和实际 `R1Env` 配置边界测试；真实 R2R/RxR 转换配置也已覆盖同进程与新进程 `pickle` 往返、深复制兼容，并确认不会遮蔽普通 OmegaConf 的同名数据字段。测评机最终运行 Task1+Task2 共 35 项测试通过，`python run.py --help` 退出码为 0。

2026-09-01，排查测评机 RGB-fusion R2R `val_unseen` 测评停止。根分区当时为 100%、可用空间 0，`native_cls_eval/watch.log` 在 `iter8000` 起明确报 `OSError: [Errno 28] No space left on device`。按用户授权停止该评测 watcher，并从测评机 `data/logs` 删除 186 个检查点文件及未完成传输片段，共 311,873,764,749 字节；训练机原件和已有评测结果保留。磁盘恢复到 68%、约 291 GB 可用。完整结果截至 `iter7400`，`iter7600` 的 episode 文件为 0 字节，因此从训练机经 2.5 GbE 直连原子同步 `iter7600` 至 `iter10000` 共 13 个待测检查点并重启 `scripts/manage_rae_r2r_native_cls_rgb_fusion_eval_watch_host.sh`。启动验收时 `iter7600` 已推进到 47/1839，GPU 利用率约 85%、显存约 13.1 GiB，根盘在同步后仍有约 272 GB 可用。
