# Project Research

## Project Goal

ETP-R1 是一个 VLN-CE 项目：让智能体在连续三维环境里，根据自然语言指令导航到目标位置。代码包含预训练、在线 SFT、在线 RFT/GRPO、R2R-CE 和 RxR-CE 评测流程。

## Quick Start And Environment

README 原始说明要求创建 `etpr1` conda 环境，核心环境为 Python 3.6.12、PyTorch 1.9.1+cu111，并使用 Habitat-Sim 0.1.7 和 Habitat-Lab 0.1.7。该说明和本机已有 `etpnav` 环境只用于了解旧 CLIP 链路，不作为本次 RAE/DINOv2 工作的运行方案。

已检查本机状态：

- `etpr1` 已按用户要求删除。
- `/home/gwl/miniconda3/envs/etpnav` 是旧链路参考环境，Python 3.7.16。
- `etpnav` 已能导入 Habitat-Lab 0.1.7、Habitat-Sim 0.1.7、PyTorch 1.9.1+cu111、TorchVision 0.10.1+cu111、TorchAudio 0.9.1。
- Habitat-Lab 0.1.7 来自 `/home/gwl/project/DGNav/habitat-lab`，通过 develop/egg-link 方式接入 `etpnav`。
- 运行项目入口需要 `torch.utils.tensorboard`，因此 `tensorboard` 使用 1.15.0。它和 `tensorflow 1.13.1` 的声明版本范围不完全一致，但实测两者都能导入，项目入口也能导入。

本次 RAE/DINOv2 替换的正式运行位置是测评机，不是本机：

- SSH 入口：`ssh 4090`；有线地址 `10.10.10.2`；GPU 为 RTX 4090 24GB；工作根为 `/home/a6000/gwl`。
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

## Important Modules And Functions

- `vlnce_baselines/ss_trainer_ETP_R1.py`: SFT/监督训练相关 trainer。
- `vlnce_baselines/GRPO_trainer_ETP_R1.py`: GRPO/RFT 相关 trainer。
- `vlnce_baselines/models/R1Policy.py`: ETP-R1 策略网络封装。
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py`: 核心多模态模型。
- `vlnce_baselines/common/env_utils.py`: Habitat 环境创建和并行环境工具。
- `habitat_extensions/task.py`: VLN-CE/RxR 数据集和任务扩展。
- `habitat_extensions/habitat_simulator.py`: 对 Habitat-Sim simulator 的项目定制封装。

## Data, Configs, And Artifacts

README 要求准备 Matterport3D 数据，目标结构是 `data/scene_datasets/mp3d/{scene}/{scene}.glb`。当前项目内 `data/scene_datasets/mp3d` 是软链接，指向本机已有数据 `/home/gwl/project/dataset/mp3d_unzipped/mp3d`，跟随软链接可看到 90 个 `.glb` 场景。

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

- 2026-08-14 已确认测评机单卡续训 `pretrain_resume_source_250000` 的停止原因是宿主机 CPU 内存耗尽，不是显存溢出。任务使用提交 `94f4372`，配置为 `n_workers=2`、`pin_mem=true`、`thread_prefetch=false`；它从第 250,000 步运行到第 305,000 步，在 RxR MLM 验证开始后由内核 OOM killer 以 `SIGKILL` 结束。内核现场有 7 个约 18 GiB RSS 的 Python 进程，正好对应 1 个训练主进程、MLM/SAP 共 4 个长期训练 worker 和当前验证的 2 个 worker；容器累计内存峰值为 65,468,919,808 字节，2 GiB swap 已耗尽。`R2RTextPathData(in_memory=True)` 会让每个 worker 独立、单调填充 RGB/深度视点缓存，完整缓存约 1.36 GB（十进制），同时 worker 通过 Linux `fork` 继承装有 321 万条 Python 记录的约 18 GiB 主进程，长期访问会增加写时复制的私有页。它是有上界的缓存和进程复制，不是计算图无界泄漏，但实际表现为缓慢增内存并最终 OOM；验证额外 worker 构成最后峰值。最近完整恢复点是第 302,500 步。此前已稳定长跑的 `n_workers=0`、`pin_mem=false`、`thread_prefetch=true` 路径没有这些子进程；本次只读诊断未恢复任务、未修改训练代码或远端产物。
- 2026-08-14 已实现保持 `n_workers=2` 的有界内存路径：321 万条 JSONL 改为约 24.5 MiB 的 mmap 行索引并按需解析；worker 使用 `spawn`；RGB+深度特征采用 256 MiB/worker 的按字节 LRU；预取降为 1；`pin_mem=false`；验证使用 `val_n_workers=0` 和零特征缓存。真实数据在 CUDA 先初始化的顺序下完成 2,000 micro-batch，500--2,000 batch 的 cgroup 内存稳定在约 22.82--22.88 GB，进程私有内存为主进程约 1.36 GiB、四个训练 worker 各约 1.38--1.43 GiB，退出正常。惰性/eager 的真实 R2R 首中尾样本和完整输入逐项一致，针对性测试 `44 passed`。测试中还发现 PyTorch `persistent_workers=true` 会在提前关闭时触发本地库 `SIGABRT`，最终配置已关闭。正式预训练现已在测评机从第 302,500 步恢复；2026-08-15 本次只读核验时已到约第 462,000 步，训练进程正常。当前最佳是第 452,500 步：MLM 平均准确率 `0.8666867801`、SAP 平均准确率 `0.8011502767`、联合选择分数 `1.6678370568`；对应 R2R/RxR 的 MLM 为 `0.8199916701`/`0.9133818901`，SAP 为 `0.8063005534`/`0.796`。新配置的每轮验证稳定约 166--172 秒（R2R 约 35 秒，RxR 约 131--136 秒）；旧配置共用 `n_workers=2` 时约 49 秒。差异主要来自新设置显式使用 `val_n_workers=0`，使验证的 HDF5 读取、样本构造和 batch 整理与 GPU 前向串行；这是为避免旧路径在验证 worker 创建时再次触发内存峰值的保守配置，不是训练 worker 卡死或验证逐轮退化。
- 2026-08-15 对比当前 raw-CLS 实验与上一次完整 `rae_dinov2_cls_mlp` 实验：两者不是同配置复跑。旧实验使用经 `stat.pt` 归一化的 DINO CLS，再经可学三层 `768→768→768→512` 投影和 `512→768` 映射；当前实验为对齐 ETPNav，改用未应用 `stat.pt` 的 raw CLS 和单层 `768→768` 映射。旧实验有效 batch 为 128；当前实验前 10,000 步为 128，之后主动允许改为 64，因此同为 500,000 次参数更新时约只看到旧实验一半的训练样本。分数差距在前半程已存在：旧实验第 250,000 步最佳 `1.666969`，当前实验第 252,500 步为 `1.646896`；故后续转移到测评机、OOM 和 `val_n_workers=0` 不是主因。验证子集由固定 seed 选取，但 MLM mask 和 SAP 终点类型仍是每轮随机生成，因此单点最佳分数也包含一定验证噪声。现有证据能说明主要差异来自视觉链路和有效 batch，但没有单变量实验能精确分解两者各自的贡献。
- 2026-08-15 当前训练机 R2R SFT 正式入口为 `scripts/run_rae_r2r_sft_server_job.sh`，使用双卡 DDP、每卡 8 个 Habitat 环境、每卡 batch 8、梯度累积 1，即每次优化器更新全局约 16 条轨迹。学习率 `1e-5`，预热 500 次；`min_lr_ratio=1.0` 使预热后实际保持恒定学习率。DAgger 教师采样初始比例 `0.75`，每 3,000 次按幂衰减；最长轨迹 15，最长文本 150，waypoint augmentation 开启。DINO 主干、深度编码器和路点预测器冻结，零初始化的 CLS residual MLP 与导航策略其余部分参与训练。上一次 `r2r_sft_formal` 实际使用 15,000 次；当前脚本默认已改为 2,000 次的 `panorama_order` 检查实验，且默认预训练路径仍是 raw-CLS 实验的 `model_best_step_220000.pt`，启动新正式 SFT 前必须明确覆盖为本轮选定的预训练 checkpoint 和所需总迭代数。上一次 50 万步 `rae_dinov2_cls_mlp/model_best_step_452500.pt` 属于已退役的旧视觉接口：其经预训练的 CLS MLP 是 `768→768→768→512`，后接 `img_linear 512→768`。当前 SFT 接口则是 raw CLS 上的残差 MLP `768→768→768→768`，再接 `img_linear 768→768`；当前代码会直接拒绝加载旧 checkpoint。若使用旧的 nonvisual-transfer 转换，旧 `rgb_projection` 和形状不匹配的 `img_linear` 都被丢弃；新 residual MLP 与 `img_linear 768→768` 均未经联合预训练，视觉桥接层需从 SFT 开始学习。
- 2026-08-15 18:12（Asia/Shanghai），已在训练机从上述旧最佳的转换产物 `rae_dinov2_etpnav_cls_768_legacy_base_transfer/model_step_452500_nonvisual_transfer.pt` 启动新的双卡 R2R SFT。实验名为 `rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft`，输出目录为 `data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815`，总迭代数 15,000；checkpoint 同步目标也已隔离到测评机对应的新目录。启动提交为训练机分支 `feature/world-model-migration` 的 `97d397f`。两个 rank 均成功加载转换 checkpoint，连续运行到第 200 次更新后已成对写出约 1.53 GB 的 `ckpt.iter200.pth` 和约 3.00 GB 的 `train_state.iter200.pth`，随后继续进入下一轮更新；模型 checkpoint 也已通过直连同步到测评机的新目录。保存后两张 A6000 显存约 21.7/20.1 GiB、利用率约 62%/61%，未见缺失键、形状冲突、异常退出或显存溢出。启动日志为 `supervisor_server/start_20260815T181211.log`。
- 2026-08-15 18:33（Asia/Shanghai），测评机已启动本次 SFT 的专用评测 watcher，日志为 `data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_legacy452500_nonvisual_20260815/eval_watch_val_unseen/watch.log`。启动时已收到 2 个 SFT checkpoint、结果数为 0；日志明确以 `reason=blocking_project_task` 等待仍在运行的预训练，没有创建评测 `run.py` 进程。18:44 收紧进程匹配条件以排除长期 TensorBoard 后重启，当前 PID 为 `1489031`；预训练退出后还会检查 4090 显存不超过 1 GiB，才按迭代正序逐个进行完整 R2R `val_unseen` 评测。
- 2026-08-15 18:45（Asia/Shanghai），训练机已启动“当前 SFT 完成后使用测评机最终预训练最佳模型再训一次”的接力 watcher，PID 为 `983219`，日志为 `data/logs/rae_dinov2_etpnav_cls_768/eval_best_sft_followup_monitor/watch.log`。部署时当前 SFT 仍正常运行，watcher 状态为 `reason=current_sft_running`。测评机预训练当时的临时最佳为第 465,000 步、联合分数 `1.6718887749`，但 watcher 不提前固定该点；它只会在测评机第 500,000 步模型和训练状态存在且 supervisor 为 `exit_code=0` 后读取最终最佳。预训练进程判定同时匹配 `train_r2r.py` 与输出根目录，不会把长期 TensorBoard 误认为训练进程。下一轮固定沿用双卡、每卡 8 环境、每卡 batch 8、梯度累积 1、15,000 次及当前全部 SFT 调度参数，并使用新的实验与同步目录。
- 2026-08-16 10:21（Asia/Shanghai），两个自动接力均已生效。测评机预训练于 01:11 正常完成第 500,000 步，最终最佳仍为第 465,000 步、联合分数 `1.6718887749`。第一轮 legacy452500 SFT 于 10:09 正常完成 15,000 次并以 `exit_code=0` 退出；训练机接力 watcher 随后复制并校验最终最佳模型，于 10:10 启动同参数第二轮 `rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft`，核验时约到第 227 次且第 200 次 checkpoint 已同步测评机。测评机 watcher 已完成第一轮 SFT 的 60/75 个完整 R2R `val_unseen` 评测，正在评估第 61 个 `ckpt.iter12200.pth`；两台机器均无运行时错误。
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
- 测评机只有一张 RTX 4090。现有 ETPNav 任务占用 GPU 时，不得并行启动全量特征生成、预训练、SFT、GRPO 或完整评测，也不得擅自中断 ETPNav。

## Last Reviewed

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
