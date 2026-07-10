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

`extra_files.zip` 位于 `/home/gwl/project/etpr1/dataset/extra_files.zip`，已解压到 `/home/gwl/project/etpr1/dataset/extra_files`，并已合并复制到项目根目录。关键资源已存在：

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

本地完整 checkpoint 评测记录见 `docs/ETP-R1_checkpoint_eval_comparison.md`，里面包含单卡评测命令、R2R/RxR 四个 checkpoint 的指标、结果文件路径和并行环境数量调整记录。ETP-R1 与原版 ETPNav 的代码差异分析见 `docs/ETP-R1_vs_ETPNav_diff_analysis.md`。

## Current Caveats And Open Questions

- `pip check` 会报告 `tensorflow 1.13.1` 声明要求 `tensorboard<1.14`，但 PyTorch 1.9 的 tensorboard 接口要求 `tensorboard>=1.15`。当前选择 `tensorboard==1.15.0`，因为这是项目入口能导入的最低可用折中。
- 真实训练/评测会启动 Habitat 环境和加载大模型，尚未完整跑一轮；当前完成的是依赖导入、配置解析和资源路径验证。
- 默认旧配置里还有 `habitat_extensions/config/vlnce_task.yaml` 这类历史路径，但 README 的实际脚本使用 `run_r2r/iter_train.yaml` 和 `run_rxr/iter_train.yaml`，这两个路径已验证可解析。
- 联合预训练配置将 `max_txt_len` 设为 250，`dataset.py` 会截断更长的指令。现有数据中 RxR-Marky 有 38,456 条、RxR train 有 3,097 条超过 250 个词元；这是训练配置造成的截断，不是数据文件缺失。
- 5 类数据的训练就绪 JSONL 都完整，但转换脚本引用的部分原始源文件和 Gemini 标注中间文件未按原路径保存在当前仓库中。因此可以直接运行联合预训练，但若要从原始指令和 Gemini API 输出开始重新生成全部 JSONL，还需要另行补齐源数据。
- 本机 `etpnav` 环境是 Python 3.7、PyTorch 1.9.1、Transformers 4.12.5，不包含 RAE-NWM 所用的 `Dinov2WithRegistersModel`，因此只能参考旧 CLIP 链路。测评机现已建立独立容器 `gwl-etpr1-rae`、独立 `etpr1_rae` 环境和 ETP-R1 自有 Habitat 0.3.3 依赖目录。现代导入链还需要仓库原环境固定的 `boto3==1.20.31`；它只安装在 `etpr1_rae` 中。
- 测评机只有一张 RTX 4090。现有 ETPNav 任务占用 GPU 时，不得并行启动全量特征生成、预训练、SFT、GRPO 或完整评测，也不得擅自中断 ETPNav。

## Last Reviewed

2026-06-08，为把 ETP-R1 建立在 `etpnav` 环境上并组织本地资源而检查。主要检查了 `README.md`、`environment.yaml`、运行脚本、配置文件、本机 conda 环境、本地 Habitat 目录、`extra_files.zip` 和 MP3D 数据。

2026-07-08，为项目熟悉和代码导览而复查。主要检查了 `README.md`、`research.md`、`run.py`、`run_r2r/main_server.bash`、`run_rxr/main_server.bash`、`run_r2r/iter_train.yaml`、`run_rxr/iter_train.yaml`、`vlnce_baselines/ss_trainer_ETP_R1.py`、`vlnce_baselines/GRPO_trainer_ETP_R1.py`、`vlnce_baselines/models/R1Policy.py`、`vlnce_baselines/models/graph_utils.py`、`vlnce_baselines/common/env_utils.py`、`habitat_extensions/task.py`、`pretrain_src/run_pt/run_mix_server.bash`、`pretrain_src/run_pt/mix_pretrain_server.json` 和 `pretrain_src/pretrain_src/train_r2r.py`。

2026-07-08，为对比 ETP-R1 与原版 ETPNav 差异而检查。对比基准为本地 ETPNav 仓库早期提交 `85bed52`，临时 worktree 位于 `/tmp/etpnav-baseline-85bed52`。主要对照了 `run.py`、`run_r2r/main.bash`、`run_r2r/main_server.bash`、`run_r2r/iter_train.yaml`、`run_rxr/iter_train.yaml`、`vlnce_baselines/ss_trainer_ETP.py`、`vlnce_baselines/ss_trainer_ETP_R1.py`、`vlnce_baselines/GRPO_trainer_ETP_R1.py`、`vlnce_baselines/models/Policy_ViewSelection_ETP.py`、`vlnce_baselines/models/R1Policy.py`、`vlnce_baselines/models/etp/vilmodel_cmt.py`、`vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py`、`pretrain_src/run_pt/r2r_pretrain_habitat.json`、`pretrain_src/run_pt/mix_pretrain_server.json`、`pretrain_src/run_pt/r2r_model_config_dep.json` 和 `pretrain_src/run_pt/mix_model_config_dep.json`。

2026-07-08，为确认更换视觉编码器前的预训练图像采集和特征文件覆盖范围而检查。主要检查了 `precompute_img_features/save_img.py`、`precompute_img_features/extract_rgb_features.py`、`precompute_img_features/extract_depth_features.py`、`precompute_img_features/run.bash`、`pretrain_src/run_pt/mix_pretrain_server.json`、`pretrain_src/run_pt/mix_model_config_dep.json` 和 `pretrain_src/pretrain_src/data/dataset.py`；并用 `etpnav` 环境读取 HDF5 key 数和 shape。

2026-07-10，为核对论文列出的全部预训练数据而检查。对照本地论文 `/home/gwl/桌面/论文/ETP_r1.pdf` 的 Offline Joint Pretraining 小节，检查了 5 个训练 JSONL、2 个验证 JSONL、RGB/深度 HDF5、Matterport3D 场景、connectivity、`mix_pretrain_server.json`、`train_r2r.py` 和文本编码器权重；完成约 322 万条样本的逐行解析及轨迹视点覆盖校验。

2026-07-10，为分析将 CLIP 更换为本地 RAE-NWM 的 RAE/DINOv2-B 编码器而检查。主要检查了离线特征生成脚本、`R1Policy.py`、`CLIPEncoder`、预训练与在线 `ImageEmbeddings`、预训练 checkpoint 键名、路点预测器，以及 `/home/gwl/project/RAE-NWM/raenwm` 中的 DINOv2-with-registers-base、RAE 编码与 native 224 预处理链路。

2026-07-10，为把 RAE/DINOv2 的全部运行工作迁移到测评机而复查。只读核验了 `ssh 4090` 对应主机的 GPU、磁盘、Docker 容器、远端 `raenwm` 包版本、`etpnav-local-deps.pth` 和 RAE-NWM 权重位置；确定使用独立容器 `gwl-etpr1-rae`、独立环境 `etpr1_rae` 和 ETP-R1 自有 Habitat 依赖目录。本轮只更新约定与设计，没有创建远端环境或运行实验。用户随后明确授权停止当时正在运行的一个 ETPNav 评测任务；已向其 `torchrun` 主进程发送正常终止信号，任务进程树退出、GPU 释放，`gwl-etpnav` 容器未停止。该授权只适用于这个具体任务，后续仍默认禁止停止 ETPNav 进程。

2026-07-10，在设计获批后编写 RAE/DINOv2 实施计划。计划位于 `docs/superpowers/plans/2026-07-10-rae-dinov2-visual-encoder.md`，依次覆盖测评机隔离环境、现代 Habitat 兼容、三层投影、冻结编码器、离线预训练、在线 SFT/GRPO、checkpoint 过滤、全量 HDF5、CLIP 回归和最终冒烟。本阶段只形成计划，尚未创建 `gwl-etpr1-rae`、`etpr1_rae` 或开始模型实现。

2026-07-10，为 Task2 现代运行兼容复查。检查了 Python 3.11.15、NumPy 1.26.4、Habitat/Habitat-Sim/Habitat-Baselines 0.3.3、Transformers 4.49.0 的导入链，补齐旧配置与入口兼容，并确认 `run.py`、两个 R1 trainer、`habitat_extensions.task` 和 trainer 注册可用。代码审查后又补充了 R2R/RxR 旧配置到现代 OmegaConf 的桥接、原生配置入口幂等保护、动作编号保护和实际 `R1Env` 配置边界测试；真实 R2R/RxR 转换配置也已覆盖 `pickle` 往返与深复制兼容。测评机最终运行 Task1+Task2 共 34 项测试通过，`python run.py --help` 退出码为 0。
