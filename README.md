# PILOT

**面向连续环境视觉语言导航的预测增强拓扑规划。**

PILOT 让智能体根据自然语言指令，在三维环境中逐步选择路点并到达目标。项目基于 [ETP-R1](https://github.com/Cepillar/ETP-R1) 的导航策略，引入共享的冻结世界模型：先预测候选位置的视觉特征来增强地图表示，再利用更远位置的预测辅助比较候选动作。

本仓库包含导航基座预训练与监督训练、候选状态增强、第二阶段离线数据采集和评分头训练，以及在线导航对照评测。当前集成分支为 **`feature/e24-joint-sft`**，已合入第二阶段开发代码。

## 方法概览

智能体维护一张在线拓扑图，其中包含已访问位置和尚未访问的候选位置。候选位置在代码中称为 `ghost`，是可供导航选择的空间位置。

```text
语言指令 + 当前全景观测
          │
          ▼
   视觉编码与在线拓扑图
          │
          ▼
第一阶段：预测候选位置的到达特征
          │ 与已有观测表示拼接，学习残差并维护候选状态
          ▼
     基础导航策略评分
          │ 选取少量移动候选
          ▼
第二阶段：从预测视图提出后继位置，并预测后继视觉特征
          │ 结合指令和候选信息，计算有界评分修正
          ▼
    选择移动目标或停止
```

### 第一阶段：候选表示增强

- 使用 RAE/DINOv2 视觉特征及世界模型预测能力。
- 根据候选目标与历史位姿选择全景上下文，预测候选到达后的视觉表示。
- 将观测表示与预测表示拼接，通过残差网络更新候选特征。
- 支持跨导航步保留候选融合状态，而不是每一步丢弃已有增强结果。

主要实现位于 [`ghost_concat_fusion.py`](vlnce_baselines/nwm/ghost_concat_fusion.py)、[`panorama_runtime.py`](vlnce_baselines/nwm/panorama_runtime.py) 和 [`ss_trainer_ETP_R1.py`](vlnce_baselines/ss_trainer_ETP_R1.py)。

### 第二阶段：利用后继预测修正候选评分

- 固定第一阶段导航模型，采集按路线组织的预测特征与监督标签。
- 从候选到达预测中提出后继路点，再调用世界模型生成后继特征；不使用未来真实图像作为在线输入。
- 离线训练 E24 评分头，将语言、候选表示、基础分数与后继特征结合，学习候选分数修正。
- 在线评分不依赖教师标签，并保留基础停止决策；当前在线实现将最终修正限制在 `[-1, 1]`。
- 提供零修正一致性检查、权重冻结审计，以及包含／移除未来信息的对照实验。

采集、数据读取、训练和在线接入分别位于 [`stage2_collect.py`](vlnce_baselines/nwm/active_lookahead/stage2_collect.py)、[`stage2_data.py`](vlnce_baselines/nwm/active_lookahead/stage2_data.py)、[`stage2_training.py`](vlnce_baselines/nwm/active_lookahead/stage2_training.py) 和 [`stage2_online.py`](vlnce_baselines/nwm/active_lookahead/stage2_online.py)。

## 当前实现范围

| 能力 | 当前状态 |
|---|---|
| R2R / RxR 导航基座 | 包含预训练、监督训练与评测入口 |
| 候选预测融合与持久状态 | 已接入导航训练和评测 |
| 二阶段预测数据采集 | 包含数据来源记录、完整性校验和冻结检查 |
| 二阶段离线训练 | 支持检查点保存、恢复及开发集评价 |
| 二阶段在线导航 | 已接入 R2R 对照评测及零修正检查 |
| RxR 性能优化 | 包含独立快速配置与完整训练状态恢复 |
| GRPO 强化训练 | 保留基座实现；当前候选拼接融合配置不支持 GRPO |
| 多步递归前瞻 | 保留在独立开发分支，未并入当前集成代码 |

已有二阶段在线实验主要针对 R2R，不能据此宣称完整 PILOT 已在 RxR 上验证。实验设置、结果和局限记录在对应文档中，本页不将原版 ETP-R1 的论文成绩作为 PILOT 的结果。

## 代码结构

```text
run.py                         导航训练、评测与推理入口
run_r2r/                       R2R 配置与运行脚本
run_rxr/                       RxR 配置与运行脚本
pretrain_src/                  导航基座预训练
precompute_img_features/       离线视觉特征生成
habitat_extensions/           导航任务与模拟环境扩展
vlnce_baselines/
  models/                      导航策略、视觉编码器与拓扑图
  nwm/                         世界模型接入、全景上下文与候选融合
    active_lookahead/           后继查询、E24 评分及二阶段流程
scripts/                       采集、训练、检查、报告和任务管理工具
tests/                         单元测试与集成检查
docs/                          设计、实施记录和实验报告
```

## 环境与数据

当前代码使用现代 Habitat 兼容层。已验证的项目环境包括：

| 组件 | 验证版本 |
|---|---|
| Python | 3.10.14（训练环境）、3.11.15（测评环境） |
| PyTorch | 2.2.2 + CUDA 12.1 |
| Transformers | 4.49.0 |
| Habitat-Lab / Habitat-Sim | 0.3.3 / 0.3.3 |

根目录 `environment.yaml` 保留了上游历史环境定义，不能直接视为当前完整方案的一键安装配置。运行脚本依赖项目自有 Habitat 目录、环境路径和外部资产；迁移到新机器时需要先调整这些路径。现有环境入口为 [`rgb_only_optimization_runtime.sh`](scripts/rgb_only_optimization_runtime.sh) 和 [`etpr1_rae_runtime_exec.sh`](scripts/etpr1_rae_runtime_exec.sh)。

运行前需另行准备：

- Matterport3D 场景、R2R-CE / RxR-CE 数据及导航连接信息。
- 文本编码器、视觉编码器和路点预测器权重。
- 导航基座检查点；使用预测增强时还需世界模型、统计文件及对应预测模块权重。
- 进行二阶段实验时所需的第一阶段模型、预测路点模块及采集数据。

数据、模型权重、运行环境和日志不随 Git 仓库分发。常用本地目录包括 `data/`、`pretrained/`、`.runtime/` 和 `stage2_assets/`；论文草稿与绘图资产也在本地单独维护。

## 使用入口

### 获取当前代码

```bash
git clone --branch feature/e24-joint-sft https://github.com/guowenlong1234/PILOT.git
cd PILOT
```

私有仓库需要有访问权限的 GitHub 账号。

### 导航基座与候选增强

统一导航入口为 `run.py`，支持 `dagger`（监督训练）、`grpo`、`eval` 和 `inference`。具体运行必须使用与模型、数据及环境匹配的配置。

| 用途 | 入口或配置 |
|---|---|
| 持久候选状态训练 | [`iter_train_rae_dino_ghost_concat_persistent.yaml`](run_r2r/iter_train_rae_dino_ghost_concat_persistent.yaml) |
| RxR 快速训练配置 | [`iter_train_rae_dino_sft_fast.yaml`](run_rxr/iter_train_rae_dino_sft_fast.yaml) |
| RxR 训练管理 | [`manage_rae_rxr_sft_fast_server.sh`](scripts/manage_rae_rxr_sft_fast_server.sh) |
| RxR 完整测评 | [`run_rxr_full_eval_server.sh`](scripts/run_rxr_full_eval_server.sh) |

### 二阶段流程

按“采集与校验 → 离线训练 → 开发集评价 → 在线对照”的顺序执行：

| 步骤 | 脚本 |
|---|---|
| 采集、校验与前向检查 | [`run_stage2_collection.py`](scripts/run_stage2_collection.py) |
| 离线训练评分头 | [`train_stage2_e24.py`](scripts/train_stage2_e24.py) |
| 离线开发集评价 | [`evaluate_stage2_e24.py`](scripts/evaluate_stage2_e24.py) |
| 离线／在线评分重放检查 | [`check_stage2_online_replay.py`](scripts/check_stage2_online_replay.py) |
| 基线、零修正与评分头导航对照 | [`run_stage2_navigation_comparison.py`](scripts/run_stage2_navigation_comparison.py) |
| 9200 基座未来信息对照准备 | [`run_stage2_9200_preparation.py`](scripts/run_stage2_9200_preparation.py) |

参数、资产要求和完整命令见下方实施文档。新实验应使用独立输出目录，恢复训练时需同时核对模型、优化器、调度器、随机状态和数据来源。

## 验证与实验文档

最近一次主线与二阶段合并的 CPU 回归结果为 **165 项通过、1 项跳过**，包含二阶段数据和评分逻辑、全景上下文、候选融合以及部分 RxR 恢复逻辑。这不替代真实模拟环境和完整导航测评。

| 文档 | 内容 |
|---|---|
| [主线合并验证](docs/main-stage2-merge-20260921.md) | 合并范围、实际环境及测试结果 |
| [二阶段数据采集](docs/stage2-offline-collection-operations-20260916.md) | 采集流程、数据约束和验收 |
| [二阶段离线训练](docs/stage2-offline-training-operations-20260917.md) | 训练、恢复和开发集评价 |
| [二阶段在线导航](docs/stage2-online-navigation-20260920.md) | 在线接入、零修正检查和成对测评 |
| [9200 基座未来信息对照](docs/stage2-9200-future-ablation-20260921.md) | 后续采集与对照设置 |
| [RxR 完整训练状态恢复](docs/rxr-fast-resume-20260921.md) | 快速配置与恢复边界 |
| [工程研究索引](research.md) | 模块说明、历史记录和进一步阅读入口 |

历史文档保留当时的分支、机器路径和运行状态；具体实验以对应提交及其配置、数据来源记录为准。

## 致谢

本项目基于 ETP-R1，并沿用或集成 ETPNav、RAE-NWM、DINOv2 和 Habitat 生态中的相关实现。上游方法、数据和依赖的来源说明及引用应在使用时保留。PILOT 的新增实现与实验结论以本仓库代码和配套记录为准。
