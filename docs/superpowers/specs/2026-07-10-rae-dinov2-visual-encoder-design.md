---
title: ETP-R1 RAE/DINOv2 视觉编码器替换设计
status: approved
owner: codex
scope: 保留 CLIP 基线，新增 RAE/DINOv2-B CLS + 三层 MLP 视觉分支
last_verified_date: 2026-07-10
---

# ETP-R1 RAE/DINOv2 视觉编码器替换设计

## 1. 目标

在不删除现有 CLIP 方案的前提下，为 ETP-R1 新增一条与本地 RAE-NWM 一致的 RAE/DINOv2-B 视觉链路。

新链路使用 DINOv2-with-registers-base 的整图 CLS 向量。原始 CLS 为 768 维，经过三层可学习 MLP 投影到 512 维，再进入现有 ETP-R1 视觉、深度、全景、拓扑图和 DPFT 流程。

已经确认的训练约束：

- DINOv2 编码器始终冻结。
- 三层 MLP 在联合预训练和在线 SFT 阶段参与训练。
- 三层 MLP 在 GRPO 阶段冻结。
- 原有 CLIP 配置、特征、checkpoint 和运行入口全部保留。
- RAE/DINOv2 使用独立配置、特征文件、环境和输出目录。
- 环境搭建、依赖安装、特征生成、全部测试和全部实验只允许在测评机执行；本机只用于代码编辑、分析和文档整理。

## 2. 不在本次范围内的内容

- 不使用 DINOv2 的 256 个 patch token 作为 ETP-R1 图节点输入。
- 不修改深度编码器和深度 HDF5。
- 不修改路点预测器的数学逻辑或权重。
- 不改变 MLM、SAP、DAgger 和 GRPO 的任务定义。
- 不在第一版中引入半精度 DINO 在线编码。
- 不尝试继续复用 CLIP 预训练得到的视觉投影权重。

## 3. 当前工程事实

当前 ETP-R1 的 RGB 链路有两个入口：

- 离线联合预训练读取 `CLIP-ViT-B-32-views-habitat.hdf5`，每个视角是 512 维。
- 在线 SFT、GRPO 和评测由 `R1Policy.py` 中的 CLIP 编码器实时产生 512 维特征。

当前路点预测器 `BinaryDistPredictor_TRM` 的 RGB 分支已被注释，实际只使用深度特征。因此更换 RGB 编码器不需要修改或重训路点预测器。

本机 `etpnav` 环境是 Python 3.7、PyTorch 1.9.1、Transformers 4.12.5，不包含 `Dinov2WithRegistersModel`，只保留为旧 CLIP 链路的只读参考，不承担本次实现和验证。

2026-07-10 已只读核验测评机：有线地址为 `10.10.10.2`，GPU 为 RTX 4090 24GB，根分区可用约 335GB。远端 `raenwm` 环境是 Python 3.11.15、PyTorch 2.2.2+cu121、Transformers 4.49.0，包含 `Dinov2WithRegistersModel`。现有 `gwl-etpnav` 容器中仍有 ETPNav 评测进程使用该环境，不能修改或中断。

## 4. 总体方案选择

采用“测评机独立 Docker 容器 + 独立 conda 环境 + ETP-R1 自有 Habitat 依赖目录”的方案。

不采用以下方案：

- 不在本机旧 `etpnav` 环境用 `timm` 近似重建。该方案需要人工映射 Hugging Face 权重、寄存器 token 和非仿射最终层归一化，难以保证与 RAENWM 数值一致。
- 不直接在测评机宿主机运行。宿主机 conda 无法提供与现有 ETPNav Docker 相同的图形、CUDA 和 Habitat 运行边界。
- 不复用现有 `gwl-etpnav` 容器执行本项目。该容器中有正在运行的 ETPNav 评测，且共享进程空间会增加误操作风险。
- 不使用跨进程视觉编码服务。该方案会增加在线导航延迟、多卡通信和异常恢复复杂度。

测评机上的 ETPNav 已经验证过现代 `raenwm` 环境配合 Habitat 0.3.3 本地依赖的运行思路。本设计只把镜像和构建方法作为只读参考，所有新容器、conda 环境、项目目录、依赖目录和输出归当前 ETP-R1 独立所有。

## 5. 环境隔离设计

### 5.1 测评机身份与入口

“测评机”或“4090”固定指：

```text
SSH: ssh 4090
有线地址: 10.10.10.2
用户: a6000
工作根: /home/a6000/gwl
GPU: NVIDIA GeForce RTX 4090 24GB
```

每次远程操作前先执行 `hostname`、`whoami`、`ip -br addr`、`docker ps` 和 `nvidia-smi`，确认没有连到本机或错误容器。

### 5.2 新 Docker 容器

新建独立容器：

```text
gwl-etpr1-rae
```

只读复用现有基础镜像：

```text
gwl-etpnav:etpnav-runtime-20260701185256
```

容器使用 GPU、`ipc=host`、16GB 共享内存，并只绑定测评机工作根：

```text
/home/a6000/gwl -> /home/a6000/gwl
```

容器默认工作目录为 `/home/a6000/gwl/ETP-R1`。不得在现有 `gwl-etpnav` 容器中安装依赖、生成特征或运行本项目测试和实验。

### 5.3 新 conda 环境

新环境固定为：

```text
/home/a6000/gwl/miniconda3/envs/etpr1_rae
```

它在新容器中通过只读克隆远端 `raenwm` 创建。创建后所有安装、卸载和升级都只发生在 `etpr1_rae`，不得修改 `/home/a6000/gwl/miniconda3/envs/raenwm`。

克隆前后保存：

- 远端 `raenwm` 的 `conda list --explicit`
- `pip freeze`
- 所有 `.pth` 文件的 SHA256
- 现有 ETPNav 进程与 GPU 状态

### 5.4 清除 ETPNav 路径绑定

远端 `raenwm` 环境存在 `etpnav-local-deps.pth`，它指向：

```text
/home/a6000/gwl/ETPNav
/home/a6000/gwl/dino_cwp
/home/a6000/gwl/_deps/habitat-lab-v0.3.3
/home/a6000/gwl/_deps/habitat-sim-v0.3.3
```

克隆后只删除新环境中的这份 `.pth`，原环境文件保持不变。新环境不得把 ETPNav 工程目录加入 Python 搜索路径。

### 5.5 ETP-R1 自有运行依赖

建立：

```text
/home/a6000/gwl/ETP-R1/.runtime/etpr1_habitat/
```

这里保存 ETP-R1 在现代 Python 下运行所需的 Habitat-Lab、Habitat Baselines、Habitat-Sim Python 模块、动态库和兼容文件。

具体做法是把 ETPNav 已验证的 Habitat 0.3.3 构建逻辑复制为 ETP-R1 自有脚本，在当前项目目录重新构建或复制后验证。不得软链接、硬链接或直接引用 `/home/a6000/gwl/ETPNav` 的运行目录。

同时把 ETP-R1 的旧接口迁移到现代环境：

- 将 `pytorch_transformers.BertConfig` 改为现代 `transformers.BertConfig`。
- 处理 NumPy 1.26 中已删除的 `np.bool` 等旧别名。
- 移植运行 ETP-R1 所需的最小 Habitat 配置兼容层。
- 保持任务定义和训练算法不变，不做无关的 Habitat 重构。

### 5.6 专用启动脚本

新增：

```text
scripts/etpr1_rae_runtime_exec.sh
```

脚本只对当前子进程设置 `PYTHONPATH`、`LD_LIBRARY_PATH` 和旁路运行标志，不修改宿主机 shell 配置。所有测试和实验都通过新容器、新环境和该脚本启动。

### 5.7 本机与测评机边界

本机工作区 `/home/gwl/project/etpr1/ETP-R1` 只用于阅读、编辑、评审、Git 操作和文档整理。本次任务禁止在本机执行：

- 新环境创建或依赖安装
- 单元测试、集成测试、冒烟测试和性能测试
- DINO/RAE 数值一致性验证
- 全量 HDF5 生成或校验
- 预训练、SFT、GRPO 和评测

代码同步到测评机 `/home/a6000/gwl/ETP-R1` 后，所有验证都在 `gwl-etpr1-rae` 内完成。

### 5.8 GPU 与磁盘约束

设计时测评机根分区可用约 335GB，远端 `raenwm` 环境约 7.4GB，全量 float32 DINO CLS HDF5 原始数据约 1.2GB，空间足够。

测评机只有一张 RTX 4090。启动特征生成、训练或完整评测前必须检查 ETPNav 是否仍占用 GPU；不得终止它，也不得与它并行启动本项目重型任务。新实验使用独立输出目录并限制 checkpoint 保留数量，不为每个 checkpoint 重复保存冻结 DINO 权重。

## 6. RAE/DINOv2 编码语义

### 6.1 本地模型来源

视觉编码器与测评机 `/home/a6000/gwl/RAE-NWM/raenwm` 中的 RAE-NWM 保持一致：

- 架构：DINOv2-with-registers-base
- 隐藏维度：768
- patch size：14
- register token 数量：4
- 输入尺寸：224×224
- DINO 输入归一化：ImageNet mean/std
- RAE latent 统计：使用本地 `stat.pt`
- 输出：RAE 归一化后的 CLS 向量

只把运行所需文件复制到测评机 ETP-R1 自有模型目录 `/home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base`：

- `config.json`
- `preprocessor_config.json`
- `model.safetensors`
- RAE latent `stat.pt`

不复制或加载 RAE 图像 decoder，因为导航只需要编码器。

### 6.2 编码器封装

实现一个 encoder-only 的 `RaeDinov2ClsEncoder`：

1. 把 RGB 转成 `[0,1]`、NCHW、float32。
2. 要求输入为原生 224×224，不做中心裁剪。
3. 在 DINO 内部应用与本地配置一致的 ImageNet mean/std。
4. 读取最后一层输出的第 0 个 token 作为 CLS。
5. 跳过 4 个 register token；patch token 不进入 ETP-R1。
6. 使用 RAE latent mean/variance 对 CLS 做标准化。
7. 返回 float32 `[B,768]`。

DINO 参数在构造完成后统一设置 `requires_grad=False`，并保持 `eval()`。编码过程使用 `torch.no_grad()`。

### 6.3 数值一致性要求

同一张固定 224×224 图片分别经过：

- 本地 RAENWM 原始 RAE 编码链路
- ETP-R1 encoder-only 封装

两者的归一化 CLS 必须形状一致。float32 模式下要求最大绝对误差不超过 `1e-5`，余弦相似度不低于 `0.999999`。通过该校验前不得开始生成全量 HDF5。

## 7. 三层 MLP

三层 MLP 的结构固定为：

```text
Linear(768, 768)
GELU
Linear(768, 768)
GELU
Linear(768, 512)
```

第一版不增加额外 dropout 和残差连接：

- 后续网络已经有环境特征 dropout。
- 输入输出维度不同，不能做直接恒等残差。
- 后续原有 `img_linear(512,768)` 和 LayerNorm 保持不变。

MLP 使用 PyTorch 标准线性层初始化。DINO 分支需要重新进行联合预训练，不从 CLIP 视觉投影权重热启动。

## 8. MLP 的归属和权重传递

MLP 不放在冻结的 DINO 编码器内部，而放在 ETP-R1 的 `img_embeddings` 中，统一命名为：

```text
img_embeddings.rgb_projection
```

离线预训练模型中的键为：

```text
bert.img_embeddings.rgb_projection.*
```

在线模型中的目标键为：

```text
img_embeddings.rgb_projection.*
```

沿用现有预训练模型到在线模型的 Hugging Face base-prefix 加载规则。增加集成测试，给 MLP 写入已知权重后保存预训练 checkpoint，再构建在线模型并确认所有 MLP 参数逐值相等。

### 8.1 离线调用位置

预训练 `ImageEmbeddings.forward()` 先调用 `rgb_projection`：

```text
raw DINO CLS 768
-> rgb_projection 512
-> 原有 img_linear 512->768
-> 原有后续流程
```

### 8.2 在线调用位置

`R1Policy` 的 waypoint 分支从实时编码器获得 raw CLS 768，然后调用：

```text
self.vln_bert.img_embeddings.rgb_projection
```

投影后的 512 维特征再整理成 `cand_rgb` 和 `pano_rgb`。`forward_panorama()` 仍接收 512 维输入，因此不重复应用 MLP。

## 9. 全量离线特征

### 9.1 文件

新增：

```text
pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5
```

### 9.2 覆盖范围

覆盖当前 connectivity 的全部数据：

```text
10,567 个视点
× 36 个离散观察方向
× 768 维
= 380,412 个 CLS 向量
```

每个 HDF5 数据项保持现有接口：

```text
scan_id_viewpoint_id -> [36,768], float32
```

纯数组数据大小为 1,168,625,664 字节，约 1.2GB。HDF5 压缩后的实际文件通常更小。

### 9.3 图像几何

为只改变视觉编码器而不扩大实验变量，离线预训练继续使用 ETP-R1 原有的 36 视角和 VFOV 60° 几何。渲染分辨率改为原生 224×224，传感器高度保持 1.25m。

特征生成脚本直接从 MP3D 场景渲染并编码，不要求长期保留原始 RGB HDF5。需要保证 BGR/RGB 只转换一次。

在线阶段继续使用各任务原有传感器几何：R2R 和 RxR 的 HFOV 不因本次视觉编码器替换而改变。

### 9.4 HDF5 元数据

根属性至少记录：

- `feature_extractor=rae_dinov2_with_registers_base_cls`
- `feature_dim=768`
- `dtype=float32`
- `num_views=36`
- `image_size=224`
- `vfov=60`
- `sensor_height=1.25`
- `latent_normalized=true`
- DINO 权重 SHA256
- RAE stat SHA256
- 预处理版本

### 9.5 数据完整性

全量生成后必须验证：

- key 数恰好为 10,567。
- key 集合与 connectivity 和现有 CLIP HDF5 完全一致。
- 所有 shape 均为 `[36,768]`。
- 不存在 NaN、无穷值或全零向量。
- 抽样重新在线编码时，与 HDF5 对应视角一致。

深度继续使用现有 `ddppo_resnet50_depth_features.hdf5`，预训练标注文件不变。

## 10. 配置设计

保留现有 CLIP 配置，新增 DINO 专用配置：

```text
pretrain_src/run_pt/mix_pretrain_rae_dino.json
pretrain_src/run_pt/mix_model_config_rae_dino.json
run_r2r/iter_train_rae_dino.yaml
run_rxr/iter_train_rae_dino.yaml
```

在线配置接口：

```yaml
MODEL:
  RGB_ENCODER:
    type: rae_dinov2
    model_dir: pretrained/rae_dinov2_with_registers_base
    raw_output_size: 768
    output_size: 512
    projection_hidden_size: 768
```

CLIP 配置继续使用：

```yaml
MODEL:
  RGB_ENCODER:
    type: clip
    output_size: 512
```

### 10.1 预训练尺寸字段

当前数据读取和模型输入共用 `image_feat_size=512`，DINO 分支不能继续复用同一个字段，否则数据读取器会把 768 维 CLS 截断到 512。

新增明确字段：

```text
raw_image_feat_size=768
image_feat_size=512
```

- 数据加载器按 `raw_image_feat_size` 读取 HDF5。
- MLP 把 raw 768 投影成 `image_feat_size=512`。
- 后续 `img_linear` 和所有消费者仍然只看到 512。

## 11. 三阶段训练

### 11.1 联合预训练

数据流：

```text
DINO CLS HDF5 768
-> 可训练三层 MLP 512
-> 原有 img_linear 和视觉深度融合
-> MLM + SAP
```

初始化方式：

- 文本部分继续从本地 XLM-RoBERTa 初始化。
- DINO 特征已经离线生成，不把 DINO 模型放进预训练计算图。
- MLP 和 ETP-R1 后续网络从新实验初始化并训练。
- 不加载原 CLIP 联合预训练 checkpoint。

联合预训练输出目录固定为：

```text
pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/
```

### 11.2 在线 SFT

数据流：

```text
Habitat 224×224 RGB
-> 冻结 RAE/DINOv2
-> raw CLS 768
-> 从预训练 checkpoint 加载的 MLP 512
-> 原有 DAgger SFT
```

DINO 始终冻结，MLP 继续训练。R2R 和 RxR 分别使用独立输出目录。

### 11.3 GRPO

加载对应 SFT checkpoint。沿用当前 GRPO 行为：先冻结整个策略，再只解冻全局图编码器、文本引导图模块、DPFT 相关模块和动作预测头。

MLP 位于 `img_embeddings`，不属于 GRPO 的解冻列表，因此自动保持冻结。

## 12. Checkpoint 设计

### 12.1 不重复保存冻结 DINO

在线 policy 的普通 `state_dict()` 会包含冻结编码器。为避免每个 checkpoint 重复增加约 346MB，保存时过滤 DINO 主干参数。

checkpoint 仍保存：

- 三层 MLP
- ETP-R1 后续网络
- 优化器状态
- 调度器状态
- 训练迭代数
- RGB 编码器类型
- DINO 模型相对路径
- DINO 权重 SHA256
- RAE stat SHA256

加载时先从固定本地目录构建冻结 DINO，再加载导航网络和 MLP 权重。

### 12.2 配置保护

以下情况直接报错，不允许静默继续：

- CLIP checkpoint 加载到 DINO 配置。
- DINO checkpoint 加载到 CLIP 配置。
- raw feature dim 不是 768。
- projection output dim 不是 512。
- DINO 或 RAE stat 哈希不一致。
- 预训练 checkpoint 缺少 MLP 权重。

已知被过滤的 DINO key 不应污染 missing-key 报告；其他 missing 或 unexpected key 继续完整打印。

## 13. 错误处理

- DINO 只允许本地加载，禁止自动联网回退。
- 在线 RGB 不是 224×224 时直接报错，第一版不在编码器内部偷偷裁剪或缩放。
- 编码输出不是 `[B,768]` 时直接报错。
- MLP 输出不是 `[B,512]` 时直接报错。
- 特征包含 NaN 或无穷值时停止生成或训练。
- HDF5 元数据与配置不一致时停止预训练。
- DINO 配置下找不到本地模型或统计文件时停止，不回退到 CLIP。

## 14. 验证计划

本节全部命令只能在测评机 `gwl-etpr1-rae` 容器的 `etpr1_rae` 环境中执行。本机不得用“快速检查”为由代跑任何测试。

### 14.1 单元测试

- RAE/DINO 输入预处理：布局、范围和 224×224 检查。
- encoder-only CLS 与 RAENWM 原始链路数值一致。
- MLP 输入 `[B,768]`、输出 `[B,512]`。
- DINO 参数始终冻结。
- CLIP 默认分支行为不变。

### 14.2 数据测试

- 小规模特征生成支持断点续写。
- 全量 HDF5 key、shape、dtype、有限值和元数据检查。
- 同一图像的离线 HDF5 与在线编码结果一致。

### 14.3 Checkpoint 测试

- 预训练 MLP 权重能够完整映射到在线模型。
- DINO 权重不会写入训练 checkpoint。
- DINO 哈希不匹配时加载失败。
- CLIP/DINO 类型错配时加载失败。

### 14.4 三阶段冒烟

依次通过：

1. 一个 MLM batch。
2. 一个 SAP batch。
3. 一个单环境 SFT 前向和反向。
4. 一次 GRPO 参数冻结检查。
5. 一个 R2R episode。
6. 一个 RxR episode。

训练阶段需要检查：

- 预训练和 SFT 中 MLP 梯度非零。
- DINO 梯度始终为空。
- GRPO 中 MLP `requires_grad=False`。

### 14.5 CLIP 回归

使用原配置和现有 CLIP checkpoint，确认：

- 仍读取原 CLIP HDF5。
- 输出仍为 512 维。
- checkpoint 不出现新的 missing key。
- 原评测命令可以启动。

## 15. 性能验证

DINOv2-B/14 的视觉 token 数多于 CLIP ViT-B/32，在线编码预计更慢。第一版在测评机 RTX 4090 上使用 float32，先保证离线和在线一致。

冒烟阶段记录：

- 12 个方向的 DINO 编码耗时。
- 单环境和多环境显存。
- 每个 SFT iteration 时间。
- GRPO rollout 时间。

如果多环境运行显存不足，只调整 DINO 专用配置的环境数量，不修改 CLIP 配置。半精度优化作为后续独立改动处理。

## 16. 主要文件范围

预计涉及：

- 新 conda 环境与环境核验脚本。
- 测评机独立 Docker 容器约定、ETP-R1 自有 Habitat 依赖和启动脚本。
- 根目录 `AGENTS.md` 中的本机/测评机执行边界。
- RAE/DINO encoder-only 封装。
- RGB 编码器配置工厂。
- `R1Policy.py` 在线 RGB 分支。
- 预训练与在线两套 `ImageEmbeddings`。
- 预训练数据读取尺寸字段。
- 全量 RAE/DINO CLS 特征生成与校验脚本。
- 四份 DINO 专用训练配置。
- SFT/GRPO checkpoint 保存与加载过滤。
- 单元测试、集成测试和冒烟脚本。

不修改原 CLIP HDF5、原 CLIP 配置和已有 checkpoint。

## 17. 完成标准

以下条件全部满足才算替换完成：

- 测评机现有 `gwl-etpnav` 容器、远端 `raenwm` 环境和正在运行的 ETPNav 未被修改或中断。
- 测评机新容器 `gwl-etpr1-rae` 与新环境 `etpr1_rae` 能独立运行 ETP-R1 和远端 RAE/DINOv2。
- 本机未创建新环境、未生成特征、未运行任何测试或实验。
- encoder-only CLS 与 RAENWM 原始链路通过数值一致性测试。
- 全量 10,567 个视点的 DINO CLS HDF5 通过完整性验证。
- MLP 权重能从预训练 checkpoint 正确传到 SFT。
- DINO 在所有阶段冻结。
- MLP 在预训练/SFT 可训练、GRPO 冻结。
- CLIP 基线仍可运行。
- MLM、SAP、SFT、GRPO 和 R2R/RxR 冒烟全部通过。
