# ETP-R1 预训练与零 SFT 导航排查汇总

更新日期：2026-08-09

## 1. 排查目的

本轮排查围绕以下现象展开：

- 本工程使用零 SFT 预训练基座时，R2R 导航成功率约为 3%。
- 早期 SFT 约 200 次迭代后，本工程提升只有不到 1 个百分点。
- 用户提供的 ETPNav 对照结果为：零 SFT 成功率约十几%，约 200 次迭代后超过 30%。
- 因此需要确认本工程的联合预训练、DINO 编码器、特征路径、模型加载和导航评测链路是否存在错误。

本文汇总目前已经完成的代码审计、数据验证、训练稳定性测试和导航实验。不同机器、不同代码链路以及不同证据来源会分开记录，避免将不能直接比较的结果混为一谈。

## 2. 核心结论

目前已经确认：

1. ETP-R1 官方设计中，R2R 和 RxR 共用一个联合预训练基座，之后再分别进行 SFT/RFT，并非两个任务各自独立预训练。
2. 官方离线预训练视觉数据采用每个 viewpoint 36 个视角：水平 12 个方向、相邻 30°，俯仰为 `-30°/0°/+30°`，每个单视角的水平和垂直视场角均为 60°。
3. DINO 本地模型目录、权重内容、编码器输出和全量 HDF5 特征均已通过一致性与完整性检查，没有发现“模型路径指错”或“加载了错误 DINO 权重”的证据。
4. 旧链路 452500 步 checkpoint 包含经过预训练的完整视觉投影层。
5. 当前新 ETPNav 风格接口使用的转换 checkpoint 只转移了非视觉权重。新建的 `img_linear 768→768` 是确定性随机初始化，未接受任何预训练更新。这是目前最明确的接口与权重断层。
6. 因此，新接口的零 SFT 2.50% 和早期 SFT 结果不能代表“完整训练了 452500 步的新视觉接口基座”的能力。
7. 训练机上旧完整链路、RGB 60°、Depth 60°的正式零 SFT 结果为 SR 0.3263%。但该实验同时把只适配 90°深度输入的路径点预测器改成了 60°，存在明显混杂因素，不能据此单独判定 DINO 编码器在 60°下有问题。

当前最值得优先验证的是：在训练机保持旧完整链路和同一 checkpoint，补齐 `RGB 90°/Depth 90°` 与 `RGB 60°/Depth 90°` 两组全量实验，再与已有的 `RGB 60°/Depth 60°` 结果比较。

## 3. R2R 与 RxR 的联合预训练关系

### 3.1 官方代码和提交记录

ETP-R1 的 README 明确写有：联合 R2R 和 RxR 数据进行预训练，即 “unifying data from both R2R and RxR tasks for joint pretraining”。

在早期官方提交 `69139a8` 中，`mix_pretrain_server.json` 已同时包含：

- R2R train
- Prevalent
- Prevalent Gemini Aug
- RxR train
- RxR-Marky

五类训练数据合计为 3,210,737 条。联合验证配置同时包含 R2R unseen 和 RxR unseen。

因此，官方流程是：

```text
R2R + RxR 多来源数据
        ↓
共享联合预训练基座
        ↓
分别进行 R2R / RxR 的 SFT 或 RFT
```

### 3.2 离线预训练视角

早期官方提交 `69139a8` 的 `save_img.py` 已使用以下设置：

- 每个 viewpoint 共 36 个视角。
- 水平 12 个方向，相邻方向间隔 30°。
- 三层俯仰：`-30°`、`0°`、`+30°`。
- 单个视角的 HFOV 和 VFOV 均为 60°。

这不是后续修改造成的视场角变化，而是官方早期版本已经采用的设计。

### 3.3 在线导航传感器视场角

官方在线传感器设置与离线预训练并不完全相同：

| 任务 | RGB HFOV | Depth HFOV |
|---|---:|---:|
| R2R | 90° | 90° |
| RxR | 63° | 63° |

所以“离线预训练单视角 60°、在线 R2R 传感器 90°”本身是官方已有设计，不能直接当成当前工程的配置笔误。

## 4. DINO 编码器和视觉链路审计

### 4.1 模型路径与数值一致性

当前 DINO 模型目录为：

```text
pretrained/rae_dinov2_with_registers_base
```

已记录的文件摘要为：

```text
DINO 权重 SHA256:
7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362

stat.pt SHA256:
84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59
```

本工程编码器与 ETPNav 实际编码器的同输入对照结果：

```text
max_abs = 0
cosine = 0.9999999404
autocast_max_abs = 0
```

这说明当前本地路径加载到的 DINO 编码器和参考工程一致，没有发现路径指向其他模型或权重内容不一致的问题。

### 4.2 全量特征文件检查

全量 DINO HDF5 共包含 10,567 个 viewpoint，每个特征的形状和类型均为：

```text
[36, 768], float32
```

以下异常项检查结果均为 0：

- 缺失 viewpoint
- 额外 viewpoint
- 错误形状或数据类型
- 非有限值
- 全零特征

因此目前没有发现全量特征生成遗漏、维度错误或损坏。

### 4.3 旧视觉链路

旧链路对应提交 `7431266`，数据流为：

```text
DINO CLS 768
→ 使用 stat.pt 归一化
→ 训练过的三层投影 768→768→768→512
→ 训练过的 img_linear 512→768
```

旧链路正式预训练 checkpoint：

```text
pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/best/model_best_step_452500.pt
```

摘要：

```text
ac944a51c4fe3177c45f273fc0ba112d276511c7712f1d17ec09047f2122ccff
```

其中视觉层实际形状为：

```text
rgb_projection.0.weight  (768, 768)
rgb_projection.2.weight  (768, 768)
rgb_projection.4.weight  (512, 768)
img_linear.weight        (768, 512)
```

这些视觉参数包含在 452500 步预训练结果中。

### 4.4 新 ETPNav 风格视觉链路

新链路由提交 `711be1e` 引入，数据流改为：

```text
原始 DINO CLS 768
→ 不使用 stat.pt
→ ETPNav 风格 CLS residual MLP
→ img_linear 768→768
```

当前转换产物为：

```text
pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_legacy_base_transfer/
model_step_452500_nonvisual_transfer.pt
```

摘要：

```text
83220a4853202d6731633ec19637568ad121532e4dd8e5e7cb096b18476a6577
```

转换清单已经明确记录：

- 删除旧 `rgb_projection` 的 6 个参数。
- 将旧 `img_linear 512→768` 替换为新的 `img_linear 768→768`。
- 新 `img_linear` 来自预训练 smoke test 前捕获的确定性随机初始化。
- `provenance` 明确标注该层 “has received no pretraining update”。
- 只转移了非视觉权重，没有重新进行联合预训练。

训练机当前也不存在以下完整新接口 checkpoint：

```text
pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768/best/
model_best_step_452500.pt
```

因此，把 `model_step_452500_nonvisual_transfer.pt` 称为“452500 步新接口完整预训练模型”是不准确的。它只继承了旧模型的非视觉部分，新视觉桥接层仍是未训练状态。

## 5. 已完成的正确性和工程稳定性测试

### 5.1 编码器、数据与基础 smoke test

2026-07-15 完成：

- 自动测试：`269 passed, 3 warnings`。
- 15 阶段 smoke test 全部通过。
- RAE/DINO 编码器数值一致性通过。
- 全量 HDF5 特征完整性通过。

### 5.2 完整数据 batch 和显存

| 设置 | 峰值显存 | 结果 |
|---|---:|---|
| batch 32 | 23684 MiB | 通过，但余量过小 |
| batch 16 | 17692 MiB | 通过 |
| batch 16，梯度累积 8 | 17842 MiB | 通过，有效 batch 128，推荐设置 |

### 5.3 断点续训

已进行真实的新进程恢复验证：

- 第 1 步保存 checkpoint。
- 退出后由新进程恢复，并完成第 2 步。
- 成功恢复 484 组优化器状态。
- 对应自动测试更新为 `276 passed, 3 warnings`。

### 5.4 最佳 checkpoint 选择

最佳 checkpoint 选择逻辑和硬链接均已验证，对应自动测试为：

```text
280 passed, 3 warnings
```

### 5.5 数据加载稳定性

预训练曾在第 11195、80204、117624 步发生本地内存层崩溃。现场未发现：

- CUDA 显存不足
- NVIDIA Xid 错误
- 容器重启

稳定性设置随后改为：

```text
n_workers = 0
pin_mem = false
使用线程预取
```

恢复后从 117500 步推进到 117703 步，未再次出现同类异常。对应自动测试为：

```text
290 passed, 3 warnings
```

### 5.6 SFT 启动前验收

已完成：

- 正式权重结构审计。
- 8 环境运行验证。
- 梯度累积验证。
- 断点恢复验证。
- 自动测试：`305 passed, 3 warnings`。

这些结果说明基本训练工程链路可运行，但不能消除第 4.4 节中“新视觉接口没有预训练权重”的问题。

## 6. 导航实验汇总

### 6.1 训练机正式实验：新接口、非视觉转移 checkpoint、零 SFT

模型：

```text
rae_dinov2_etpnav_cls_768_legacy_base_transfer/
model_step_452500_nonvisual_transfer.pt
```

重要说明：该模型的新 `img_linear 768→768` 没有经过预训练。

R2R `val_unseen`，共 1839 条：

| 指标 | 结果 |
|---|---:|
| 成功数 | 46/1839 |
| SR | 2.5014% |
| SPL | 2.3125% |
| Oracle SR | 2.8820% |
| nDTW | 33.4157% |
| sDTW | 1.8261% |
| 平均终点距离 | 9.089995 |

R2R `val_seen`，共 778 条：

| 指标 | 结果 |
|---|---:|
| 成功数 | 35/778 |
| SR | 4.4987% |
| SPL | 4.2092% |
| Oracle SR | 5.1414% |
| nDTW | 32.2947% |
| sDTW | 3.3841% |

结果文件：

```text
data/logs/rae_dinov2_etpnav_cls_768/
r2r_zero_sft_pretrain452500_20260809/full/val_unseen/results/
rae_dino_zero_sft_pretrain452500_val_unseen_20260809/
eval_results/stats_ckpt_0_val_unseen.json
```

### 6.2 训练机正式实验：新接口早期 SFT

该实验同样由上述非视觉转移 checkpoint 初始化。源码日志记录的提交为 `db3a99e`。

R2R `val_unseen`，共 1839 条：

| checkpoint | 成功数 | SR | SPL | Oracle SR | nDTW | sDTW |
|---|---:|---:|---:|---:|---:|---:|
| 零 SFT | 46 | 2.5014% | 2.3125% | 2.8820% | 33.4157% | 1.8261% |
| iter 200 | 60 | 3.2626% | 2.1352% | 6.6340% | 27.1872% | 1.9936% |
| iter 400 | 88 | 4.7852% | 3.3626% | 8.1566% | 27.9359% | 3.1247% |

相对零 SFT：

- iter 200：SR 增加 0.7612 个百分点，SPL 下降 0.1773 个百分点。
- iter 400：SR 增加 2.2838 个百分点，SPL 增加 1.0501 个百分点。

这组结果与“视觉接口层需要从随机初始化开始学习”的现状一致，不能用来判断完整的新接口联合预训练是否有效。

结果目录：

```text
data/logs/rae_dinov2_etpnav_cls_768/
r2r_sft_legacy_pretrain_transfer_15k_noaccum_20260807/
eval_ckpt_200_400_val_unseen_env2_20260807/
```

### 6.3 测评机探索实验：旧完整链路 452500，RGB 90°/Depth 90°

该实验只加载预训练权重，未加载 SFT 权重。

| 环境数 | 成功数 | SR | SPL | Oracle SR | nDTW |
|---:|---:|---:|---:|---:|---:|
| 8 | 30/1839 | 1.6313% | 1.5342% | 2.0663% | 32.9122% |
| 4 | 31/1839 | 1.6857% | 1.5750% | 2.1207% | 32.9279% |

结果目录：

```text
/home/a6000/gwl/ETP-R1/data/logs/rae_dinov2/
r2r_zero_sft_legacy452500_20260809/
```

这组结果来自测评机，只作为探索性参考。用户已明确要求正式实验使用训练机，因此不能把它作为最终训练机结论。

### 6.4 训练机正式实验：旧完整链路 452500，RGB 60°/Depth 60°

运行信息：

```text
机器：训练机 gwl-sever
工程：/home/gwl/project/etpr1/ETP-R1
隔离 worktree 提交：74312665f2e4b7a2a3352feec2e1c88125ba3c9d
Python：3.10.14
PyTorch：2.2.2+cu121
CUDA：12.1
Transformers：4.49.0
GPU：RTX A6000
环境数：8
RGB HFOV：60°
Depth HFOV：60°
episode：1839
退出码：0
```

全量结果：

| 指标 | 结果 |
|---|---:|
| 成功数 | 6/1839 |
| SR | 0.3263% |
| SPL | 0.3041% |
| Oracle 成功数 | 13/1839 |
| Oracle SR | 0.7069% |
| nDTW | 32.6993% |
| sDTW | 0.2220% |
| 平均终点距离 | 9.070411 |
| 平均路径长 | 2.792257 |
| 平均步数 | 44.374660 |

评测完成后已经按 episode 独立复算，结果一致。

结果目录：

```text
/home/gwl/project/etpr1/ETP-R1/data/logs/rae_dinov2/
r2r_zero_sft_legacy452500_hfov60_server_20260809/
full_rgb60_depth60_env8/
```

### 6.5 RGB 60°/Depth 90°隔离实验

该设置的目的，是只让 RGB 视场角与离线预训练的 60°视角一致，同时让 Depth 保持路径点预测器所适配的 90°输入分布。

单 episode smoke test 已通过，并且与 90°/90°对应样本的轨迹完全一致：

```text
steps_taken = 19
distance_to_goal = 14.648360
path_length = 0.212065
ghost_cnt = 9
```

测评机全量尝试运行到 1617/1839 时，一个 Habitat 子进程退出，主进程收到 EOF。该次运行没有生成最终 JSON，因此不能报告为有效全量结果。

训练机目前尚未完成 RGB 60°/Depth 90°的全量正式实验。

### 6.6 60°/60°实验的混杂因素

R2R 路径点预测器固定加载：

```text
data/wp_pred/check_cwp_bestdist_hfov90
```

该预测器的 RGB 分支已经注释，实际主要使用深度特征。因此，把 Depth 从 90°改成 60°会改变路径点预测器的输入分布。

训练机 60°/60°的 SR 0.3263% 同时包含两个变化：

1. DINO RGB 输入由在线默认 90°改成 60°。
2. 路径点预测器的 Depth 输入由其训练时适配的 90°改成 60°。

所以，这个结果不能单独证明“DINO 在 60°视场角下更差”。必须补齐 RGB 60°/Depth 90°实验，才能较好地隔离 RGB 视场角的影响。

### 6.7 用户提供的 ETPNav 对照数字

用户提供的问题背景为：

- ETPNav 零 SFT 成功率约为十几%。
- ETPNav 约 200 次迭代后成功率超过 30%。

当前尚未在受保护的 ETPNav 工程目录中找到与这两个数字对应的正式 JSON，因而暂时不能确认其具体 checkpoint、数据划分、视场角、环境数、停止策略和评测代码。

这两个数字应视为待统一设置复测的对照，不与上述有 JSON 佐证的实验视为同等级证据。

## 7. 实验矩阵总览

| 编号 | 机器 | 视觉链路 | checkpoint 语义 | RGB/Depth HFOV | 数据集 | SR | 状态 |
|---|---|---|---|---|---|---:|---|
| A | 训练机 | 新接口 | 只转移非视觉权重，视觉层未预训练 | 当前新接口配置 | R2R unseen | 2.5014% | 正式完成 |
| B | 训练机 | 新接口 + SFT 200 | 同上 | 当前新接口配置 | R2R unseen | 3.2626% | 正式完成 |
| C | 训练机 | 新接口 + SFT 400 | 同上 | 当前新接口配置 | R2R unseen | 4.7852% | 正式完成 |
| D | 测评机 | 旧完整链路 | 完整 452500 步旧视觉投影 | 90°/90° | R2R unseen | 1.63%～1.69% | 探索性完成 |
| E | 训练机 | 旧完整链路 | 完整 452500 步旧视觉投影 | 60°/60° | R2R unseen | 0.3263% | 正式完成，但有 Depth 混杂 |
| F | 测评机 | 旧完整链路 | 完整 452500 步旧视觉投影 | 60°/90° | R2R unseen | 无最终值 | 1617/1839 失败，无 JSON |
| G | 训练机 | 旧完整链路 | 完整 452500 步旧视觉投影 | 60°/90° | R2R unseen | 待测 | 优先补齐 |
| H | 训练机 | 旧完整链路 | 完整 452500 步旧视觉投影 | 90°/90° | R2R unseen | 待测 | 优先补齐 |

## 8. 失败或不能采用为正式结论的实验

以下结果需要明确排除或限制解释范围：

1. 新接口的非视觉转移 checkpoint 不是完整的 452500 步新接口预训练模型，不能用它排除随机初始化视觉层的影响。
2. 测评机 60°/90°运行没有最终 JSON，不能从中估算或外推全量 SR。
3. 测评机 90°/90°结果只能作为探索性参考，正式结论需要在训练机复现。
4. 训练机 60°/60°结果受 90°路径点预测器与 60°深度输入失配影响，不能只归因于 DINO RGB 编码器。
5. 用户提供的 ETPNav 数字目前缺少同设置评测文件，不能直接与本工程结果做严格差值比较。

## 9. 当前结论边界

### 9.1 已经可以确认

- DINO 模型目录和实际加载权重没有发现错误。
- DINO 编码结果与 ETPNav 参考编码器数值一致。
- 全量离线特征的数量、形状和数值完整性没有发现异常。
- R2R 与 RxR 在官方工程中共享联合预训练基座。
- 官方离线预训练采用 36 视角、水平间隔 30°、三层俯仰、单视角 60°视场角。
- 旧 452500 步 checkpoint 具有训练过的旧视觉投影层。
- 新接口转换 checkpoint 的新视觉桥接层未经过预训练。
- 训练机旧链路 60°/60°零 SFT 的正式 SR 为 0.3263%。

### 9.2 目前还不能确认

- 不能仅由 60°/60°实验断言 DINO 编码器存在 bug。
- 不能把新接口零 SFT 的 2.5014%当成完整 452500 步视觉预训练效果。
- 尚未得到训练机上 90°/90°与 RGB 60°/Depth 90°的同环境全量对照。
- 尚未在统一代码、checkpoint、传感器设置和 episode 顺序下复现 ETPNav 的十几%与 30%+。
- 尚未直接验证离线 HDF5 图像与 Habitat 在线渲染图像在同一位置、朝向下产生的 DINO CLS 是否一致。

## 10. 下一步实验优先级

### 第一优先级：训练机补齐视场角对照

保持以下条件完全一致：

- 旧完整视觉链路提交。
- `model_best_step_452500.pt`。
- 同一训练机环境。
- 相同环境数和 episode 顺序。
- 相同停止策略和评测脚本。

补齐：

1. RGB 90°/Depth 90°。
2. RGB 60°/Depth 90°。
3. 与已有 RGB 60°/Depth 60°结果统一比较。

这三组可以分别观察在线默认设置、只改变 DINO RGB 输入以及同时改变 RGB/Depth 的影响。

### 第二优先级：离线与在线 DINO 特征逐姿态对照

对同一个 MP3D viewpoint、相同朝向和俯仰，分别取得：

- 离线 HDF5 中的原始 DINO CLS。
- Habitat 在线渲染图像重新编码得到的 DINO CLS。

比较余弦相似度和最大绝对误差。该实验可以检查相机姿态、图像预处理、RGB 通道顺序、尺寸缩放和视场角是否造成离线/在线特征错位。

### 第三优先级：真正训练新接口预训练基座

新接口必须从联合预训练阶段开始训练其 `img_linear 768→768` 和相关视觉层。不能继续把非视觉转移 checkpoint 当作完整的新接口预训练基座。

### 第四优先级：ETPNav 严格同设置复测

对照时至少固定：

- checkpoint 的真实语义。
- R2R 数据划分。
- RGB 和 Depth HFOV。
- 路径点预测器。
- 是否加载 SFT 权重。
- 环境数、episode 顺序和停止规则。

只有这些条件一致，才能判断成功率差距来自模型结构、预训练数据、权重加载还是评测链路。

## 11. 相关详细文档

- [ETP-R1 与 ETPNav 差异分析](./ETP-R1_vs_ETPNav_diff_analysis.md)
- [checkpoint 评测对比](./ETP-R1_checkpoint_eval_comparison.md)
- [RAE/DINOv2 测评机验收记录](./rae-dinov2-eval-host-validation.md)
- [RAE/DINOv2 预训练操作记录](./rae-dinov2-pretrain-operations.md)
- [RAE/DINOv2 R2R SFT 操作记录](./rae-dinov2-r2r-sft-operations.md)
