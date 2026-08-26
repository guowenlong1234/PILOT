---
title: ETP-R1 原生 CLS RAE-NWM 导航前瞻替换设计
status: approved
owner: codex
scope: R2R SFT 与评测中的 RGB 注入和 Top-5 前瞻
last_reviewed_date: 2026-08-26
---

# ETP-R1 原生 CLS RAE-NWM 导航前瞻替换设计

## 1. 目标

把当前 R2R SFT/评测链路中的两处旧 RAE-NWM 使用统一替换为训练机上的原生 CLS 世界模型：

```text
/home/gwl/project/RAE-NWM/raenwm/logs/
raenwm_mp3d_h125_224_96_e50_fresh_cls_20260617/
checkpoints/checkpoint_step_75000.pth.tar
```

两处接入分别是：

1. 用预测未来 CLS 修正当前 ghost 候选的 RGB CLS 表示。
2. 为 Top-5 前瞻生成 q1 的完整 257 token，并在 E24 前用可训练 adapter 调整第 0 个 CLS token。

新世界模型全程冻结。RGB adapter、Top-5 CLS adapter 和 E24 随 R2R SFT 训练；技术验收以代码正确、测试通过、保存/恢复可用和单 episode 评测可运行作为标准，不要求 SR/SPL 提升，也不启动长训练。

## 2. 已核验事实

### 2.1 指定 checkpoint

2026-08-26 已在训练机只读核验：

- 主机：`gwl-sever`
- 用户：`gwl`
- 文件大小：`5,811,765,519` 字节
- 修改时间：`2026-06-22 12:36:18 +08:00`
- SHA-256：`38b24af13b76ba8faef367559244c3a0401e0557e7c299870c273cbee8a07064`
- checkpoint 顶层键：`model`、`ema`、`opt`、`args`、`epoch`、`train_steps`、`scaler`、`scheduler`
- `epoch=49`
- `train_steps=75000`
- 推理必须使用 `ema`；模型权重键可能带 `_orig_mod.` 前缀，EMA 键不带。
- EMA 含 314 个张量，共 365,318,996 个 float32 元素。

完整文件约 5.8 GB 是因为同时包含 model、EMA 和 AdamW 状态。导航 checkpoint 不复制这些权重，只保存外部资产路径、哈希和推理 provenance。

### 2.2 新模型的真实输入输出

新实验配置为：

- `model: CDiT-B/2`
- `context_size: 4`
- `predict_cls_token: true`
- `image_size: 224`
- `latent_dim: 768`
- `latent_size: 16`
- `head_width: 2048`
- `head_depth: 2`
- `head_num_heads: 16`
- 运动条件为局部 `[dx, dy, dtheta]`
- 时间条件为 `rel_t = horizon / 128`

DINOv2-with-registers-base 每帧产生：

```text
CLS              [B,768]
patch            [B,768,16,16]
packed sequence  [B,257,768]
```

257 个 token 由 1 个 CLS 和 256 个 patch 组成；4 个 register token 不参与 RAE latent。CDiT 对完整序列建模，采样最终结果 `samples_latent[:, 0, :]` 才是原生未来 CLS。

该模型没有独立 token head，也没有 confidence head。当前工程的 `nwm_heads.pt` 不能继续作为新模型的 CLS 或置信度来源。

### 2.3 特征空间边界

新 NWM 的上下文 CLS、上下文 patch、预测 CLS 和预测 patch 都位于 RAE 归一化 latent 空间。当前导航候选 CLS 和原始 `e24_avg3.pth` 使用的是原始 DINO 特征空间。

因此数据边界固定为：

```text
原始 DINO CLS/patch
  → RAE 统计量归一化
  → NWM context / prediction
  → 只对预测 CLS 做反归一化
  → RGB adapter 或 Top-5 CLS adapter
```

CLS 归一化严格复现 RAE-NWM 的 `_stat_for_shape` 规则：空间 mean/var 对 16×16 位置求均值后作用于 `[B,768]`；patch 使用原有逐位置 mean/var。反归一化使用相同统计量和 epsilon。

预测 patch 保持 RAE 归一化空间，继续给 DINO-CWP 和 E24 使用，与当前在线 NWM patch 路径一致。

### 2.4 当前工程的两条旧链路

当前 RGB 注入是：

```text
旧 patch-only NWM
  → 外置 token head 生成 pred_cls
  → 外置 confidence head
  → RaeNwmRgbFusionAdapter
  → ghost 候选视觉表示
```

当前 Top-5 是：

```text
q0 patch-only NWM
  → DINO-CWP
  → q1 patch-only NWM
  → 外置 token head 生成 CLS
  → [CLS + 256 patch]
  → E24
```

现有 E24 输入已经是 `[B,5,257,768]`，所以保留完整 q1 序列能够复用 E24 的结构和 `e24_avg3.pth` 权重。

## 3. 范围

### 3.1 本次包含

- 扩展 ETP-R1 内现有 NWM 兼容层，使其支持原生 CLS+patch 序列。
- R2R SFT trainer 中的 RGB NWM 注入。
- R2R SFT trainer 中的 Top-5 q0/q1 NWM 前瞻。
- 复用 `e24_avg3.pth` 作为 E24 仅权重初始化。
- 新建两个互不共享权重的 adapter。
- 为 E24 修正后的完整导航 logits 增加教师动作交叉熵。
- 新实验配置、资产校验、checkpoint、恢复、诊断、单元测试和真实冒烟。
- 复用同一 SFT trainer 的 R2R 评测路径。

### 3.2 本次不包含

- 不接入 GRPO trainer。
- 不接入 RxR。
- 不修改预训练数据、预训练模型或 HDF5 特征。
- 不训练或微调 NWM、DINOv2、DINO-CWP。
- 不修改 waypoint predictor 的数学逻辑或权重。
- 不要求完整 R2R `val_unseen` 性能提升。
- 不启动正式长训练。
- 不覆盖或删除旧 NWM/E24 实验配置、代码路径和产物。
- 不直接从外部 RAE-NWM 工作区导入运行时代码。

## 4. 方案选择

采用“扩展当前工程内的 NWM 兼容层”。

具体做法是只迁移原生 CLS 推理所需的最小能力：257-token 打包/拆包、序列噪声、序列上下文、CDiT 序列采样、严格 EMA 加载和输出拆分。ETP-R1 的导航几何、上下文缓存、DINO-CWP、E24、checkpoint 和训练器仍由当前工程管理。

不采用以下方案：

- 不在运行时向 `sys.path` 注入 `/home/gwl/project/RAE-NWM/raenwm`。该方案绑定外部工作区和提交状态，测评机难以复现。
- 不整体复制 RAE-NWM 的训练、数据集、decoder 和评测代码。导航只需要推理核心，整体迁移会扩大维护面。
- 不同时运行新 CLS NWM 和旧 patch NWM。双模型会增加计算量，并产生语义不一致的 CLS/patch 组合。

## 5. 总体数据流

### 5.1 在线 DINOv2 输出拆分

现有 `RaeDinov2RgbEncoder` 的一次 DINOv2 前向要显式保留三类结果：

```text
raw_cls       [B,768]          原始冻结 DINO CLS
nav_cls       [B,768]          raw_cls 经现有导航 ClsResidualMlp
raw_patch     [B,768,16,16]    原始冻结 DINO patch
```

使用边界：

- `nav_cls` 继续进入候选 RGB、panorama、graph 和基础导航策略。
- `raw_cls` 与 `raw_patch` 只用于构造 NWM 上下文。
- NWM 上下文不得使用经过 `ClsResidualMlp` 的 `nav_cls`。
- DINOv2 主干仍在 `torch.no_grad()` 中执行并保持 `eval()`。

`R1Policy` 需要向 trainer 返回前向视图对应的 raw CLS 和 raw patch。12 视图全景仍按当前顺序编码，但 NWM 只消费第 0 个正前方 pinhole 视图，保持与模型训练域一致。

### 5.2 上下文缓存

runtime 在收到前向 raw CLS/raw patch 后完成归一化和打包：

```text
normalized_cls      [B,768]
normalized_patch    [B,768,16,16]
packed_frame        [B,257,768]
context             [B,4,257,768]
```

上下文缓存仍记录位置和 yaw，继续使用：

- `context_size=4`
- `max_buffer_size=64`
- 位置/朝向去重阈值
- 静止序列丢弃逻辑
- historical source snapshot
- env pause 时同步删除 buffer

snapshot 的形状契约从 `[4,768,16,16]` 升级为 `[4,257,768]`，并继续保存为 detached CPU float32 副本。

历史不足 4 帧时不重复第一帧，也不制造伪上下文：

- RGB 不融合，使用当前 `nav_cls`。
- Top-5 future 槽无效，不加 E24 残差。
- 最终退回原导航动作。

### 5.3 原生序列采样

新模式固定：

- `predict_cls_token=true`
- `num_steps=10`
- `sampling_method=euler`
- `final_only_euler=false`
- `return_rgb=false`

初始噪声形状为：

```text
[N,257,768]
```

采样输出也为 `[N,257,768]`。runtime 统一生成：

```text
pred_tokens          [N,257,768]       RAE 归一化空间
pred_cls_normalized  [N,768]
pred_cls_raw         [N,768]           反归一化后
pred_patch           [N,768,16,16]     RAE 归一化空间
```

`NwmPrediction.pred_latent` 继续指向 `pred_patch`，减少 DINO-CWP 和旧测试的机械改动；新增明确的 `pred_tokens` 字段。新模式禁止调用 `NwmOutputHeads`。

## 6. RGB 注入设计

### 6.1 输入语义

每个 new/existing ghost 候选使用：

- `raw_rgb`：当前候选方向的 `nav_cls`，`[N,768]`
- `wm_rgb`：新 NWM 的 `pred_cls_raw`，`[N,768]`
- `agreement`：两者余弦一致度映射到 `[0,1]`
- `distance`：候选距离

一致度定义为：

```text
agreement = (cosine_similarity(raw_rgb, wm_rgb) + 1) / 2
```

它只作为 gate 的显式提示，不声明为概率意义上的置信度。

### 6.2 复用当前结构

RGB adapter 只复用现有 `RaeNwmRgbFusionAdapter` 的网络结构，不加载任何旧 adapter 权重：

```text
delta = wm_rgb - raw_rgb

residual:
  Linear(768,768)
  GELU
  Linear(768,768)
  GELU
  Linear(768,768)

gate input:
  raw_rgb[768] + wm_rgb[768] + delta[768]
  + agreement[1] + distance[1] = 2306

gate:
  Linear(2306,768)
  GELU
  Linear(768,1)
  Sigmoid

fused = raw_rgb + alpha * gate * residual(delta)
```

默认 `alpha=1.0`。residual 最后一层权重和偏置初始化为 0；gate 最后一层权重和偏置也初始化为 0。这样初始输出严格等于 `raw_rgb`，同时 gate 初始为 0.5，避免原 `bias=-8` 导致 residual 最后一层梯度过小。

RGB adapter 只修改 new/existing ghost，不修改已访问 node。若同一 ghost 对应多个候选方向，继续复用同一 NWM 预测。

## 7. Top-5 前瞻设计

### 7.1 q0 和 q1

Top-5 候选、persistent q0、historical context 和几何逻辑保持不变：

```text
基础导航 Top-5 ghost
  → historical 4×257 context
  → q0 原生序列 NWM
  → q0 pred_patch
  → 冻结 DINO-CWP
  → q1 position/yaw/horizon
  → 同一 historical context 的 q1 原生序列 NWM
```

DINO-CWP 继续只消费 q0 的 256 个 patch token。q1 不查询模拟器、不渲染 RGB。

### 7.2 Top-5 CLS adapter

q1 的 `pred_cls_raw` 经独立 adapter 调整。该 adapter 不与 RGB adapter 共享权重，也不读取文本或导航 owner embedding，避免与 E24 的职责重叠。

位姿条件展开为：

```text
[dx, dy, sin(dtheta), cos(dtheta), rel_t]  # 5 维
```

其中 `dx`、`dy` 固定使用 NWM 实际接收的归一化局部动作坐标，
即 `record.condition.dx/dy`，不使用米制局部位移。这样 adapter
看到的位姿条件与生成对应 q1 CLS 时的 NWM 条件完全一致。

结构固定为：

```text
cls branch:
  LayerNorm(768)

condition branch:
  Linear(5,128)
  GELU
  Linear(128,128)

fusion:
  concat [768 + 128]
  Linear(896,768)
  GELU
  Linear(768,768)

adapted_cls = pred_cls_raw + delta_cls
```

fusion 最后一层权重和偏置初始化为 0，因此新实验起点为恒等映射。

完整 E24 future 为：

```text
[adapted_cls, pred_patch_token_1, ..., pred_patch_token_256]
```

只有第 0 位被替换，256 个 patch 必须逐值保持不变。future 仍为 `[B,5,257,768]`，可以复用 E24 结构。

### 7.3 E24 初始化

E24 从原始 `e24_avg3.pth` 仅加载模型权重：

- 不恢复旧 optimizer。
- 不恢复旧 scheduler。
- 不恢复旧 RNG。
- 不恢复旧 iteration。
- 不使用旧 NWM 联合 pilot checkpoint 初始化。
- E24 全部参数重新开放训练。

当前 base navigation checkpoint 继续固定为 `iter14200`。

## 8. 损失、replay 与梯度边界

### 8.1 两项导航损失

总损失为：

```text
L_total = L_base_navigation + lambda_adjusted * L_adjusted_navigation
```

- `L_base_navigation`：现有基础导航 logits 对教师动作的交叉熵。
- `L_adjusted_navigation`：E24 修正后的完整导航 logits 对同一教师动作的交叉熵。
- `lambda_adjusted` 默认 1.0，并写入配置和 checkpoint provenance。
- 两项损失都沿用现有 `ignore_index=-100` 和有效教师动作归一化规则。

新实验不使用旧 E24 独立 decision-aware loss；旧实现保留给历史配置，不做无关删除。

### 8.2 调整后完整 logits

replay 保存包含 STOP、全部可执行 ghost 和 padding mask 的完整基础 logits。E24 只为 Top-5 ghost 产生残差：

- STOP 不加残差。
- 非 Top-5 ghost 不加残差。
- Top-5 残差仍受 `delta_max` 约束。
- 教师动作在 Top-5 外时，完整交叉熵仍会对 Top-5 产生压低梯度。
- 教师动作是 STOP 时，完整交叉熵会学习抑制错误的 Top-5 MOVE，但不会直接创造 STOP 残差。

只有至少一个有效 future 槽的 row 参与 adjusted loss。某 rank 没有有效 row 时，使用包含所有 Top-5 adapter/E24 参数的零值 dummy graph 完成 DDP 同步。

### 8.3 延迟 replay

为避免整条 rollout 保存 E24 和 257-token 计算图，保留 CPU replay：

rollout 阶段保存 detached CPU fp16/整数输入：

- 原始 q1 257 token
- q1 condition
- owner embedding
- 文本 token 和 mask
- q0 geometry
- 完整基础 logits 和 mask
- Top-5 到完整 logits 的索引
- 教师动作

replay 阶段重新执行：

```text
raw q1 tokens
  → Top-5 CLS adapter
  → E24
  → adjusted full logits
  → cross entropy
  → backward
```

rollout 选动作时使用 Top-5 adapter+E24 的 eval 前向；action residual 保留当前 400 次更新预热。训练 replay 使用 train 模式和配置中的 dropout。

### 8.4 梯度边界

固定梯度契约：

| 模块 | 基础导航损失 | 调整后导航损失 |
|---|---:|---:|
| 导航策略 | 更新 | 不更新 |
| 导航 CLS residual MLP | 更新 | 不更新 |
| RGB adapter | 更新 | 不更新 |
| Top-5 CLS adapter | 不更新 | 更新 |
| E24 | 不更新 | 更新 |
| NWM | 冻结 | 冻结 |
| DINOv2 | 冻结 | 冻结 |
| DINO-CWP | 冻结 | 冻结 |
| waypoint predictor | 冻结 | 冻结 |

adjusted 分支中的基础 logits、owner、文本、geometry、NWM CLS/patch 在进入可训练 Top-5 adapter/E24 前均 detached。Top-5 adapter 的参数梯度来自 replay 中重新构造的 adjusted loss。

### 8.5 DDP 与优化器

Top-5 CLS adapter 和 E24 由一个联合 DDP wrapper 持有，统一完成 forward 和梯度同步。RGB adapter 继续使用当前 rank0 广播与手工梯度 all-reduce 方式，避免无关改造。

优化器参数组固定为：

```text
navigation_decay
navigation_no_decay
top5_cls_adapter
e24
```

- RGB adapter 位于 navigation 两组中，学习率沿用导航 `1e-5`。
- Top-5 CLS adapter 默认学习率 `1e-5`。
- E24 默认学习率 `5e-6`。
- Top-5 adapter 和 E24 的 weight decay、梯度裁剪分别配置并写入 provenance。

## 9. 配置与资产

### 9.1 独立配置

新增以下独立配置，不修改旧实验语义：

```text
configs/nwm/raenwm_mp3d_fresh_cls.yaml
run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml
```

新配置至少固定：

```text
predict_cls_token: true
context_size: 4
latent_dim: 768
latent_size: 16
token_count: 257
num_steps: 10
sampling_method: euler
final_only_euler: false
```

新配置不包含 `head_checkpoint_path` 或 `head_checkpoint_sha256`。启动时计算固化推理配置的 SHA-256，并写入模型 checkpoint provenance。

### 9.2 稳定资产入口

训练机 ETP-R1 项目内使用以下忽略路径的稳定软链接指向用户指定 checkpoint：

```text
pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar
```

软链接目标必须解析为已核验的训练机文件，并在反序列化前校验 SHA-256。

测评机使用 `/home/a6000/gwl/ETP-R1` 自己拥有的资产副本。允许从受保护 ETPNav/RAE-NWM 位置只读复制必要文件，但不得原地修改。两端必须校验相同 SHA。

大 checkpoint、DINOv2、stat、DINO-CWP、日志和运行结果不提交 Git。

### 9.3 严格加载

新模式加载 75k checkpoint 时：

1. 校验完整文件 SHA。
2. 读取非空 `ema`。
3. 清理可能存在的 `_orig_mod.` 前缀。
4. 用 native-CLS CDiT 结构严格加载。
5. missing key 或 unexpected key 任一非空都失败。
6. 检查配置声明的 257-token 模式与 checkpoint 结构一致。
7. 禁止回退到 patch-only 采样。

## 10. 导航 checkpoint 与恢复

### 10.1 新模型 checkpoint

模型 checkpoint 格式固定升级为 `etpr1-native-cls-e24-joint-v2`，不与 `etpr1-e24-joint-v1` 混用。新 checkpoint 保存：

- 导航 `state_dict`
- RGB adapter state
- Top-5 adapter+E24 wrapper state
- E24 原始初始化 metadata
- 新 NWM/E24 provenance
- 配置快照
- iteration

不再保存 `raenwm_heads_state_dict`，也不嵌入外部 NWM 权重。

provenance 至少包含：

- 75k checkpoint SHA
- NWM 推理配置 SHA
- `num_steps=10`
- DINOv2 资产身份
- stat SHA
- DINO-CWP SHA
- base `iter14200` SHA
- `e24_avg3.pth` SHA
- 两个 adapter 的结构配置
- 两项 loss 权重
- action warmup
- Top-5 数量

### 10.2 新实验初始化

新实验允许 base `iter14200` 不含新 adapter 状态：

- 导航策略从 base 加载。
- RGB adapter 新建并零残差初始化。
- Top-5 CLS adapter 新建并零残差初始化。
- E24 从 `e24_avg3.pth` 仅加载权重。

该豁免只适用于明确的 new-run 初始化，不能用于评测或 requeue。

### 10.3 严格恢复

正式恢复必须找到成对的模型 checkpoint 和 training state，并严格恢复：

- 四个 optimizer 参数组及名称
- scheduler
- scaler
- iteration
- episode iterator state
- Python/NumPy/PyTorch/CUDA RNG
- 每 rank 的 NWM `torch.Generator` 状态
- Top-5 adapter/E24 DDP state
- RGB adapter state

training-state 的整数 `format_version` 固定升级为 4。旧 E24 checkpoint、旧三组 optimizer 状态或缺少 NWM generator 状态的文件不能作为新实验 requeue 来源。

评测必须含两个 adapter 和 E24 state；缺失时立即失败，不使用随机初始化或静默关闭功能。

## 11. 失败处理与诊断

### 11.1 启动失败

以下情况启动立即失败：

- 外部资产不存在或 SHA 不匹配。
- NWM 配置不是 native CLS 257-token 模式。
- EMA 为空、missing keys 或 unexpected keys。
- DINOv2/stat 与 NWM 配置不兼容。
- E24 权重结构不匹配。
- 评测/requeue checkpoint 缺少任一 adapter 或 provenance 不一致。

### 11.2 行级安全回退

运行时保留当前“batch 失败后逐行重试”：

- batch 预测异常时对每个 request 单独预测。
- 单行仍异常、shape 错误或含非有限值时只废弃该行。
- RGB 行失败时保留原始候选 CLS。
- Top-5 q0/q1 行失败时对应 future 槽无效。
- DINO-CWP `none` 时对应 q1 槽无效。
- 其他有效环境和槽继续运行。

不会把随机值、零 CLS 或旧 head 输出伪装成有效预测。

### 11.3 诊断

训练和评测至少记录：

- 上下文未就绪数
- q0/q1 请求数、成功数、batch/row failure
- DINO-CWP none/invalid
- future valid rate
- native normalized/raw CLS 均值、标准差、范数
- 预测 patch 统计
- RGB agreement、gate、修正范数、融合候选数
- Top-5 adapter CLS 修正范数
- E24 delta 范数
- adjusted loss 和 base loss
- action flip 数与比例
- NWM 推理耗时
- fallback 数和原因

## 12. 测试设计

### 12.1 单元测试

覆盖：

- `[CLS+patch]` 打包/拆包往返。
- CLS spatial-stat 归一化和反归一化。
- raw CLS、nav CLS、raw patch 输出边界。
- 257-token context/snapshot/batch shape。
- 257-token 初始噪声与 10 步采样 shape。
- strict EMA 加载和旧 patch-only checkpoint 拒绝。
- 新模式不创建/调用旧 heads。
- `NwmPrediction` 的 tokens/CLS/patch 一致性。
- RGB adapter 网络层形状保持不变。
- RGB adapter 初始恒等、agreement 输入和非零梯度。
- Top-5 CLS adapter 初始恒等、condition 编码和非零梯度。
- Top-5 adapter 只改变 token 0，patch bitwise 不变。
- adjusted full-logits CE 的 STOP、Top-5 内教师、Top-5 外教师和 padding。
- adjusted loss 只更新 Top-5 adapter/E24。
- base loss 能更新 RGB adapter。
- DDP dummy row 和跨 rank 梯度一致。
- 新 checkpoint 保存/严格加载/错误格式拒绝。
- requeue 恢复 NWM generator 和四个 optimizer group。

### 12.2 真实数值一致性

在训练机使用固定真实四帧 context、固定 condition、固定 initial noise，分别运行：

```text
RAE-NWM 原始实现，10 步
ETP-R1 新兼容层，10 步
```

比较：

- 完整 257 token
- normalized CLS
- raw CLS
- patch
- shape 和有限值
- max absolute error
- cosine similarity

参考与生产环境的 Python、PyTorch、CUDA、Transformers 和精度设置必须记录。若依赖版本不同，使用预先声明的数值容差，不用只比较 shape 代替数值对齐。

### 12.3 训练机 SFT 冒烟

在 R2R SFT 实际运行环境依次完成：

1. 环境导入与版本记录。
2. 真实 75k checkpoint 严格加载。
3. 单环境单 rollout。
4. 冻结模块与两个 adapter/E24 梯度审计。
5. 至少两次优化器更新并保存模型/训练状态。
6. 新进程恢复并完成下一次更新。
7. optimizer/scheduler/scaler/iteration/episode/RNG 一致性检查。
8. 双卡 DDP 短冒烟。

不从 smoke 目录接续任何后续正式实验。

### 12.4 测评机单 episode

执行前按项目规则检查：

```text
nvidia-smi
docker ps
gwl-etpnav 中的 torchrun/run.py/train.py
```

若 ETPNav 正占用 GPU，等待或向用户说明，不终止其任务。

在 `gwl-etpr1-rae` 与 `etpr1_rae` 中完成固定 R2R `val_unseen` 单 episode，验证：

- 两条新链路都实际被调用。
- 上下文未满 4 帧时（通常为前三个有效观测）安全回退生效。
- 真实 checkpoint/adapter/E24 状态加载成功。
- 冻结模块未改变。
- 轨迹、结果和诊断正常写出。
- 无 NaN、哈希错误或 silent fallback。

### 12.5 完整回归

在目标机器的正式环境运行全部相关测试和完整现有测试套件。旧配置保持关闭时，旧 CLIP、RAE/DINOv2、旧 NWM 和普通 SFT 行为不得改变。

## 13. 技术验收条件

本任务在以下条件全部满足时完成：

- 新兼容层真实运行 75k native CLS 模型的 10 步序列采样。
- 真实数值一致性测试通过。
- RGB 使用反归一化后的原生预测 CLS，且 adapter 初始恒等。
- Top-5 使用调整后 CLS + 原始 256 patch，E24 从 avg3 权重初始化。
- 基础/调整后两项导航损失和梯度边界符合设计。
- NWM、DINOv2、DINO-CWP、waypoint predictor 冻结审计通过。
- checkpoint 保存、严格评测加载和跨进程恢复通过。
- 单元测试、完整回归、单/双卡 SFT smoke 和单 episode 评测通过。
- 未修改受保护环境、受保护工程或 ETPNav 容器。
- 未运行长训练或完整 `val_unseen` 性能实验。

## 14. 预计影响的代码边界

实施会集中在以下区域，最终逐文件步骤由后续实施计划确定：

- `configs/nwm/`
- `run_r2r/`
- `vlnce_baselines/config/default.py`
- `vlnce_baselines/models/encoders/rae_dinov2_encoder.py`
- `vlnce_baselines/models/R1Policy.py`
- `vlnce_baselines/nwm/types.py`
- `vlnce_baselines/nwm/etp_adapter.py`
- `vlnce_baselines/nwm/runtime.py`
- `vlnce_baselines/nwm/predictor.py`
- `vlnce_baselines/nwm/raenwm_core/`
- `vlnce_baselines/nwm/rgb_fusion.py`
- `vlnce_baselines/nwm/active_lookahead/`
- `vlnce_baselines/ss_trainer_ETP_R1.py`
- `scripts/`
- `tests/` 与 `tests/integration/`

## 15. 已确认的用户决策

- Top-5 q1 保留完整 257 token，只先调整 CLS，patch 继续作为 E24 辅助信息。
- 修正后的完整导航 logits 直接计算教师动作交叉熵，同时保留原始导航损失。
- adjusted loss 不反向更新基础导航策略。
- RGB adapter 与 Top-5 adapter 使用独立权重。
- RGB adapter 只复用当前结构，不复用旧权重。
- E24 复用原始 `e24_avg3.pth` 权重。
- 首阶段只覆盖 R2R SFT/评测。
- NWM 固定使用 10 步。
- 4 帧历史不足时安全回退。
- base 使用当前 `iter14200`。
- 只要求改好代码并通过技术测试，不设置性能提升条件。
- Top-5 CLS adapter 的 `dx/dy` 使用 NWM 的归一化局部动作坐标。

## 16. 最后复查

2026-08-26，为设计原生 CLS RAE-NWM 替换而复查：

- 当前分支和未提交修改；`research.md` 的既有用户修改未被触碰。
- 当前 RGB fusion、NWM runtime、外置 heads、q0/q1、DINO-CWP、E24 replay、DDP、checkpoint 和评测调用链。
- 训练机指定 75k checkpoint 的文件身份、EMA 结构、实验配置、原生 CLS+patch 打包、RAE 统计量归一化和已有 CLS 评测。
- R2R 当前配置、管理脚本、资产验证脚本和相关测试。

本设计没有未决功能范围；实施阶段的具体文件拆分、测试先后和提交粒度在用户确认本文档后写入可执行实施计划。
