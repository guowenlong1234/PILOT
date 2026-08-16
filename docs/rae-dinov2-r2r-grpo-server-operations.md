# RAE/DINOv2 R2R GRPO 训练机操作说明

## 当前默认实验

本入口只用于训练机 `/home/gwl/project/etpr1/ETP-R1`，固定使用两张
RTX A6000。当前暂定的 SFT 起点是：

```text
data/logs/rae_dinov2_etpnav_cls_768/
  r2r_sft_legacy452500_nonvisual_20260815/checkpoints/
  rae_dinov2_etpnav_cls_768_legacy452500_nonvisual_r2r_sft/
  ckpt.iter8000.pth
```

对应的预训练基座是：

```text
pretrained/r2r_rxr_ce/
  rae_dinov2_etpnav_cls_768_legacy_base_transfer/
  model_step_452500_nonvisual_transfer.pt
```

这两个路径只是默认值，不做哈希绑定，可以在启动命令中覆盖。

默认使用每卡 8 个 Habitat 环境、每组 8 条采样轨迹和 1000 次更新。
原始发布配置是 4 卡、每卡 8 环境、500 次更新；双卡运行 1000 次可保持
相同的总 rollout 数量。其余 GRPO 参数沿用原始 R2R 发布入口：学习率
`2e-5`、无预热、一次更新 epoch、`beta=0.04`、裁剪范围 `0.2`。

## 启动和查看状态

启动前先确认当前分支和工作区干净，并确认两张 GPU 都空闲：

```bash
ssh server
cd /home/gwl/project/etpr1/ETP-R1
git status --short --branch
scripts/manage_rae_r2r_grpo_server.sh status
```

正式启动：

```bash
scripts/manage_rae_r2r_grpo_server.sh start
```

查看日志：

```bash
scripts/manage_rae_r2r_grpo_server.sh logs
scripts/manage_rae_r2r_grpo_server.sh tail
```

发送正常终止信号：

```bash
scripts/manage_rae_r2r_grpo_server.sh stop
```

管理脚本要求训练机恰好能看到两张 GPU，并且每张卡启动前的显存占用不
超过 1 GiB。已有输出目录中一旦存在模型 checkpoint，`start` 会拒绝覆盖，
必须使用 `resume` 或指定新的实验名和输出目录。

## 临时更换加载权重

更换 SFT checkpoint 和对应预训练基座时，不需要修改脚本：

```bash
ETPR1_R2R_GRPO_SFT_CHECKPOINT=/absolute/path/to/ckpt.iterXXXX.pth \
ETPR1_R2R_GRPO_PRETRAINED_PATH=/absolute/path/to/model_step_XXXX.pt \
ETPR1_R2R_GRPO_EXP_NAME=my_dino_r2r_grpo \
ETPR1_R2R_GRPO_OUTPUT_ROOT=data/logs/my_dino_r2r_grpo \
scripts/manage_rae_r2r_grpo_server.sh start
```

可覆盖的训练项还包括：

```text
ETPR1_R2R_GRPO_ITERS
ETPR1_R2R_GRPO_LOG_EVERY
ETPR1_R2R_GRPO_KEEP_LAST_STATES
ETPR1_R2R_GRPO_KEEP_STATE_EVERY
ETPR1_R2R_GRPO_MASTER_PORT
ETPR1_R2R_GRPO_CHECKPOINT_SYNC_ENABLED
ETPR1_R2R_GRPO_CHECKPOINT_SYNC_DESTINATION
```

## Checkpoint 同步到测评机

正式入口默认开启模型 checkpoint 同步。每次本地模型和训练状态都原子保存
完成后，会启动独立后台进程，把模型 checkpoint 经 2.5 GbE 直连同步到：

```text
a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/<OUTPUT_ROOT>/checkpoints/<EXP_NAME>
```

传输先写入测评机目标目录下的 `.incoming`，核对文件大小后再原子改名；
评测端不会看到半份 checkpoint。同步失败会写入训练机 checkpoint 目录下的
`checkpoint_sync.log`，不会中断 GRPO 训练。

和 SFT 当前策略一致，只同步可用于评测的 `ckpt.iterN.pth`。包含优化器、
调度器和随机状态的 `train_state.iterN.pth` 留在训练机，用于本机断点恢复，
不复制到测评机。

自定义相对输出目录时，同步目标会自动跟随实验名和输出目录。使用绝对输出
目录时，应显式指定测评机目标：

```bash
ETPR1_R2R_GRPO_OUTPUT_ROOT=/mnt/data2tb/my_grpo \
ETPR1_R2R_GRPO_CHECKPOINT_SYNC_DESTINATION=a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/data/logs/my_grpo/checkpoints/my_grpo \
ETPR1_R2R_GRPO_EXP_NAME=my_grpo \
scripts/manage_rae_r2r_grpo_server.sh start
```

如需临时关闭：

```bash
ETPR1_R2R_GRPO_CHECKPOINT_SYNC_ENABLED=False \
scripts/manage_rae_r2r_grpo_server.sh start
```

## 断点恢复

每个保存点包含一对文件：

```text
checkpoints/<experiment>/ckpt.iterN.pth
checkpoints/<experiment>/train_states/train_state.iterN.pth
```

模型文件保存导航网络权重；独立训练状态保存优化器、学习率调度器、混合
精度缩放器、两个 rank 的环境 episode 队列以及 Python、NumPy、PyTorch
和 CUDA 随机状态。恢复时只选择迭代数最大的完整文件对，忽略没有配对的
残留文件。

恢复命令：

```bash
scripts/manage_rae_r2r_grpo_server.sh resume
```

如果启动时覆盖过实验名、输出目录、权重或迭代数，恢复时必须重复相同的
环境变量。例如：

```bash
ETPR1_R2R_GRPO_SFT_CHECKPOINT=/absolute/path/to/ckpt.iterXXXX.pth \
ETPR1_R2R_GRPO_PRETRAINED_PATH=/absolute/path/to/model_step_XXXX.pt \
ETPR1_R2R_GRPO_EXP_NAME=my_dino_r2r_grpo \
ETPR1_R2R_GRPO_OUTPUT_ROOT=data/logs/my_dino_r2r_grpo \
scripts/manage_rae_r2r_grpo_server.sh resume
```

恢复会核对 GPU/rank 数、每 rank 环境数、总迭代数、保存间隔、学习率、
采样数、更新 epoch、GRPO 系数、dropout、轨迹与文本长度、环境噪声、数据
后缀和预训练路径。关键配置不一致时会明确拒绝恢复，避免把不同实验接在
一起。
