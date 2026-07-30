# RAE/DINOv2 R2R SFT 运行手册

## 正式配置

- 测评机：单张 RTX 4090 24GB。
- 并行环境：8，与原工程每张 GPU 的设置相同。
- 梯度累积：4 次。
- 有效 batch：`8 x 4 = 32`，与原工程 `4 GPU x 8` 一致。
- 参数更新：30,000 次。
- 学习率：`1e-5`。
- 预热：500 次更新。
- 最低学习率比例：`1.0`，与原工程发布脚本一致，即预热后保持
  `1e-5`。
- DAgger 采样率：初始 `0.75`，每 2,000 次更新衰减。
- 路点增强：开启。
- 训练数据：R2R `train_90`。
- 预训练初始化：
  `pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/best/model_best_step_452500.pt`。

单卡有效 batch 与原四卡相同，因此训练样本量和优化器更新次数保持
一致。代价是单卡需要串行完成四个 micro-batch。正式权重短测中一次
累积更新约 13.3 秒，据此估算 30,000 次更新约需 4.6 天，checkpoint
写入和运行波动会使实际时间略长。

## Checkpoint

每 200 次更新分别保存：

- `ckpt.iter<iteration>.pth`：只包含模型参数、编码器元数据和配置，
  全部永久保留，供训练结束后逐一验证并选择最佳模型。
- `train_states/train_state.iter<iteration>.pth`：包含优化器、学习率调度
  器、混合精度缩放器、迭代数、对应模型文件名，以及每个训练进程中
  每个并行环境的 episode 剩余队列、迭代器计数和随机数状态，只用于
  断点恢复。

两类文件都先写入临时文件再原子改名。恢复时只承认模型和训练状态同时
存在的完整配对。模型目录顶层没有训练状态文件，可直接交给验证流程
扫描。模型 checkpoint 不清理；训练状态自动保留最近 3 个和每 5,000
次更新的里程碑。

带 episode 状态的新训练状态会从保存点的下一个 episode 继续，不再
从新建环境队列的头部开始。为保证保存的环境队列仍然有效，恢复时必须
保持训练进程数、每个进程的并行环境数和数据划分不变；否则会直接报错，
避免静默使用错误顺序。旧训练状态仍可恢复模型、优化器、调度器、缩放器
和迭代数，但由于历史文件没有 episode 信息，只能从新建环境队列开始，
日志会明确警告这一点。这里保证的是 episode 顺序连续，不承诺策略采样
等训练随机过程逐位复现。

按短测估算，模型 checkpoint 约 1.5GB，独立训练状态约 3.0GB。30,000
次训练会产生 150 个模型文件，约占 225GB；恢复状态最多保留 8 个，
约占 24GB。加上原子写入临时余量，正式输出预计需要约 255GB，当前
测评机空间足够。

输出目录：

```text
data/logs/rae_dinov2/r2r_sft_formal/
```

## 启动与管理

以下命令从笔记本执行。

首次启动：

```bash
ssh eval 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_r2r_sft_host.sh start'
```

状态：

```bash
ssh eval 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_r2r_sft_host.sh status'
```

日志：

```bash
ssh eval 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_r2r_sft_host.sh logs'
```

持续查看：

```bash
ssh eval 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_r2r_sft_host.sh tail'
```

从最新完整 checkpoint 恢复：

```bash
ssh eval 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_r2r_sft_host.sh resume'
```

请求正常停止：

```bash
ssh eval 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_r2r_sft_host.sh stop'
```

宿主机入口会检查 ETPNav 任务和 GPU 使用情况，发现冲突时拒绝启动。
训练由容器内 `tmux` 会话 `etpr1-rae-r2r-sft` 托管。

## 2026-07-29 启动前验收

- 正式 `model_best_step_452500.pt` 加载和结构审计通过。
- 8 个并行环境单次更新通过，峰值显存 17,192MiB。
- 8 环境、4 次累积完成第 1 次更新并保存；新进程恢复后完成第 2 次
  更新，恢复进程峰值显存 21,241MiB。
- 恢复状态含 479 组优化器状态，学习率调度器位于第 2 次更新，
  混合精度缩放器状态完整。
- SFT 审计确认 6 个 RGB 投影参数均已更新，冻结的 DINOv2 骨干未写入
  在线 checkpoint。
- 独立保存实测：模型文件约 1.53GB，不含优化器；恢复状态约 3.00GB，
  不含模型参数。删除旧恢复状态后，同迭代的模型文件保持不变。
- 完整测试：`305 passed, 3 warnings`。
