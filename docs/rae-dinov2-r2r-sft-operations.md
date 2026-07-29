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

每 200 次更新保存一个包含模型、优化器、学习率调度器、混合精度缩放器
和迭代数的完整 checkpoint。写入先落到临时文件，再原子改名。

自动保留：

- 最近 3 个 checkpoint；
- 每 5,000 次更新的里程碑；
- 最终 30,000 次 checkpoint。

实测完整 checkpoint 约 4.5GB。训练结束时最多保留 6 个里程碑和最近
3 个中的两个非里程碑文件，稳定占用约 36GB；原子写入新文件时还需
约 4.5GB 临时余量。

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
- checkpoint 含 479 组优化器状态，学习率调度器位于第 2 次更新，
  混合精度缩放器状态完整。
- SFT 审计确认 6 个 RGB 投影参数均已更新，冻结的 DINOv2 骨干未写入
  在线 checkpoint。
- 完整测试：`302 passed, 3 warnings`。
