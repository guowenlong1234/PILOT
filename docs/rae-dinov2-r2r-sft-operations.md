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
一致。代价是单卡需要串行完成四个 micro-batch，预计训练持续数天。

## Checkpoint

每 200 次更新保存一个包含模型、优化器、学习率调度器、混合精度缩放器
和迭代数的完整 checkpoint。写入先落到临时文件，再原子改名。

自动保留：

- 最近 3 个 checkpoint；
- 每 5,000 次更新的里程碑；
- 最终 30,000 次 checkpoint。

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
