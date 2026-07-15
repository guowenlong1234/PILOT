# RAE/DINOv2 联合预训练运行手册

## 当前正式配置

- 测评机：单张 RTX 4090 24GB。
- 单次小 batch：16。
- 梯度累积：8 次。
- 有效 batch：128，与原四卡 `4 x 32` 一致。
- 总参数更新：500,000。
- 每 2,500 次更新验证并保存一次。
- 最近 3 个检查点同时保留模型和完整训练状态，可用于恢复。
- 每 25,000 步额外保留一个模型里程碑；较旧的优化器状态自动清理。

每个可恢复点由两个文件组成：

```text
ckpts/model_step_<step>.pt
ckpts/train_state_<step>.pt
```

前者供下游加载模型，后者记录优化器、混合精度缩放器、全局步数、数据混合步数和 Python/NumPy/PyTorch 随机状态。只有两者都存在时，`latest` 才会把该步视为有效恢复点。文件先写临时文件再原子改名，进程在写入中途退出时不会把半个文件误当成有效 checkpoint。

## 长任务命令

以下命令都从本机执行。宿主机入口会先确认 `eno1=10.10.10.2`，并检查 ETPNav 进程和 GPU；发现已有任务时会拒绝启动，不会停止别的任务。

首次启动：

```bash
ssh 4090 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_pretrain_host.sh start'
```

查看状态：

```bash
ssh 4090 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_pretrain_host.sh status'
```

持续查看日志：

```bash
ssh 4090 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_pretrain_host.sh tail'
```

电脑或容器异常退出后，从最新完整检查点恢复：

```bash
ssh 4090 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_pretrain_host.sh resume'
```

正常请求停止：

```bash
ssh 4090 'cd /home/a6000/gwl/ETP-R1 && scripts/manage_rae_pretrain_host.sh stop'
```

托管会话名默认为 `etpr1-rae-pretrain`，日志位于：

```text
pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/supervisor/
```

每次启动还会在同一目录生成 `*_source_identity.json` 和 `*_source_manifest.sha256`。测评机的 `.git` 指针不可用时，以逐文件清单的整体校验和标识真实训练源码，不依赖远端 Git 提交号。

训练本体仍是原来的 `torchrun` 命令，只由容器内的 `tmux` 保持运行。SSH 断开不会结束训练。`status` 同时显示 tmux 会话、GPU、磁盘、最近 checkpoint 和日志末尾。

## 恢复边界

- 恢复时强制核对单卡 batch、梯度累积、GPU 数量和模型结构配置；不一致会直接拒绝，避免接错实验。
- `num_train_steps` 可以在恢复时增加，因此允许延长训练。
- 当前每 2,500 步生成一个恢复点。突然断电最多会丢失最近一个保存间隔内的进度。
- 数据加载器会从新的随机采样流继续，不承诺中断前后的逐样本、逐位完全一致；模型、优化器、学习率所对应的全局步数和随机状态会恢复。
- 旧的 `model_step_*.pt` 没有配套 `train_state_*.pt` 时只能作为模型初始化或下游权重，不能恢复优化器和训练步数。

## 已完成验证

2026-07-15 在专用容器和环境中完成：

- 恢复与保留策略单元测试通过。
- 真实模型先训练到第 1 步并写出约 2.2GB 模型和约 2.7GB 训练状态，再由新进程恢复并完成第 2 步。
- 第 2 步状态记录 `step=2`、`meta_loader_step=2`、484 组优化器状态。
- 全量测试结果为 `276 passed, 3 warnings`。
- 验证日志保存在测评机 `data/logs/rae_dino_resume_validation/20260715/`。

本轮没有启动正式 500,000 步训练。
