# RAE/DINOv2 联合预训练运行手册

> 当前视觉契约已切换为 ETPNav 导航链路：原始 768 维 CLS、不对 CLS
> 使用 `stat.pt`、下游直接 `img_linear(768→768)`。旧的
> `rae_dinov2_cls_mlp` 目录、归一化 HDF5 和对应 checkpoint 与当前代码
> 不兼容，不能续训；当前脚本使用独立目录
> `rae_dinov2_etpnav_cls_768`。

## 当前正式配置

- 测评机：单张 RTX 4090 24GB。
- 单次小 batch：16。
- 梯度累积：8 次。
- 有效 batch：128，与原四卡 `4 x 32` 一致。
- 数据加载：`n_workers=0`、`pin_mem=false`，不创建 DataLoader 子进程。
- CPU batch 后台线程预取：开启；只在线程中准备下一批数据，CUDA 搬运仍由训练主线程完成。
- 总参数更新：500,000。
- 每 2,500 次更新验证并保存一次。
- 最近 3 个检查点同时保留模型和完整训练状态，可用于恢复。
- 每 25,000 步额外保留一个模型里程碑；较旧的优化器状态自动清理。
- 每次验证计算联合准确率总分，并永久保留历史总分最高的模型。

每个可恢复点由两个文件组成：

```text
ckpts/model_step_<step>.pt
ckpts/train_state_<step>.pt
```

前者供下游加载模型，后者记录优化器、混合精度缩放器、全局步数、数据混合步数和 Python/NumPy/PyTorch 随机状态。只有两者都存在时，`latest` 才会把该步视为有效恢复点。文件先写临时文件再原子改名，进程在写入中途退出时不会把半个文件误当成有效 checkpoint。

## 最佳检查点

每 2,500 步完成 R2R 和 RxR 验证后，按照下面的固定公式计算总分：

```text
MLM平均准确率 = (R2R MLM acc + RxR MLM acc) / 2
SAP平均准确率 = (R2R SAP gacc + RxR SAP gacc) / 2
总分 = MLM平均准确率 + SAP平均准确率
```

只有新总分严格高于历史最高分时，才更新：

```text
pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768/best/
  model_best_step_<step>.pt
  best_metrics.json
```

`best_metrics.json` 记录四个原始准确率、两个平均准确率、总分、选择公式和训练步数。最佳模型通过硬链接指向刚保存的周期模型：周期模型仍存在时不会重复占用 2.2GB；周期模型以后被清理时，最佳模型链接仍会保留同一份数据。旧的最佳模型会在更高总分出现后删除。

最佳模型用于下游训练和最终比较，不代替断点恢复。训练中断时仍从 `ckpts/` 中最新的模型与 `train_state` 完整配对恢复。

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
pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768/supervisor/
```

每次启动还会在同一目录生成 `*_source_identity.json` 和 `*_source_manifest.sha256`。测评机的 `.git` 指针不可用时，以逐文件清单的整体校验和标识真实训练源码，不依赖远端 Git 提交号。

训练本体仍是原来的 `torchrun` 命令，只由容器内的 `tmux` 保持运行。SSH 断开不会结束训练。`status` 同时显示 tmux 会话、GPU、磁盘、最近 checkpoint 和日志末尾。

## 恢复边界

- 恢复时强制核对单卡 batch、梯度累积、GPU 数量和模型结构配置；不一致会直接拒绝，避免接错实验。
- 不要把 `n_workers` 改回 1，也不要重新开启 `pin_mem`。旧配置会在 CUDA、HDF5 和大模型初始化后，以 Linux 默认 `fork` 为 MLM、SAP 各创建一个 worker；它曾先后触发 worker 堆内存错误、worker 段错误和 rank 0 段错误。
- 当前 `thread_prefetch=true` 使用同一进程内的一个后台线程重叠 CPU batch 准备和 GPU 计算，不经过多进程队列、共享内存或锁页内存线程。
- `num_train_steps` 可以在恢复时增加，因此允许延长训练。
- 当前每 2,500 步生成一个恢复点。突然断电最多会丢失最近一个保存间隔内的进度。
- 数据加载器会从新的随机采样流继续，不承诺中断前后的逐样本、逐位完全一致；模型、优化器、学习率所对应的全局步数和随机状态会恢复。
- 旧的 `model_step_*.pt` 没有配套 `train_state_*.pt` 时只能作为模型初始化或下游权重，不能恢复优化器和训练步数。

## 已完成验证

2026-07-15 在专用容器和环境中完成：

- 恢复与保留策略单元测试通过。
- 真实模型先训练到第 1 步并写出约 2.2GB 模型和约 2.7GB 训练状态，再由新进程恢复并完成第 2 步。
- 第 2 步状态记录 `step=2`、`meta_loader_step=2`、484 组优化器状态。
- 联合准确率最佳模型完成真实一步 GPU 验证；模型硬链接、指标 JSON 和更新日志均正确落盘。
- 全量测试结果为 `280 passed, 3 warnings`。
- 验证日志保存在测评机 `data/logs/rae_dino_resume_validation/20260715/`。
- 最佳模型验证日志保存在测评机 `data/logs/rae_dino_best_checkpoint_validation/20260715/`。

2026-07-23 又完成数据加载稳定性修复验证：

- 原 `n_workers=1` 实际会为 MLM、SAP 创建两个 worker；真实数据压力测试确认每个 worker 初始 RSS 约 14GB，并随特征缓存增长到约 15GB。
- `pin_mem=true` 与 `pin_mem=false` 各完成 5,000 个真实 micro-batch；短测不能稳定复现低概率段错误，但确认关闭锁页内存不能消除大进程 `fork` 结构。
- 新的单进程后台线程预取用正式 `train_state_117500.pt` 完成真实模型短测，从 117,500 推进到 117,703；没有 DataLoader 子进程或异常。
- 同步单进程路径约 1.48 秒/步，后台线程预取约 1.27 秒/步，短测提速约 14%。
- 修复后完整测试为 `290 passed, 3 warnings`。
- 压力测试日志位于测评机 `data/logs/dataloader_stress/`，真实模型测速日志位于 `data/logs/thread_prefetch_benchmark/`。
