# 持久候选状态组：两机并行测评

2026-09-14，按用户要求测评 `ghost_concat_persistent_compiled_10k_20260911`。共50个检查点，两机按交错步数分工，单卡8环境，每点完整评测R2R `val_unseen` 的1839条路线。

| 机器 | 检查点 | 队列 |
| --- | --- | --- |
| 训练机 / GPU0 | 400、800……10000，共25点 | `ghost_concat_job.py watch --machine server` |
| 测评机 / GPU0 | 200、600……9800，共25点 | `ghost_concat_job.py watch --machine eval` |

两边均显式设置 `--memory-mode persistent_node_state --train-policy --environments 8 --panorama-context-mode world_exact_select --observation-source direct --visual-precision fp16 --dino-batch 64 --nwm-batch 64 --compile-model --compile-backend inductor`，第二阶段关闭，seed100。使用对应机器的既有专用运行环境；测评机仍为 `gwl-etpr1-rae` / `etpr1_rae`。这是同一实验的检查点分工，不是重复测评；汇总时应保留机器标记。

## 模型交付与结果

两台机器的实验根均为各自工程下 `data/logs/ghost_concat_persistent_compiled_10k_20260911/`。

- 训练机交付进程逐个传送测评机负责的25点，经专线原子发布；测评完成后核对模型SHA256和1839条路线，才释放测评机临时模型副本。训练机所有原件保留。
- 两机各自写入 `eval_summary.json`，只包含自己负责的25点；需要合并两份表才能判断全50点完成。
- 逐点结果及运行版本在 `eval/ghost_concat_v1_joint_persistent_eval_iter<步数>/`。
- 精确命令保存在两机 `eval_launch_20260914.json`；训练机实际重启命令及隔离库环境保存在 `eval_launch_isolated_driver_20260914.json`。交付记录为训练机 `delivery_launch_20260914.json`。
- 测评机启动时间10:49:54、主管PID2373165；训练机实际重启时间10:55:13、主管PID429768；交付PID429413。以上为北京时间，后续必须用实际命令核验PID身份。

## 训练机驱动问题与本次处理

系统9月12日升级驱动用户态库至580.178.04，但运行中的内核模块仍为580.173.02。初次启动能完成CUDA实算，导航模拟器EGL渲染失败；失败manifest和日志保留。没有重启主机或安装/替换系统驱动。

从 Ubuntu 官方历史快照 `https://snapshot.ubuntu.com/ubuntu/20260910T000000Z/` 下载 `libnvidia-gl-580` 和 `libnvidia-compute-580`，精确版本 `580.173.02-0ubuntu0.22.04.1`、架构amd64。仅以 `dpkg-deb -x` 解压到训练机 `.runtime/nvidia-580.173.02/root/`，包及SHA256记录在其 `packages/` 子目录。实际测评主管及其子进程设置：

```bash
LD_LIBRARY_PATH=/home/gwl/project/etpr1/ETP-R1/.runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu
```

此环境下 `nvidia-smi` 已恢复正常，模拟器已越过此前失败点并初始化任务。系统默认环境仍存在版本不匹配，后续重启或系统升级后应重新核对，不能盲目沿用此隔离目录。

## 代码和检查

- `839fa97`：支持训练机单卡watch队列、持久状态模型交付及指定检查点子集。
- `1f36528`：交付子集校验与测试；训练机遇到明确NVML版本错误时可用CUDA实算及显存探针检查资源，仍保留每卡锁和1GiB占用阈值。实际重新启动使用匹配的隔离驱动库，走正常nvidia-smi检查。
- 训练机实际环境：入口、交付及资源检查共24项测试通过，退出0。
- 测评机实际环境：入口12项、持久状态和训练器34项测试通过，退出0。
- 第200步测评机加载SHA256与训练机原件一致：`cc679cdd871a1ec83d5dcb4682d5c3ed7656cea03ac32de817b06baf0b97f497`。
- 测评机日志记录Python3.11.15、PyTorch2.2.2+cu121、CUDA12.1、Transformers4.49.0、Habitat/Habitat-Sim0.3.3。

启动验收：训练机第400步已推进18/1839，GPU0显存12875MiB、计算利用率98%；测评机第200步已推进183/1839，显存12780MiB、利用率99%。两侧主管、实际测评子进程和训练机交付进程均存活，日志持续增长。当前是启动记录，不能作为完整测评已结束的证明。
