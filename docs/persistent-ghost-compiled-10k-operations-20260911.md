# 持久候选状态：编译版 10000 步训练

2026-09-11 按用户要求，在训练机完成持久状态真实验证后，将 `feature/persistent-ghost-state` 合入当前 `feature/e24-joint-sft`。合并提交 `1093ee8`，已通过中央裸仓库同步训练机主工作区。笔记本原有论文图片、笔记、`.gitignore`、文档删除和 `research.md` 未提交修改保留。

验证记录见 [persistent-ghost-gpu-validation-20260911.md](persistent-ghost-gpu-validation-20260911.md)。新长训练重新使用原始 `base_iter14200.pth`，不使用验证过程中的模型或优化器。基座 SHA-256 为 `1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61`。

## 参数与启动

训练位置：`/home/gwl/project/etpr1/ETP-R1`，环境 `etpnav_unified`，本工作区 `.runtime/server_sft`。

```bash
python3 scripts/ghost_concat_job.py train \
  --memory-mode persistent_node_state --train-policy \
  --gpus 0,1 --batch 4 --iters 10000 --log-every 200 \
  --policy-lr 2e-6 --fusion-lr 1e-5 \
  --panorama-context-mode world_exact_select \
  --observation-source direct --visual-precision fp16 \
  --dino-batch 64 --nwm-batch 64 \
  --compile-model --compile-backend inductor \
  --output data/logs/ghost_concat_persistent_compiled_10k_20260911
```

展开配置与旧 `ghost_concat_direct_compiled_10k_20260909` 实验逐字段对照：除了候选记忆模式与输出相关路径，计算参数一致。双卡总批量 8、随机种子 100、10000 次更新、每 200 步保存；第二阶段关闭。恢复状态沿用最近三份及每 2000 步保留。

训练机没有可用 `tmux` 命令，本次采用已有实验相同的独立后台进程方式：标准输入关闭，日志重定向，独立会话托管，训练命令本体不变。启动时间北京时间 **2026-09-11 14:31:45**，主管 PID `317695`，分布式父进程 `317703`，两张卡训练进程 `317716`、`317717`。后续按实际命令核验 PID，不能仅凭历史数字判断进程。

## 产物与查看

实验相对根目录 `data/logs/ghost_concat_persistent_compiled_10k_20260911/` 链接到数据盘 `/mnt/data2tb/ETP-R1_data/experiments/ghost_concat_persistent_compiled_10k_20260911/`。

- `train_launch.json`：启动命令、PID、源码提交、时间。
- `train_supervisor.log`：主管日志。
- `train/ghost_concat_v1_joint_persistent_train/manifest.json`：运行状态及完整参数。
- `train/ghost_concat_v1_joint_persistent_train/run.log`：训练日志与实际环境版本。
- `train/ghost_concat_v1_joint_persistent_train/checkpoints/ghost_concat_v1_joint_persistent_train/`：模型及 `train_states/` 恢复状态。

测评机仍处理旧实验队列。本次用户要求启动新训练，未启动新实验的完整测评队列。训练机保留模型；当前 `stream_ghost_checkpoints.py` 硬编码旧实验名，后续接入持久状态自动交付前须先适配名称，不能直接沿用旧交付命令。
