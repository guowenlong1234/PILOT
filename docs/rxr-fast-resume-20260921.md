# RxR 从2200步继续训练（2026-09-21）

在2200步完整RxR测评结束后，按用户要求继续训练。实际运行位置为训练机 `/home/gwl/project/etpr1/ETP-R1`，环境 `etpnav_unified`。2026-09-21 10:28:08后台启动，监督进程55217；两个rank均于10:29:06确认恢复至2200步。

## 恢复内容与边界

- 实验名仍为 `rxr_dino_baseline_20260920`，目标仍为30000步。
- 完整恢复2200步模型、AdamW参数状态、学习率调度器、混合精度缩放状态和两卡主进程随机状态。不是只加载权重重新训练。
- 启动前CPU审计：模型/状态/调度器均为2200，479份优化器参数状态的step均为2200；学习率1.4974686872186908e-5；缩放因子32768、增长计数200；模型张量均有限。
- 使用已验证的快速配置：2卡×12环境×1次累积，总批量仍24；视觉编译、导航重算及其他保留优化开启。
- 原采样队列为每卡6环境，不能逐位迁移为12环境。通过显式 `IL.allow_env_count_change_on_resume=True` 重建环境队列；原checkpoint没有被修改。模型、优化器等恢复照常执行，主进程随机状态仍恢复。
- 该开关仅在环境数确实变化且旧状态结构有效时生效；同环境数时仍执行完整队列恢复，不允许改变训练rank数，也不掩盖损坏状态。默认关闭。
- 教师概率起点0.75、衰减间隔5000、偏移0及阈值0.15保持原样；全局iteration从2200继续，当前概率0.75，不重置或提前推进教师计划。
- 学习率、预热1000、最低比例0.6、200步记录/保存、轨迹上限25、文本上限250、训练数据及语言等22项关键设置均与原展开配置一致。

## 产物位置与操作

新产物根目录（训练机）：

```text
/mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/train_fast_resume_20260921/
```

原 `train/` 目录与2200步测评结果保留。新目录通过硬链接引入完整的2200步模型/训练状态对，避免重复复制；后续检查点、状态清理和日志仅作用于新目录。

实际启动命令：

```bash
cd /home/gwl/project/etpr1/ETP-R1
export ETPR1_RXR_SFT_EXP_NAME=rxr_dino_baseline_20260920
export ETPR1_RXR_SFT_OUTPUT_ROOT=/mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/train_fast_resume_20260921
export ETPR1_RXR_SFT_ALLOW_ENV_COUNT_CHANGE_ON_RESUME=True
bash scripts/manage_rae_rxr_sft_fast_server.sh resume
```

同样的实验名和输出目录配合 `status`、`logs` 查询。后续从已保存的12环境状态恢复时不需要开启迁移开关；即使开关仍为True，相同环境数也会恢复原队列。

日志：`supervisor_server/resume_20260921T102808.log`，`supervisor_server/latest.log` 指向它。检查点：`checkpoints/rxr_dino_baseline_20260920/`。

使用已有的nohup/setsid管理，断开SSH后继续运行。实际版本：Python3.10.14、PyTorch2.2.2+cu121、CUDA12.1、Transformers4.49.0、Habitat/Habitat-Sim0.3.3。

## 验证

- `python -m pytest -q tests/test_resume_environment_resize.py tests/test_online_checkpoint.py tests/test_rxr_native_cls_e24_workflow.py`：退出0，27 passed。
- `python -m pytest -q tests/test_episode_iterator_state.py tests/test_rae_checkpoint.py`：退出0，68 passed。
- 已核实实际展开配置中的视觉编译、导航重算、异步有限值检查与并行教师查询均开启；总批量24、教师概率0.75、衰减间隔5000。
- 源状态审计：实验根 `resume_audit_20260921.json`。
- 22项不变配置及新批量/目录核对：实验根 `resume_config_audit_20260921.json`。
- 日志确认两卡均恢复AdamW、2200步模型，并显式记录6→12环境迁移；实际采样概率0.75。
- 首个新增2400步检查点尚在等待验收。
