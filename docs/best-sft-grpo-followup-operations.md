# 两轮 SFT 选优后自动接力一次 GRPO

入口脚本是 `scripts/manage_best_sft_grpo_followup_server.sh`，部署并运行在训练机。
它通过训练机到测评机的 2.5 GbE 直连链路完成以下流程：

1. 严格等待两轮 SFT 各 75 份完整的 R2R `val_unseen` 结果。
2. 校验结果 JSON 和对应的非空 SFT checkpoint。
3. 在两轮合计 150 个候选中，按 `success + spl` 选择唯一的全局最佳模型。
4. 用该模型在训练机的两张 A6000 上只启动一次 DINO-GRPO。
5. 将 GRPO 模型 checkpoint 原子同步到测评机，并由 4090 串行完整评测。
6. 等 GRPO 正常完成、100 个 checkpoint 全部同步且产生有效结果后结束。

`success` 就是 SR（成功率）。并列时依次比较 SPL、SR、iteration 和轮次名，
最终选择会保存在监控目录的 `selection.json`，不会做哈希绑定或哈希校验。

## 启动和查看

```bash
ssh server
cd /home/gwl/project/etpr1/ETP-R1
scripts/manage_best_sft_grpo_followup_server.sh start
scripts/manage_best_sft_grpo_followup_server.sh status
scripts/manage_best_sft_grpo_followup_server.sh tail
```

监控默认每 60 秒检查一次。第二轮 SFT 评测未完成、训练机 GPU 忙、测评机
GPU 忙或磁盘空间不足都不会触发重复训练；条件满足后会自动继续。

默认 GRPO 保持正式入口的原始保存策略：总计 1000 次更新，`log_every=10`，
因此会保存、同步并评测 100 个模型 checkpoint。训练状态默认保留最近 3 份，
并每 250 次保留一个里程碑。磁盘门槛只负责等待，不会改变保存间隔，也不会
删除 checkpoint。默认要求测评机在容纳预计 checkpoint 后仍至少剩余 8 GiB，
训练机至少预留 30 GiB。

需要改变存储门槛或明确指定已经迁移后的目录时，可在启动时覆盖：

```bash
ETPR1_BEST_SFT_GRPO_EVAL_MIN_FREE_GIB=8 \
ETPR1_BEST_SFT_GRPO_TRAIN_MIN_FREE_GIB=30 \
ETPR1_BEST_SFT_GRPO_OUTPUT_ROOT=data/logs/my_grpo \
ETPR1_BEST_SFT_GRPO_REMOTE_GRPO_CKPT_DIR=/home/a6000/gwl/ETP-R1/data/logs/my_grpo/checkpoints/my_grpo \
ETPR1_BEST_SFT_GRPO_EXP_NAME=my_grpo \
scripts/manage_best_sft_grpo_followup_server.sh start
```

同一次工作流的环境变量应保持一致，尤其是实验名、输出目录、总迭代数和
保存间隔。

## 停止与恢复

```bash
scripts/manage_best_sft_grpo_followup_server.sh stop
```

`stop` 只停止总监控，不会连带终止已启动的 GRPO 或测评任务。若 GRPO 异常
退出，总监控会保持等待并报告 `grpo_not_successfully_complete`，不会擅自从
错误状态重复启动。修复外部问题后，使用同一组环境变量执行：

```bash
scripts/manage_best_sft_grpo_followup_server.sh resume
```

该命令复用持久化的 SFT 选择结果，重新保证测评 watcher 存活，再调用正式
GRPO 管理入口的完整断点恢复。优化器、学习率调度器、随机状态和两个 rank
的环境队列都从最新完整模型/训练状态文件对恢复。总监控继续运行时，会在
恢复训练完成后自动接着等待同步与测评。

主要状态文件位于：

```text
data/logs/rae_dinov2_etpnav_cls_768/best_sft_grpo_followup_monitor/
  selection.json
  launched.env
  completed.env
  watch.log
```
