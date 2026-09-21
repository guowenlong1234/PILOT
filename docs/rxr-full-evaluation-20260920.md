# RxR 第2200步模型双卡全量测评（2026-09-20）

## 完成结果（2026-09-21核查）

任务于2026-09-20 19:24:00正常结束，退出码0；从17:59:18启动计约84分42秒。两卡各5503条，交集为空，合并的11006个episode ID与原val_unseen guide数据集完全一致；数值均有限，按逐路线结果重算的均值与汇总一致。

| 指标 | 结果 |
|---|---:|
| 成功率 SR | 51.3538%（5652/11006） |
| SPL（兼顾成功与路径效率） | 42.7064% |
| nDTW（路线匹配程度） | 60.1102% |
| SDTW | 42.7691% |
| 曾进入成功范围的比例 OSR | 58.8043%（6472/11006） |
| 最终距目标 | 6.4125米 |
| 路径长度 | 16.7136米 |

按语言重新汇总：

| 语言 | 条数 | SR | SPL | nDTW |
|---|---:|---:|---:|---:|
| en-IN | 2446 | 51.594% | 42.857% | 60.093% |
| en-US | 1223 | 50.777% | 40.874% | 58.505% |
| hi-IN | 3669 | 52.385% | 43.922% | 60.926% |
| te-IN | 3668 | 50.354% | 42.001% | 59.841% |

测评链路、覆盖范围和数值检查均正常，各语言表现较均衡。该2200步模型完成了约一半路线；OSR比SR高约7.45个百分点，反映部分路线曾接近目标但最终未停在成功范围内。只有一个早期检查点，尚不能判断是否收敛或优于同设置基线；需后续检查点做同条件比较。

以下保留原启动记录，原文中的进度与时间估计是启动现场信息。

用户要求先对最新已保存模型进行完整RxR测评，启动后短暂监控并给出预计完成时间。本次使用训练机主工作区 `/home/gwl/project/etpr1/ETP-R1` 和 `etpnav_unified`，两张A6000各运行6个环境。

最新完整模型是原正式训练的 `ckpt.iter2200.pth`；此前吞吐基准没有保存新的训练模型。

测评范围为完整 `val_unseen` guide：11006条不同episode、11个场景，四语言分布为hi-IN 3669、te-IN 3668、en-IN 2446、en-US 1223。两卡按episode ID分片，各5503条，最终按实际条数加权汇总。

采用原RxR测评规则：`ALLOW_SLIDING=False`、`IL.back_algo=control`、`EVAL.fast_eval=False`、`EVAL.EPISODE_COUNT=-1`；模型贪心选动作，不使用教师动作。使用原DINO基座配置，世界模型与二阶段关闭，视觉编译关闭。

入口：`scripts/run_rxr_full_eval_server.sh <checkpoint> <new-output-root>`。通过 `nohup setsid` 后台托管，SSH断开后继续运行；两个rank的输出分别写入torchrun日志。

首次启动在设备初始化阶段退出1，尚未测评路线。原因是旧的双卡测评入口把 `self.device` 设为整数，而策略初始化需要 `torch.device.index`。已以 `48b0ae2` 修复类型转换；失败日志保留在不带 `_retry1` 的目录。

当前任务输出根目录（训练机）：

```text
/mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/eval_iter2200_dual_20260920_retry1/
```

- `metadata.log`：启动时间、PID、模型、提交和实际Python/PyTorch/CUDA/Transformers/Habitat版本。
- `launcher.log`：torchrun管理日志。
- `torchrun/*/attempt_0/{0,1}/{stdout,stderr}.log`：两个rank的模型加载、进度与异常日志。
- `exit_code`、`finished_at`：完成时写入。
- `results/rxr_baseline_full_eval/eval_results/stats_ckpt_2200_val_unseen.json`：最终汇总指标，包括SR、SPL、nDTW、距离及路径长度等。
- 同一结果目录的 `stats_ep_ckpt_2200_val_unseen_r{0,1}_w2.json`：逐路线结果。

实际命令为：

```bash
cd /home/gwl/project/etpr1/ETP-R1
nohup setsid bash scripts/run_rxr_full_eval_server.sh \
  /mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/train/checkpoints/rxr_dino_baseline_20260920/ckpt.iter2200.pth \
  /mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/eval_iter2200_dual_20260920_retry1 \
  > /mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/eval_iter2200_dual_20260920_retry1.supervisor.log \
  2>&1 < /dev/null &
```

18:01:53（北京时间）短暂监控：监督进程33591、torchrun 33605及两卡工作进程存活，日志持续增长。rank0完成153/5503、rank1完成140/5503；各计时约125秒，速度约1.224/1.120条/秒。两卡显存约11.1GiB，采样利用率均100%；日志未见新的Traceback、Error或显存不足。

按较慢rank的累计速度线性估算，剩余约80分钟，预计当天19:20左右完成；考虑场景及路线长度差异，保守预留至19:50。这是启动阶段估计，不是保证完成时间。完整SR/SPL/nDTW尚未产生，当前只能确认测评正常推进。任务后台继续运行，离开会话不终止测评。
