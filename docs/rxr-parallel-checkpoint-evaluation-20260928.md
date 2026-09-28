# RxR 全检查点双机测评（2026-09-28）

用户要求在训练机和测评机启动RxR检查点测评并充分使用计算资源。本轮覆盖`rxr_dino_baseline_20260920`从200到30000步、每200步一份的150个模型。2200步已有完整11006路线结果，启动前重新核对两份分片ID无重复、无缺失、退出0，因此保留该结果；新队列测其余149份，按步数从高到低领取。

## 执行方式

- 训练机主工作区`/home/gwl/project/etpr1/ETP-R1`，环境`etpnav_unified`，两张A6000各一个独立单卡进程、各8个模拟环境。
- 测评机主工作区`/home/a6000/gwl/ETP-R1`，专用容器`gwl-etpr1-rae`和环境`etpr1_rae`，单卡4环境。指定专线入口当前实际GPU为RTX3090 24GB。
- 三个工作进程共用训练机上的有限队列，空闲后领取下一个检查点。不是预先把模型平均分成三份，较快机器可以多做。
- 训练机监督进程通过`start_new_session=True`脱离SSH，PID **260790**，北京时间 **2026-09-28 10:04:04** 启动。PID只是启动记录，后续需同时核对命令和状态。
- 首轮训练机GPU0测30000步，GPU1测29800步，测评机测29600步；还有146份尚未领取。
- 源码三机均为`ed4ab132f8805bb6dc92d21ae0659a128508d9ff`；通过中央Git快进同步，未强制覆盖工作区。

入口是`scripts/run_rxr_checkpoint_queue.py queue`，每个模型使用`scripts/run_rxr_checkpoint_eval.sh`。精确启动命令与监督PID已写入训练机输出根的`launch.json`，各模型的实际测评命令在自身目录`command.sh`。

## 测评合同

完整RxR `val_unseen` guide、四语言、11006路线，`fast_eval=False`、`ALLOW_SLIDING=False`、`IL.back_algo=control`；使用DINO导航基座，不启用世界模型或二阶段，不修改已有模型。与9月20日2200步测评保持相同导航规则。不同机器、环境数量及此次合并后的代码版本仍会有数值差异，结果记录实际机器与源码版本。

两机验证集SHA均为`a110036736d0d7a3e4e899dbc6f2954a6ad349e80d442854aae7460940414151`。

每个模型运行结束后，要求退出0，逐路线ID完整且无重复、指标有限、逐路线均值与汇总一致，才记录completed并继续。失败会记录具体模型与退出码、停止该工作通道，不自动反复重试；其他通道继续领取余下任务。全部成功才把总队列标记completed。当前队列不支持隐式原地恢复；需要恢复时先审计已完成、失败和仍运行任务，避免重复启动。

## 传输及磁盘

测评机空间不足以一次保存全部模型。训练机经`a6000@10.10.10.2`逐份传输到本轮独立`staging/`，SHA-256一致后原子改名再测评。成功并取得完整结果验收记录后，仅删除该队列刚传入的临时模型；训练机原件、远端逐路线结果与日志保留。失败模型留在现场。远端每轮检查至少20GiB剩余空间，训练机输出盘至少10GiB。

模型及原始路线结果不回传笔记本。监督进程只从测评机读取小型`validation.json`，用于统一汇总状态和指标。

## 路径与监控

训练机输出根：

```text
/mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/eval_parallel_20260928/
```

测评机输出根：

```text
/home/a6000/gwl/ETP-R1/data/logs/rxr_eval_parallel_20260928/
```

训练机`queue.json`记录待测列表、通道、每份模型SHA、输出位置、退出码和已完成指标；`server0.log`、`server1.log`、`eval.log`保存调度与传输命令。每份目录有`launcher.log`、`metadata.log`、`supervisor.pid`、`command.sh`，完成时生成`exit_code`、`finished_at`及验收通过后的`validation.json`。原始路线及汇总位于`results/*/eval_results/`。

```bash
ssh server 'cat /mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/eval_parallel_20260928/queue.json'
ssh server 'nvidia-smi'
ssh -J server a6000@10.10.10.2 'nvidia-smi'
```

## 启动验证

- 两机分别使用30000步模型完成16路线真实短测，退出0，路线与指标验收通过。训练机8环境，测评机4环境；短测路线子集受环境分配影响，不把两次短测指标直接比较。
- 两机各执行`test_rxr_checkpoint_queue.py`和`test_graph_geometry_domain.py`，各 **12 passed**、退出0。先前直接调用Python的检查缺少项目运行时路径；加入对应运行时后通过。测评机首次短测只在`whoami`记录身份时报数字UID缺少用户名，改用`id`后重跑通过，失败目录保留。
- 训练机：Python3.10.14；测评机：Python3.11.15。两机均PyTorch2.2.2+cu121、CUDA12.1、Transformers4.49.0、Habitat/Habitat-Sim0.3.3。每份测评均记录实际版本。
- 启动时两机GPU空闲，受保护ETPNav容器已经停止，没有结束其他任务。正式启动后已核对监督进程、两卡主进程及远端子进程存活；三路加载成功并进入11006路线测评。训练机两卡采样利用率100%/90%，测评机68%，日志无Traceback或运行时异常。后续进度见实时日志，不把启动验收当作全量完成。
