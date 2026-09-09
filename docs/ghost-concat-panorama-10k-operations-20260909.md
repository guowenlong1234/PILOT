# 新全景上下文：10000 步联合训练与连续测评

用户于 2026-09-09 授权按上一轮 2000 步任务的参数，从 14200 检查点重新初始化，启用新上下文方向管理，训练 10000 步并测评全部检查点。

## 参数与目录

- 基座：`pretrained/active_lookahead/base_iter14200.pth`，SHA-256 `1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61`；重新开始第 1 步，没有接续旧实验的优化器。
- 双 RTX A6000，每卡 4 环境，总批量 8；策略学习率 `2e-6`，融合层学习率 `1e-5`；其余训练参数沿用 `ghost_concat_joint_bs8_20260908`。
- 明确指定 `world_exact_select`，总步数 10000，每 200 步保存，保留全部模型；恢复状态沿用最近 3 份并每 2000 步保留的配置。
- 测评机实际硬件为 RTX 3090 24 GB，使用 `gwl-etpr1-rae` / `etpr1_rae`，8 环境，每份模型完整测评 R2R `val_unseen` 的 1839 条路线。
- 两端工程相对目录：`data/logs/ghost_concat_panorama_bs8_10k_20260909`。训练机该目录是软链接，实际产物写入 `/mnt/data2tb/ETP-R1_data/experiments/ghost_concat_panorama_bs8_10k_20260909`，避免写满系统盘。

## 启动命令

训练机项目目录：

```bash
python3 scripts/ghost_concat_job.py train --train-policy --gpus 0,1 --batch 4 --iters 10000 --log-every 200 --policy-lr 2e-6 --fusion-lr 1e-5 --panorama-context-mode world_exact_select --output data/logs/ghost_concat_panorama_bs8_10k_20260909
python3 scripts/stream_ghost_checkpoints.py --output data/logs/ghost_concat_panorama_bs8_10k_20260909 --iters 10000 --every 200
```

测评机宿主机项目目录：

```bash
python3 scripts/ghost_concat_job.py watch --machine eval --train-policy --gpus 0 --environments 8 --panorama-context-mode world_exact_select --ready-timeout 1209600 --output data/logs/ghost_concat_panorama_bs8_10k_20260909 --eval-iterations "$(seq -s, 200 200 10000)"
```

三项任务均由独立后台进程托管，实际参数与 PID 分别写入 `train_launch.json`、`delivery_launch.json` 和 `eval_launch.json`，日志分别为 `train_supervisor.log`、`delivery_supervisor.log` 和 `eval_supervisor.log`。启动时训练 PID 为 229217、传送 PID 为 229387、测评 watcher PID 为 1890807；PID 只作为本次记录，后续必须核对命令再使用。

训练器内部的批量异步同步关闭，改由独立传送程序经 `10.10.10.2` 逐份发布：等待完整模型原子落盘，传送一份，等待该模型测评成功并核对 SHA-256 / 1839 条结果，再删除测评机临时模型副本。训练机原件和全部测评结果保留。训练不会等待测评，积压模型保存在训练机大容量数据盘。

## 验收与排查

代码版本 `785b33c`；两端通过 Git 快进同步。上一轮训练机当前模型代码的 559 项回归和两步真实联合更新已通过；本轮测评机补跑 37 项相关测试通过。逐份传送程序在训练机验证了完成状态处理、原件保留和错误校验和拒绝删除；真实预检模型也已通过专线同步。

`parameter_comparison.json` 记录与旧任务的逐项差异：只有朝向模式、总步数、输出目录与同步方式改变，基座校验和一致。训练实际日志为 `train/ghost_concat_v1_joint_train/run.log`，模型目录为 `train/ghost_concat_v1_joint_train/checkpoints/ghost_concat_v1_joint_train/`。

测评摘要为 `eval_summary.json`，逐份结果位于 `eval/ghost_concat_v1_joint_eval_iter<步数>/`。等待首份第 200 步模型时 watcher 不占用 GPU；后续实际评测前仍检查 GPU 与受保护的 ETPNav 容器。

初次测评预检使用了没有上下文元数据的原始基座，触发了检查点命名/元数据保护。随后改用已有两步全景联合模型 `preflight_model/ckpt.iter2.pth` 验证正式模型的测评合同，不放宽加载保护。

正式全景模型的测评机 8 环境检查退出 0：3 次有效调用、79 个查询、36 个倒序参考端点，全部目标位置/朝向断言通过。结果在 `preflight_panorama_model/report.json`。测评环境版本已记入 `versions.log`：Python 3.11.15、PyTorch 2.2.2+cu121、CUDA 构建 12.1、Transformers 4.49.0、Habitat/Habitat-Sim 0.3.3。

用户追加授权清理已测评且训练机有对应原件的旧副本。清理逐个检查完整结果、训练机原件及双端 SHA-256；记录在训练机实验目录的 `eval_cleanup.json` 与 `eval_cleanup_legacy.json`。预训练/初始化资源不属于这批清理范围。

最终清理 121 份（17 份由 manifest 验证、104 份由成功测评日志与完整结果验证），共 188,538,536,918 字节，即 175.59 GiB。删除后独立复查全部训练机原件仍存在、全部对应测评机副本已不存在，结果保留；测评机磁盘可用约 221 GiB。汇总为 `cleanup_verified.json`。

启动验收时训练已持续完成至少 19 次更新，两卡 GPU 利用率为 95%–100%，显存约 15.8/16.6 GB；训练与分布式子进程存活，日志持续增长，配置已写入新实验模型目录。首份正式检查点设在第 200 步，验收时尚未产生，因此新实验的正式测评结果尚未产生，传送和测评 watcher 都正常等待第 200 步。
