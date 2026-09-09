# 直接渲染与编译版 10000 步长训练

2026-09-09 晚间按用户要求重新启动长训练。使用已合并到主工作区的选定版本 `5f5df39`，保留直接定向渲染和 Inductor 世界模型编译，没有合入已放弃的静态条件缓存或导航 SDPA。

## 参数和目录

- 从 `pretrained/active_lookahead/base_iter14200.pth` 重新初始化，SHA-256 为 `1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61`；不接续已停止旧任务的优化器。
- 10000 次更新，每 200 步保存，共 50 份模型。双 RTX A6000，每卡 4 环境，总批量 8；策略学习率 2e-6，融合层学习率 1e-5。
- `world_exact_select`、直接历史位姿渲染、FP16 定向编码、DINO/世界模型批次上限 64，显式开启 `--compile-model --compile-backend inductor`。
- 实验相对目录：`data/logs/ghost_concat_direct_compiled_10k_20260909/`。训练机通过软链接写入 `/mnt/data2tb/ETP-R1_data/experiments/ghost_concat_direct_compiled_10k_20260909/`。
- 测评机使用同名实验目录、专用 `gwl-etpr1-rae` 容器和 `etpr1_rae` 环境；实际 GPU 为 RTX 3090 24 GB，8 环境，逐份完整评测 R2R `val_unseen` 的 1839 条路线。

训练机实际命令：

```bash
python3 scripts/ghost_concat_job.py train --train-policy --gpus 0,1 --batch 4 --iters 10000 --log-every 200 --policy-lr 2e-6 --fusion-lr 1e-5 --panorama-context-mode world_exact_select --observation-source direct --visual-precision fp16 --dino-batch 64 --nwm-batch 64 --compile-model --compile-backend inductor --output data/logs/ghost_concat_direct_compiled_10k_20260909
python3 scripts/stream_ghost_checkpoints.py --output data/logs/ghost_concat_direct_compiled_10k_20260909 --iters 10000 --every 200
```

测评机宿主机实际命令：

```bash
python3 scripts/ghost_concat_job.py watch --machine eval --train-policy --gpus 0 --environments 8 --panorama-context-mode world_exact_select --observation-source direct --visual-precision fp16 --dino-batch 64 --nwm-batch 64 --compile-model --compile-backend inductor --ready-timeout 1209600 --output data/logs/ghost_concat_direct_compiled_10k_20260909 --eval-iterations "$(seq -s, 200 200 10000)"
```

三个任务通过独立后台进程托管，启动 PID 分别为训练 257802、传送 257803、测评等待 1915768；训练分布式父进程 257810、两个训练进程 257813/257814。PID 仅作为本次记录，后续必须检查进程命令再使用。

训练器内部异步同步关闭，由独立传送程序逐个发布模型。每个模型完成测评并校验 SHA-256 后释放测评机临时副本，训练机保留全部模型，测评结果保留。恢复状态沿用最近三份和每 2000 步保留的配置。

## 预检与测评机准备

训练机先用相同优化配置完成两次真实更新，保存模型/恢复状态完整配对，退出 0；预检模型 SHA-256 为 `8bc1404ddbb238fff3abefa1f95879bc07cc17950debe757c292547d75714e30`，经专线同步后由测评入口再次校验一致。记录位于 `preflight/` 和 `preflight_train.log`。

测评机原来没有 GCC/G++。在联网笔记本使用测评容器的 dpkg 状态解析 Ubuntu 22.04 依赖，下载并验证 38 份软件包，经 SSH 跳板交付后离线安装到本工程专用容器。安装 G++ 11.4 及依赖，包含必要的容器系统运行库更新；没有修改受保护的 `raenwm` conda 环境或 ETPNav 容器。软件包、SHA-256 清单和安装日志保留在测评机实验目录的 `toolchain/`，`dpkg --audit` 通过。

另修复数字 UID 容器的编译缓存路径：启动脚本显式设置项目内 `TORCHINDUCTOR_CACHE_DIR` 和 `TRITON_CACHE_DIR`；Triton 2.2 的辅助 dump 路径仍采用 `/.triton`，因此在专用容器中将它链接到项目 `.runtime/triton_aux`。若重建该容器，需要恢复编译工具及这个辅助目录链接。此调整不改变模型计算；训练进程启动时仍使用原启动环境的缓存路径。

测评机相关编译/直接渲染测试为 7 passed，退出 0。最终真实预检使用两步模型、相同直接渲染/FP16/编译配置、8 环境，完成 8 个 episode 并验证结果文件，退出 0。成功记录为 `preflight_eval_final/eval/ghost_concat_v1_joint_eval/manifest.json`；此前用户名查询与 Triton 辅助目录权限失败的日志保留，不作为成功结果。

## 状态查看

实验根的 `train_launch.json`、`delivery_launch.json`、`eval_launch.json` 保存精确启动参数。训练日志为 `train/ghost_concat_v1_joint_train/run.log`，模型和状态目录为 `train/ghost_concat_v1_joint_train/checkpoints/ghost_concat_v1_joint_train/`。

传送日志为 `delivery_supervisor.log`，测评等待日志为 `eval_supervisor.log`；测评总表为 `eval_summary.json`，逐模型结果位于 `eval/ghost_concat_v1_joint_eval_iter<步数>/`。每次运行日志都记录实际 Python、PyTorch、CUDA、Transformers、Habitat/Habitat-Sim 版本。

正式训练已确认从 0 持续推进、父子进程存活、日志增长、双卡有计算负载。首份正式模型在第 200 步产生，后续状态以这些运行记录为准。

启动验收已实际闭环到第 200 步：模型 1,552,669,654 字节、恢复状态 3,051,065,418 字节；模型 SHA-256 `89a72d186f13d2a00789c24000be5a95d435dbac6cf832cd0db2b3636886bb96` 与测评机加载记录一致。状态包含 iteration=200、485 份优化器参数状态、4 个参数组、调度器 last_epoch=199、AMP scale=8192；调度器按现有逻辑随 AMP 成功的优化器更新推进。第 200 步日志的 IL loss=0.695、融合梯度范数=0.169，均有限。

测评机已经启动该模型的完整 1839 episode 评测，并出现实际完成路线的进度；训练继续超过第 200 步。校验记录为 `first_checkpoint_verified.json`、`first_state_audit.json`，首份正式评测 manifest 位于 `eval/ghost_concat_v1_joint_eval_iter200/manifest.json`。这只表示训练与测评链路已启动验收，不表示整个长任务已完成。
