# RxR DINO 基座训练（2026-09-20）

## 目标与配置

在训练机 `/home/gwl/project/etpr1/ETP-R1` 使用 `etpnav_unified`，训练不使用世界模型的 RxR 导航基座。配置为 `run_rxr/iter_train_rae_dino_sft.yaml`，显式关闭 RAENWM、RGB 世界模型融合和 ACTIVE_LOOKAHEAD；导航网络参与训练。冻结 DINOv2 backbone，保留本工程已有的 768 维 CLS、可训练残差 MLP 和视觉投影。

从已完成的 DINO 联合预训练权重 `pretrained/r2r_rxr_ce/rae_dinov2_etpnav_cls_768_eval_final_best/model_best_step_465000.pt` 初始化，不加载 R2R 导航 checkpoint，也不使用原 CLIP 预训练权重。

原始参数以 `run_rxr/main_server.bash` 的 dagger 分支为准：30000 次更新、学习率 `1.5e-5`、预热 1000、最低学习率比例 0.6、监督权重 1.0、教师采样比例 0.75、衰减间隔 5000、ndtw 专家、航点增强开启、允许滑动、训练集 `_90`、每 200 步记录及保存。轨迹上限 25、文本上限 250、位置噪声 0.5 沿用原配置。

原脚本为 4 卡×6 环境；训练机实际为 2 张 A6000，因此沿用已有 RxR 管理脚本的 2 卡×6 环境×2 次梯度累积，保持每次更新总批量 24。累积方式并不保证与四卡逐位相同。保留工程已有的混合精度、fused AdamW 和可恢复训练状态，不改变上述学习率及采样计划。

环境实测：Python 3.10.14、PyTorch 2.2.2+cu121、CUDA runtime 12.1、Transformers 4.49.0、Habitat/Habitat-Sim 0.3.3。训练机重启后驱动为 580.178.04，不再加载旧版 580.173.02 隔离库。

## 验证记录

验证产物根目录（训练机）：`/mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/validation/`。

- `python -m pytest -q tests/test_rxr_native_cls_e24_workflow.py tests/test_online_rae_projection.py tests/test_rae_dinov2_encoder.py tests/test_rae_feature_hdf5.py tests/test_online_checkpoint.py tests/integration/test_rae_dinov2_parity.py tests/integration/test_training_stage_freeze.py tests/integration/test_pretrain_tasks_smoke.py`：退出 0，126 passed。包括真实 DINO 一致性、预训练 MLM/SAP 前后向和 GRPO 冻结规则检查；日志 `pytest.log`。
- `python precompute_img_features/extract_rae_dinov2_features.py --max_viewpoints 2 --output_file <验证目录>/two_views.hdf5`：退出 0，两个视点均生成 `(36,768)` 有限特征；日志 `hdf5.log`。这是既有离线特征格式检查，在线 RxR 相机仍保持原来的 63 度视场角。
- 训练数据检查：54,270 条 guide 指令、59 个场景全部存在；有效批量24、世界模型关闭；日志 `assets.log`。
- `ETPR1_RXR_SFT_RUN_ID=20260920_baseline bash scripts/manage_rae_rxr_sft_server.sh smoke` 与 `smoke-resume`：均退出 0，从第2步恢复到第3步，恢复了优化器、调度器和环境迭代状态。产物位于 `data/logs/rae_dinov2_etpnav_cls_768/rxr_sft_smoke/20260920_baseline/`。
- 起始缩放16384出现一次溢出，自动降至8192，优化器与调度器同步跳过该次更新。第2步时有效更新仍处于零学习率预热起点，因此最初权重变化审计未通过；第3步有效更新后重新审计退出0，视觉投影及残差MLP均有真实变化（`audit_resumed.log`）。保留最初诊断日志，不把它误记为训练通过证据。
- 正式批量短测命令：`ETPR1_RXR_SFT_EXP_NAME=rxr_dino_fullbatch_smoke ETPR1_RXR_SFT_OUTPUT_ROOT=<验证目录>/fullbatch ETPR1_RXR_SFT_ITERS=4 ETPR1_RXR_SFT_LOG_EVERY=2 bash scripts/manage_rae_rxr_sft_server.sh start`。退出0，2卡×6环境×2次累积完成4次更新，loss为3.924/3.998，调度器推进4步、缩放16384无溢出；GPU0峰值分配17185MiB。`ddp_state.log` 确认两卡随机数与环境状态、模型及训练状态的iteration=4一致。
- RxR 单回合：`bash scripts/run_rxr_single_episode_eval_server.sh baseline <单环境短测目录>/checkpoints/etpr1_rxr_rae_dino_sft_smoke/ckpt.iter3.pth <验证目录>/rxr_eval`，退出0，结果JSON有效。
- R2R 单回合：`python run.py --exp_name rxr_base_r2r_smoke --run-type eval --exp-config run_r2r/iter_train_rae_dino.yaml SIMULATOR_GPU_IDS '[0]' TORCH_GPU_IDS '[0]' GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 EVAL.CKPT_PATH_DIR <上述第3步模型> EVAL.EPISODE_COUNT 1 EVAL.SAVE_RESULTS True MODEL.pretrained_path <上述465000预训练权重> CHECKPOINT_FOLDER <验证目录>/r2r_eval/checkpoints/ TENSORBOARD_DIR <验证目录>/r2r_eval/tensorboard/ RESULTS_DIR <验证目录>/r2r_eval/results/`，退出0。两次导航仅验证运行完整，不用于估计训练后成功率；短训模型两回合均未成功到达目标。

## 正式命令

```bash
cd /home/gwl/project/etpr1/ETP-R1
export ETPR1_RXR_SFT_EXP_NAME=rxr_dino_baseline_20260920
export ETPR1_RXR_SFT_OUTPUT_ROOT=/mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/train
bash scripts/manage_rae_rxr_sft_server.sh start
```

同样两个环境变量配合 `status`/`logs` 查看，`resume` 恢复。由已有管理脚本通过 `nohup setsid` 托管，SSH 断开不会结束训练。每200步保存模型，训练状态保留最近3份及每5000步里程碑。测评机当前其他测评继续运行；本任务不向其同步 checkpoint 或启动测评队列。

正式训练使用代码提交 `7bdddd7`，2026-09-20 10:07:17（北京时间）启动，管理进程PID为7718，torchrun为7789；日志为 `train/supervisor_server/start_20260920T100717.log`。这两个PID仅对应启动现场，后续应使用管理脚本查询。展开配置审计 `validation/formal_config_audit.json` 全部通过。

双卡短测模型为1,529,057,152字节，训练状态为3,014,492,264字节；正式150份模型加保留的训练状态约260GB（十进制）以内，启动时数据盘仍有567GiB可用。后续如同盘增加其他实验，需要重新核对余量。

2026-09-20 10:28:40完成首个200步区间，训练损失均值2.362，学习率3e-6；模型 `train/checkpoints/rxr_dino_baseline_20260920/ckpt.iter200.pth` 和对应 `train_states/train_state.iter200.pth` 均原子保存成功。只读CPU加载审计退出0：模型张量全部有限，模型/训练状态iteration均为200，优化器与调度器均推进200次，479组优化器状态齐全，两卡随机状态及环境迭代状态齐全，缩放16384保持不变（正式前200步无溢出），世界模型关闭。证据为 `validation/formal_checkpoint_audit.json` 和同名 `.log`。

10:29左右复核时，后台父进程7718、torchrun 7789及双卡子进程7822/7823均存活，日志已推进至全局第209步；两卡利用率71%/88%、显存约26.7/27.0GiB。任务继续运行，未重启或改变正式参数。前200步用时20分47秒，后续时长会随路线和采样比例变化，不据此保证总完成时间。
