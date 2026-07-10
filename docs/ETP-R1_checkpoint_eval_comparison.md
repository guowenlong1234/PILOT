# ETP-R1 四个 checkpoint 本地评测与论文结果对比报告

评测工程：`/home/gwl/project/etpr1/ETP-R1`

论文 PDF：`/home/gwl/project/etpr1/dataset/ETP_r1.pdf`

评测日期：2026-06-08

## 1. 实验环境摘要

本次评测全部在现有 `etpnav` 环境中执行。当前交互 shell 本身是 `base`，但所有评测命令都通过 `conda run -n etpnav ...` 进入 `etpnav` 后运行；在该环境内确认 `CONDA_DEFAULT_ENV=etpnav`。

| 项目 | 本地记录 |
|---|---|
| 工作目录 | `/home/gwl/project/etpr1/ETP-R1` |
| git 状态 | `## main...origin/main`，开始检查时仅有 `?? research.md` |
| conda 环境 | `etpnav` |
| PyTorch | `1.9.1+cu111` |
| PyTorch CUDA | `11.1` |
| CUDA 可用 | `True` |
| 可见 GPU 数量 | 1 |
| GPU | `NVIDIA RTX A6000`, 49140 MiB |
| NVIDIA driver | `580.159.03` |
| Habitat | `0.1.7`, `/home/gwl/project/DGNav/habitat-lab/habitat/__init__.py` |
| Habitat-Sim | `0.1.7`, `/home/gwl/miniconda3/envs/etpnav/lib/python3.7/site-packages/habitat_sim/__init__.py` |
| MP3D 路径 | `data/scene_datasets/mp3d` -> `/home/gwl/project/dataset/mp3d_unzipped/mp3d` |
| MP3D 场景 | `find -L data/scene_datasets/mp3d -name '*.glb'` 统计到 90 个 `.glb` |

## 2. 开始前检查

| 检查项 | 结果 |
|---|---|
| R2R DAgger checkpoint | 存在：`data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth` |
| R2R GRPO checkpoint | 存在：`data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth` |
| RxR DAgger checkpoint | 存在：`data/logs/checkpoints/release_rxr_dagger/store/ckpt.iter20600.pth` |
| RxR GRPO checkpoint | 存在：`data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth` |
| `run_r2r/iter_train.yaml` | 可解析；trainer 为 `SS-ETP-R1`，base 配置为 `run_r2r/r2r_vlnce.yaml`，eval split 为 `val_unseen`，episode_count 为 `-1` |
| `run_rxr/iter_train.yaml` | 可解析；trainer 为 `SS-ETP-R1`，base 配置为 `run_rxr/rxr_vlnce.yaml`，eval split 为 `val_unseen`，episode_count 为 `-1` |
| 评测逻辑 | 使用项目已有 `run.py` 和 `run_r2r` / `run_rxr` 配置组织方式；未修改模型、指标、奖励、episode 过滤、动作空间或导航决策逻辑 |

说明：这里的 `episode_count=-1` 表示正式评测使用完整 split，不是抽样评测。正式 R2R `val_unseen` 为 1839 个 episode，正式 RxR `val_unseen` 为 11006 个 episode。

## 3. 正式评测命令

四次正式评测均使用单 GPU：`CUDA_VISIBLE_DEVICES=0`、`SIMULATOR_GPU_IDS [0]`、`TORCH_GPU_IDS [0]`、`GPU_NUMBERS 1`。这是对项目脚本默认多 GPU 写法的本地单卡适配，只改变并行方式，不改变评测语义。

### 3.1 R2R SFT / DAgger

```bash
conda run -n etpnav env CUDA_VISIBLE_DEVICES=0 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  python -m torch.distributed.launch --nproc_per_node=1 --master_port 24311 run.py \
  --exp_name manual_eval_r2r_dagger \
  --run-type eval \
  --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 8 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth \
  IL.back_algo control \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
```

### 3.2 R2R RFT / GRPO

```bash
conda run -n etpnav env CUDA_VISIBLE_DEVICES=0 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  python -m torch.distributed.launch --nproc_per_node=1 --master_port 24312 run.py \
  --exp_name manual_eval_r2r_grpo \
  --run-type eval \
  --exp-config run_r2r/iter_train.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 8 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_grpo/store/ckpt.iter270.pth \
  IL.back_algo control \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
```

### 3.3 RxR SFT / DAgger

```bash
conda run -n etpnav env CUDA_VISIBLE_DEVICES=0 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  python -m torch.distributed.launch --nproc_per_node=1 --master_port 24323 run.py \
  --exp_name manual_eval_rxr_dagger_env6 \
  --run-type eval \
  --exp-config run_rxr/iter_train.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 6 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_rxr_dagger/store/ckpt.iter20600.pth \
  IL.back_algo control \
  IL.RECOLLECT_TRAINER.gt_file data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
```

### 3.4 RxR RFT / GRPO

```bash
conda run -n etpnav env CUDA_VISIBLE_DEVICES=0 GLOG_minloglevel=2 MAGNUM_LOG=quiet \
  python -m torch.distributed.launch --nproc_per_node=1 --master_port 24325 run.py \
  --exp_name manual_eval_rxr_grpo_env3 \
  --run-type eval \
  --exp-config run_rxr/iter_train.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 3 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING False \
  EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_rxr_grpo/store/ckpt.iter1320.pth \
  IL.back_algo control \
  IL.RECOLLECT_TRAINER.gt_file data/datasets/RxR_VLNCE_v0_enc_xlmr/{split}/{split}_{role}_gt.json.gz \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/mlm.sap_habitat_depth/store2/model_step_367500.pt
```

## 4. 本地正式评测结果

表中百分数指标已经从 JSON 的 0 到 1 小数换算为百分数。NE 是到目标点距离，越低越好；SR、OSR、SPL、nDTW、SDTW 越高越好。

| 任务 | 阶段 | checkpoint | split | episodes | 开始时间 | 结束时间 | 耗时 |
|---|---|---|---|---:|---|---|---:|
| R2R-CE | SFT / DAgger | `ckpt.iter25000.pth` | `val_unseen` | 1839 | 2026-06-08 13:55:16 CST | 2026-06-08 14:03:43 CST | 506.82s |
| R2R-CE | RFT / GRPO | `ckpt.iter270.pth` | `val_unseen` | 1839 | 2026-06-08 14:04:18 CST | 2026-06-08 14:12:55 CST | 516.82s |
| RxR-CE | SFT / DAgger | `ckpt.iter20600.pth` | `val_unseen` | 11006 | 2026-06-08 15:01:18 CST | 2026-06-08 16:21:46 CST | 4827.41s |
| RxR-CE | RFT / GRPO | `ckpt.iter1320.pth` | `val_unseen` | 11006 | 2026-06-08 16:37:14 CST | 2026-06-08 18:04:15 CST | 5221.38s |

| 任务 | 阶段 | NE | OSR (%) | SR (%) | SPL (%) | nDTW (%) | SDTW (%) | path length | collisions | steps taken |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2R-CE | SFT / DAgger | 4.1117 | 68.5155 | 63.1321 | 54.1995 | 64.4742 | 51.3214 | 13.3430 | 0.0782 | 96.0207 |
| R2R-CE | RFT / GRPO | 3.9381 | 71.5606 | 65.4160 | 55.8656 | 65.8794 | 54.1378 | 12.7271 | 0.1071 | 90.4997 |
| RxR-CE | SFT / DAgger | 5.3340 | 64.5466 | 58.6226 | 48.4129 | 63.9816 | 48.8331 | 18.6997 | 0.3569 | 181.6690 |
| RxR-CE | RFT / GRPO | 5.2728 | 65.6733 | 59.3858 | 48.6049 | 65.1429 | 50.0471 | 18.0970 | 0.3789 | 172.8086 |

结果文件：

| 任务 | 阶段 | 结果 JSON | 完整日志 |
|---|---|---|---|
| R2R-CE | SFT / DAgger | `data/logs/checkpoints/manual_eval_r2r_dagger/eval_results/stats_ckpt_25000_val_unseen.json` | `data/logs/manual_eval/full_r2r_dagger_20260608_135516.log` |
| R2R-CE | RFT / GRPO | `data/logs/checkpoints/manual_eval_r2r_grpo/eval_results/stats_ckpt_270_val_unseen.json` | `data/logs/manual_eval/full_r2r_grpo_20260608_140418.log` |
| RxR-CE | SFT / DAgger | `data/logs/checkpoints/manual_eval_rxr_dagger_env6/eval_results/stats_ckpt_20600_val_unseen.json` | `data/logs/manual_eval/full_rxr_dagger_env6_20260608_150118.log` |
| RxR-CE | RFT / GRPO | `data/logs/checkpoints/manual_eval_rxr_grpo_env3/eval_results/stats_ckpt_1320_val_unseen.json` | `data/logs/manual_eval/full_rxr_grpo_env3_20260608_163714.log` |

## 5. 论文报告结果

论文主表为 Table I：`Experimental results on R2R-CE and RxR-CE datasets`。本报告只和本地实际评测的 `val_unseen` split 对比。

| 任务 | 论文方法名 | 对应本地阶段 | NE | OSR (%) | SR (%) | SPL (%) | nDTW (%) | SDTW (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| R2R-CE Val Unseen | Ours-DAgger | SFT / DAgger | 4.11 | 69 | 63 | 54 | - | - |
| R2R-CE Val Unseen | Ours-GRPO | RFT / GRPO | 3.94 | 72 | 65 | 56 | - | - |
| RxR-CE Val Unseen | Ours-DAgger | SFT / DAgger | 5.42 | - | 58.26 | 48.19 | 63.78 | 48.53 |
| RxR-CE Val Unseen | Ours-GRPO | RFT / GRPO | 5.22 | - | 59.92 | 48.97 | 65.31 | 50.41 |

补充：论文 Table III 中 R2R GRPO 的细分数值为 SR 65.36、SPL 55.82；本地 R2R GRPO 为 SR 65.4160、SPL 55.8656，和 Table III 也基本一致。

## 6. 本地结果与论文结果差值

差值计算方式：`本地结果 - 论文结果`。NE 越低越好，所以 NE 为负数表示本地距离更低；百分数指标为正数表示本地更高。

| 任务 | 阶段 | NE 差值 | OSR 差值 (%) | SR 差值 (%) | SPL 差值 (%) | nDTW 差值 (%) | SDTW 差值 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| R2R-CE | SFT / DAgger | +0.0017 | -0.4845 | +0.1321 | +0.1995 | - | - |
| R2R-CE | RFT / GRPO | -0.0019 | -0.4394 | +0.4160 | -0.1344 | - | - |
| RxR-CE | SFT / DAgger | -0.0860 | - | +0.3626 | +0.2229 | +0.2016 | +0.3031 |
| RxR-CE | RFT / GRPO | +0.0528 | - | -0.5342 | -0.3651 | -0.1671 | -0.3629 |

## 7. 差异解释

整体看，四个 checkpoint 的本地正式评测都接近论文 Table I。R2R 两个 checkpoint 几乎完全贴合论文结果；RxR 两个 checkpoint 的差异也较小，其中最大的百分数差异约为 0.53 个百分点，最大的 NE 差异约为 0.086 米。

可能造成微小差异的因素：

1. 本地只有 1 张可见 GPU，而项目脚本原始设计更偏多卡运行。本次把 `--nproc_per_node`、`GPU_NUMBERS`、`SIMULATOR_GPU_IDS`、`TORCH_GPU_IDS` 都改成单卡设置，这会影响并行执行方式和耗时，但不改变评测 split、episode、指标或模型决策逻辑。
2. RxR 正式评测在单卡下为了稳定性降低了 `NUM_ENVIRONMENTS`。这只是减少同时开的 Habitat 环境数量，属于并行度调整，不改变 episode 总数，也不改变成功距离、轨迹评估或指标计算。
3. Habitat / Habitat-Sim / PyTorch 属于较老的栈，本地版本为 Habitat 0.1.7、Habitat-Sim 0.1.7、PyTorch 1.9.1+cu111。不同机器、驱动、并发环境下可能有很小的模拟器数值差异。
4. 论文表格通常保留 2 位小数或整数百分比，本地 JSON 保留完整浮点值；R2R 的 OSR、SR、SPL 与论文整数百分比对比时，本身就会有四舍五入差异。
5. 本次正式报告没有跳过失败 episode，也没有减少 episode 数量。失败的 RxR 尝试没有被当作正式结果；正式结果均来自完整 `val_unseen` split。

## 8. 报错与修复记录

本次没有做任何源代码修改，也没有改变导航语义或评测语义。遇到的两个问题都通过配置侧降低并行环境数量解决。

| 阶段 | 首次尝试 | 发生位置 | 报错摘要 | 根因判断 | 修复方式 | 修复后结果 |
|---|---|---|---|---|---|---|
| RxR SFT / DAgger | `manual_eval_rxr_dagger`, `NUM_ENVIRONMENTS 11` | 约 `7003/11006` | `RuntimeError: CUDA error: an illegal memory access was encountered`，随后有 `BrokenPipeError` | 单 GPU 上 RxR 以 11 个环境并行时压力较大，CUDA 异步错误导致环境清理时管道断开 | 改为 `NUM_ENVIRONMENTS 6`，其他评测语义不变 | 完成 11006/11006，`STATUS=0` |
| RxR RFT / GRPO | `manual_eval_rxr_grpo_env6`, `NUM_ENVIRONMENTS 6` | 约 `1892/11006` | `_pickle.UnpicklingError: invalid load key, '\x27'`，随后 `EOFError` | Habitat VectorEnv 子进程通信流损坏，推测为某个环境子进程异常退出 | 改为 `NUM_ENVIRONMENTS 3`，其他评测语义不变 | 完成 11006/11006，`STATUS=0` |

失败日志：

| 阶段 | 日志 |
|---|---|
| RxR SFT / DAgger 首次失败 | `data/logs/manual_eval/full_rxr_dagger_20260608_141412.log` |
| RxR RFT / GRPO 首次失败 | `data/logs/manual_eval/full_rxr_grpo_env6_20260608_162225.log` |

## 9. 最终结论

四个最终 checkpoint 都完成了正式 `val_unseen` 全量评测。本地结果与论文 Table I 基本一致：

1. R2R SFT / DAgger：NE、SR、SPL 与论文几乎一致，OSR 仅低约 0.48 个百分点。
2. R2R RFT / GRPO：NE、SR、SPL 与论文几乎一致；与论文 Table III 的 SR/SPL 细分值也非常接近。
3. RxR SFT / DAgger：本地 NE 略低，SR、SPL、nDTW、SDTW 均略高，差异都在很小范围内。
4. RxR RFT / GRPO：本地 SR、SPL、nDTW、SDTW 比论文低约 0.17 到 0.53 个百分点，NE 高约 0.053 米，仍属于接近复现。

因此，本次评测可以认为基本复现了论文报告的四个 checkpoint 结果。主要差异来自本地单 GPU 运行、RxR 并行环境数量降低、模拟器/驱动细节以及论文表格四舍五入，而不是评测语义变化。
