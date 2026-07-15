# RAE/DINOv2 测评机验收记录

## 结论

RAE/DINOv2 视觉分支已在 4090 测评机的独立容器 `gwl-etpr1-rae`、独立环境 `etpr1_rae` 和 ETP-R1 自有 Habitat 0.3.3 运行目录中完成正式验收。原 CLIP 分支仍能加载旧 checkpoint 并完成单 episode。当前完成的是正式冒烟，不代表完整规模训练或完整数据集指标已经完成。

## 环境与源码

- 日期：2026-07-15。
- 测评机：`eno1=10.10.10.2`，RTX 4090 24GB。
- Python：3.11.15。
- PyTorch：2.2.2+cu121。
- Transformers：4.49.0。
- Habitat、Habitat-Sim、Habitat-Baselines：0.3.3，均来自 `.runtime/etpr1_habitat`。
- 最终代码验证提交：`5436799`。
- 正式 smoke run：`tree-f4503e77b2e3_20260715T080515Z_98271`。
- 源码 manifest：`f4503e77b2e338cc4f8efab5a5193ca8223f7afa1ee6e39509f0d16c371e5c76`，269 个受控文件。

正式 smoke 的持久化证据位于测评机：

```text
/home/a6000/gwl/ETP-R1/data/logs/rae_dino_smoke/tree-f4503e77b2e3_20260715T080515Z_98271/
```

其中 `summary.log`、`source_identity.json` 和 `source_manifest.sha256` 分别记录阶段结果、源码身份和逐文件校验和。

## 正确性验证

完整测试命令：

```bash
scripts/etpr1_rae_runtime_exec.sh pytest -q tests
```

结果：`269 passed, 3 warnings in 59.51s`。三个 warning 分别来自 ImageIO、Python `cgi` 弃用和 PyTorch `meshgrid`，没有测试失败。

真实 RAE 对照结果：

```text
max_abs=0
cosine=0.9999999404
autocast_max_abs=0
```

全量特征文件：

```text
pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5
```

实测含 10,567 个视点，每个数据集都是 `[36,768] float32`；错误形状、错误类型、非有限值、全零、缺失键和额外键均为 0。文件记录的 DINO 权重 SHA256 为 `7a6f7b3b9fa4b8732e707476a03cd6cdce210048582f21aafb7991c17d98e362`，RAE 统计量 SHA256 为 `84ede66def5e6e3f25679334dc89cf63b12aacb99cbf0f5ae7ed4ad3187f7e59`。

正式 smoke 的 15 个阶段全部通过：运行时、契约测试、真实 MLM/SAP batch、一步预训练及审计、R2R SFT/GRPO/评测及审计、RxR SFT/评测及审计。最后一行为 `ALL_PASS`。

## CLIP 回归

CLIP 回归使用原 `run_r2r/iter_train.yaml`、原 DAgger checkpoint `ckpt.iter25000.pth` 和联合预训练权重 `model_step_367500.pt`。旧 checkpoint 中嵌入的 Habitat 0.1.x `Config` 由窄范围兼容加载器转换为普通字典；937 个模型参数正常加载。

结果：

- 实际选择 `CLIPEncoder`，未加载 DINO。
- 权重报告没有未处理的 missing layer 或 extra layer。
- 完成 1 个 R2R `val_unseen` episode，退出码 0。
- 日志：`data/logs/rae_dino_final_validation/clip_regression_final.log`。

## RTX 4090 性能基线

RAE 编码器使用 float32，对 12 张 `224x224 uint8 RGB` 图像预热 5 次后测量 50 次：

- 平均：18.290 ms/12 视角。
- 中位数：18.201 ms。
- 最小/最大：18.012/19.439 ms。
- 模型加载后显存：339.9 MiB。
- 编码峰值显存：475.3 MiB。

日志：`data/logs/rae_dino_final_validation/encoder_benchmark.log`。

正式单环境 smoke 的墙钟耗时与阶段内峰值显存：

| 阶段 | 耗时 | 峰值显存 |
|---|---:|---:|
| 一步预训练 | 20.634 s | 6,332 MiB |
| R2R SFT，一次 iteration | 14.553 s | 9,945 MiB |
| R2R GRPO，一次 rollout/update | 15.244 s | 6,791 MiB |
| R2R 单 episode | 11.742 s | 3,222 MiB |
| RxR SFT，一次 iteration | 23.545 s | 10,636 MiB |
| RxR 单 episode | 17.235 s | 3,827 MiB |

这些时间包含进程启动、模型加载和一次实际工作，不等同于长训练稳定阶段的纯 iteration 吞吐。整轮监控日志为 `data/logs/rae_dino_final_validation/final_smoke_gpu_5436799.csv`。

## 尚未执行的内容

- 完整规模联合预训练。
- 完整 SFT 或 GRPO 长训练。
- R2R/RxR 完整验证集评测和正式指标比较。
- 多环境吞吐和显存扩展测试。

这些工作应使用新的实验名和独立输出目录，不能覆盖本次 smoke、原 CLIP 配置或已有 checkpoint。

## 完整数据 batch 实测

2026-07-15 在测评机单张 RTX 4090 上，使用完整 3,210,737 条真实联合预训练数据、RAE RGB 特征、原深度特征和 float32，进行了三组独立短跑。每组都实际执行前向、反向和参数更新，并在结束时完成 1 条验证样本和 checkpoint 写入；测试 checkpoint 在确认写入成功后已删除，只保留配置、控制台输出、退出码、耗时和 GPU 监控日志。

| 单卡 batch | 梯度累积 | 有效 batch | 参数更新数 | 峰值显存 | 结果 |
|---:|---:|---:|---:|---:|---|
| 32 | 1 | 32 | 20 | 23,684 MiB | 通过，但只剩约 880 MiB，不建议长跑 |
| 16 | 1 | 16 | 20 | 17,692 MiB | 通过，约有 6.7 GiB 余量 |
| 16 | 8 | 128 | 2（16 个小 batch） | 17,842 MiB | 通过，建议用于正式预训练 |

`batch_size=16 + gradient_accumulation_steps=8` 保持了原四卡 `4 x 32` 的有效 batch 128，同时避免 batch 32 贴近显存上限。两次累积更新的稳定阶段约为每次 2 秒，据此粗算 50 万次更新仅训练部分约需 11.6 天；短跑包含的样本太少，正式排期前仍应做一次更长的吞吐试跑。

持久化证据位于：

```text
/home/a6000/gwl/ETP-R1/data/logs/rae_dino_batch_test/bs32_fp32_20260715/
/home/a6000/gwl/ETP-R1/data/logs/rae_dino_batch_test/bs16_fp32_20260715/
/home/a6000/gwl/ETP-R1/data/logs/rae_dino_batch_test/bs16_acc8_fp32_20260715/
```

本轮没有改动正式配置，也没有启动完整预训练。
