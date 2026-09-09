# 世界模型编译：数值差异定位与预测质量验证

日期：2026-09-09。工作区：`perf/panorama-training`；训练机 `/home/gwl/project/etpr1/ETP-R1-perf`。本轮按用户要求实现编译，且以预测质量而不是逐元素一致作为主要验收依据。

推荐 **Inductor 动态形状编译**：同轮双卡短训练预热后由 11.296 降至 9.265 秒/更新，耗时下降 17.98%，吞吐提高 21.92%；固定复核集的 CLS/图块余弦变化约为 -0.000090/-0.000063。实现已同步到训练机性能工作区，没有启动正式长训练。

## 上次为什么没有通过

重新读取 `compile_trial/run.log` 并固定旧回放第一组输入、八行预测批量及显式噪声后，本轮再次得到最大绝对差 **0.7890760899**。因此这不是拿不同轨迹、不同噪声或不同 checkpoint 比出来的差异。

同一份实际上下文和权重的分层实验：

| 执行方式 | 单次网络最大差 | 完整采样最大差 |
| --- | ---: | ---: |
| 计算图转换后执行原生算子（AOT eager） | 0 | 0 |
| 默认 Inductor 生成算子 | 0.078125 | 0.789076 |
| Inductor 限制融合大小、关闭 epilogue fusion | 0.0703125 | 0.727986 |
| 本轮原生算子 CUDA Graph 编译 | 0 | 0 |

误差在单次网络计算中已经出现；不是只有多步采样器才出现。关闭部分融合不能恢复原结果，因此不能把问题简单描述成一个融合开关的错误。

进一步取第一层真实前馈网络的 BF16 中间值，单独比较 `silu(x1) * x2`。原生路径先将 SiLU 输出舍入到 BF16，再乘法、再舍入；编译路径的中间计算保留 FP32，最后才舍入。实测：

- 编译结果相对原生 BF16 运算，最大差 0.03125，24.17% 元素超出原 `1e-4` 容差。
- 编译结果与“中间保留 FP32、末尾转换回 BF16”的对照**逐元素相同**。
- 仅将这个中间结果传入同一个原生输出投影，输出最大差就达到 0.0625。

这直接确认了一个实际误差来源：编译改变了低精度中间舍入的位置。注意力、条件嵌入等模块也测得差异，不声称上面这一个门控运算解释全部误差。多层网络和多步采样继续传播这些差异。它不等于权重加载错误，也不等于预测质量一定下降；本轮没有修改环境包、关闭保护或换权重来“通过”检查。

## 两种可选编译实现

入口仍使用 `torch.compile`，只编译冻结的世界模型，不编译模拟器、DINO 或参与反向的导航网络。

- `native`：计算图交给原生 PyTorch 算子执行，并捕获为 CUDA 执行图反复运行，减少逐算子调度。使用动态形状图；具体尺寸的 CUDA 图最多缓存四份，超出后淘汰。输出复制为独立存储，避免下一采样步覆盖之前的结果。不改变原运算舍入方式。
- `inductor`：使用 PyTorch 默认的算子生成后端，允许算子合并和中间舍入变化。使用动态形状，关闭其额外 CUDA Graph 层；精度由真实预测质量验证。

直接使用 PyTorch 2.2.2 自带的 `backend='cudagraphs'` 在当前模型上报 `Mixing fake modes NYI`。本轮 `native` 使用独立后端避开该失败路径，不修改安装的 PyTorch，也不静默退回未编译执行。

默认配置仍关闭编译。开启方式为 `MODEL.RAENWM.torch_compile=True`，开启后默认后端为 `inductor`；也可显式选择 `MODEL.RAENWM.compile_backend=native` 做数值对照。正式任务入口及短训练入口均提供 `--compile-model --compile-backend ...`。现有 direct/FP16/64/64 参数保持独立，FP16 指定向 DINO 编码；世界模型沿用原来的 BF16 混合精度。

## 固定输入预测质量

复用上一轮的 11 场景、22 个 episode、83 个决策片段、288 个目标。开发部分为六场景 162 个目标，复核部分为五场景 126 个目标、三个固定噪声，共 378 对目标—噪声。真实目标图像仅用于评分，各执行方式共享历史输入、噪声、批次划分、四帧上下文和十点 Euler 采样配置。

下表为复核集按场景等权结果：

| 执行方式 | CLS 余弦 ↑ | 图块余弦 ↑ | CLS RMSE ↓ |
| --- | ---: | ---: | ---: |
| 未编译 | 0.829385525 | 0.707941077 | 0.554763666 |
| native | 0.829385525 | 0.707941077 | 0.554763666 |
| inductor | 0.829295454 | 0.707877688 | 0.554956792 |

native 在全部开发/复核输出上最大绝对差为零。Inductor 复核集 CLS 余弦变化约 -0.0000901，图块余弦约 -0.0000634；最差单场景变化分别为 -0.0001329、-0.0002644。开发部分最差单场景变化分别为 -0.0001831、-0.0001811。均在上一轮采用的 0.005 场景均值余弦下降范围内。

Inductor 复核集仍存在约 2.449 的单元素最大差，但整体预测质量变化很小。这说明单元素最大误差不能替代本任务的质量指标。这里没有完整导航 SR/SPL 结论。

本轮质量运行中，两种后端都只构建一次动态计算图；native 因具体尺寸变化共捕获 14 次、回放 351 次、淘汰 10 次，结束时缓存四份，没有按每个候选数重新进行 Dynamo 图编译。

## 双卡训练速度

三组串行执行，每组 16 次真实联合更新，剔除前四次预热，不启用分段强制同步。保持双 A6000、每卡四环境、总批量八、相同 14200 基座、冻结 75000 世界模型、direct/FP16/64/64、导航学习率 2e-6 和融合学习率 1e-5。以两卡较慢的平均更新时间计。

| 执行方式 | 预热后秒/更新 | 每秒轨迹数 | 含预热的 16 次更新总秒数 |
| --- | ---: | ---: | ---: |
| 未编译 | 11.2958 | 0.7082 | 187.20 |
| native | 12.9612 | 0.6172 | 217.59 |
| **inductor** | **9.2651** | **0.8635** | **205.39** |

Inductor 首次更新约 45.29 秒，原版约 17.22 秒；这些是包含初始化/采样等工作的更新时间，不是单独的编译计时。第一轮很短的任务可能无法抵消启动成本，不能将预热后 22% 的吞吐提升说成这 16 步总墙钟时间也改善了 22%。这里没有用外推冒充长训练测量。

两种编译方式的两卡分别只编译 2/1 份动态计算图，覆盖批量一与一般批量等分支；没有为每个候选数反复进行 Dynamo 编译。native 仍需缓存具体尺寸的 CUDA 执行图，训练中分别捕获 115/119 次、淘汰 111/115 次；日志还出现捕获前等待 NCCL 通信完成的提示。它在当前四图缓存配置下慢于基线，因此仅保留为数值对照选项，不推荐用于提速。

三个版本的两卡审计全部通过：冻结视觉、路点、世界模型权重不变；导航 CLS 残差映射和融合层发生更新，优化器分组/学习率一致。Inductor 的 PyTorch 峰值分配约 11541/10783 MiB（不含全部模拟器显存）。真实训练轨迹会随浮点/采样行为变化，三个版本世界模型调用次数分别为 163/156、171/156、175/156，不能将端到端差值当作严格固定轨迹的单模块时间。

这些短训练禁用了 checkpoint 写盘，结果不包含每 200 步正式保存的成本。原有 12.105 秒是上一轮 12 步协议的结果，不能将它替换为本轮的同规格基线来计算收益。

## 使用

配置文件按顺序合并：ghost concat 基础训练配置、`configs/nwm/direct_context_fast.yaml`、`configs/nwm/direct_context_compiled.yaml`。

正式 `scripts/ghost_concat_job.py` 入口在原 direct/FP16/64/64 参数上增加：

```text
--compile-model --compile-backend inductor
```

编译开关默认关闭，未对既有实验自动启用；原有工作区限制、资产检查、续训检查和隔离环境规则仍有效。需要严格保持原生算子输出的回放时，显式选择 `--compile-backend native`；回放脚本的逐元素容差检查仍用于数值诊断，不替代质量验收。

训练机性能工作区的复现短训练命令（使用新的输出目录）：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/rgb_only_optimization_runtime.sh server -m torch.distributed.run --nproc_per_node=2 --master_port=24889 scripts/benchmark_panorama_training.py --output data/logs/panorama_perf_20260909/compile_training/<fresh_run> --updates 16 --warmup 4 --observation-source direct --visual-precision fp16 --dino-batch 64 --nwm-batch 64 --audit --compile-model --compile-backend inductor
```

## 回归与产物

训练机实际环境：Python 3.10.14、PyTorch 2.2.2+cu121、CUDA 构建 12.1、Transformers 4.49.0、Habitat/Habitat-Sim 0.3.3，双 RTX A6000。使用项目既有 `scripts/rgb_only_optimization_runtime.sh server`，没有修改受保护环境。

- CPU 回归：`CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q tests --ignore=tests/integration --ignore=tests/test_runtime_behavior.py`，退出 0，568 passed、4 skipped、2 warnings。
- GPU 编译合同：`CUDA_VISIBLE_DEVICES=1 bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q tests/test_nwm_compile.py`，退出 0，2 passed、2 warnings。覆盖不同批量/输入值、反复采样、旧输出持有、图缓存淘汰、权重不变及全局 TF32 设置不变。
- 最终默认后端调整后：`CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q tests/test_nwm_compile.py tests/test_nwm_prediction_runtime.py tests/test_ghost_concat_joint.py tests/test_ghost_concat_trainer.py tests/test_ghost_concat_fusion.py tests/test_direct_context.py`，退出 0，50 passed、1 skipped、2 warnings。入口 `--compile-model --dry-run` 另行核验默认选择 Inductor。
- 数值复现：`scripts/diagnose_nwm_compile.py`，`compile_diagnosis/` 中保留 `default_v2`、`aot_eager`、`nofusion`、`native`、`gate` 报告。最初诊断包装器缺少 `.parameters()` 的错误已修复；它和自带 cudagraphs 后端失败记录均保留，不计为通过结果。
- 质量：`scripts/validate_direct_context.py score --source . --prepared data/logs/panorama_perf_20260909/direct/quality/prepared.json --output data/logs/panorama_perf_20260909/compile_quality --compile-comparison`，通过同一运行时执行，退出 0。结果为 `quality_summary.json`、`paired_quality_deltas.json`、`quality_rows.json` 和 `compiler.json`。
- 三组双卡训练均退出 0，记录为 `compile_training/{eager,native,inductor}/rank{0,1}.json`，汇总为 `compile_training/summary.json`。训练验证实现提交为 `02c53a6`；后续仅调整启用编译时的默认后端、提供覆盖配置和完善文档。

所有原始产物均在训练机 `data/logs/panorama_perf_20260909/`。原长训练保持停止。
