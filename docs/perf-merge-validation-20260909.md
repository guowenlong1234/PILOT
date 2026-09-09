# 已验证性能版本合并与主工作区测试

2026-09-09，按用户决定放弃静态条件缓存和导航SDPA，选取两项实施前的 `75cbe1d`，合并到当前 `feature/e24-joint-sft`。

- 合并提交：`372ba21`，两个父提交为当前工作区原提交 `0c7d41f` 和选定性能提交 `75cbe1d`。
- 合并提交与选定提交的 Git tree 均为 `9e6678c4f4fee91b880ec24acb69a46ee7f30064`，合并无冲突；之后仅补充本次状态和测试文档。
- 未合入 `c16290e` 及后续静态缓存/SDPA试验，源码、配置、CLI和测试中没有这两项实现。后续性能分支保留为历史记录，不作为本次合并来源。
- 笔记本 `/home/sia/project/ETP-R1` 中用户原有 `.gitignore` 未提交改动原样保留，未提交或覆盖它。
- 已通过中央仓库同步到训练机主工作区 `/home/gwl/project/etpr1/ETP-R1`，在该目录完成以下测试。

## 测试结果

实际环境为训练机既有 `etpnav_unified`，通过工程运行脚本接入；没有安装或修改环境。Python 3.10.14、PyTorch 2.2.2+cu121、CUDA构建12.1、Transformers 4.49.0、Habitat/Habitat-Sim 0.3.3，双 RTX A6000。

```bash
CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q tests --ignore=tests/integration --ignore=tests/test_runtime_behavior.py
```

退出0，**568 passed、4 skipped、2 warnings**。日志 `data/logs/perf_merge_20260909/regression.log`。

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q tests/test_nwm_compile.py tests/test_panorama_performance.py tests/test_direct_context.py
```

退出0，**11 passed、2 warnings**。日志 `data/logs/perf_merge_20260909/gpu_tests.log`。

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/rgb_only_optimization_runtime.sh server -m torch.distributed.run --nproc_per_node=2 --master_port=24897 scripts/benchmark_panorama_training.py --output data/logs/perf_merge_20260909/train_smoke --updates 8 --warmup 2 --observation-source direct --visual-precision fp16 --dino-batch 64 --nwm-batch 64 --compile-model --compile-backend inductor --audit
```

退出0，两卡均完成8次真实联合更新。两卡所有损失及梯度范数有限；冻结视觉、路点和世界模型参数哈希不变，导航CLS残差映射及融合层实际更新。Inductor两卡分别构建2/1份计算图。日志为 `data/logs/perf_merge_20260909/train_smoke.log`，详细审计及版本在 `train_smoke/rank0.json`、`rank1.json`。

本轮是合并后功能验证，不以8步短测给出新的性能提升结论。未重新执行完整导航评测，未保存本轮训练checkpoint，未启动正式长训练。测试进程已退出，两卡GPU已释放。

保留的推荐选项为 `direct_context_fast.yaml` 加 `direct_context_compiled.yaml`，或入口的显式直接渲染、FP16和编译参数；原配置默认行为未因本次合并而额外改变。
