# 批量 ghost 拼接残差融合：实现与验证

2026-09-08。修改前快照提交为 `a7ead28`，之后按用户确认的拼接方案实现；不采用此前提议的“复制全景、单候选对照再相减”。

## 结构

```text
原始观测 → 一次全景编码 → 纯观测图 → 原始ghost历史特征 [N,768]
世界模型当前预测CLS                         [N,768]
                            拼接           [N,1536]
                           Linear          1536 → 1536
                           GELU
                           Linear          1536 → 1536
                           GELU
                           Linear          1536 → 768
                            ↓
临时导航输入 = 原始ghost特征 + alpha × MLP输出
```

三层Linear，中间不设瓶颈；参数总数5,902,080。最后一层权重和偏置零初始化，alpha默认1。不做观测与预测的显式相减，不设置额外置信度门，也没有复制全景或第二次全景前向。

观测输入是纯观测图中已经全景编码、按历史观测平均的ghost嵌入；预测输入是世界模型反归一化后的原始DINO CLS。两个输入语义不同，由拼接MLP学习融合；此版本不额外使用C组的预测侧导航映射。

导航底座冻结且固定eval行为，世界模型冻结，仅训练拼接MLP；图导航仍保留对输入的梯度。更新ghost之后，图注意力仍可能改变STOP和其他动作分数，不能保证停止概率不变。

## 批处理与原始图保护

- CPU只处理已有的环境索引和ghost ID元数据，将当前预测记录匹配到图输入行；不在Python循环内读取GPU标量。
- 所有环境中匹配到的ghost一起gather，统一转移索引/预测张量，单次MLP前向。
- 用向量mask保护已访问节点、STOP、padding和非有限预测；无匹配预测时直接返回原输入。
- 用一次out-of-place `index_copy`生成临时图输入，不原地修改原输入或GraphMap。
- 原始 `ghost_embeds`继续只接收真实观测推导的特征；预测不参与历史平均，不跨步缓存。
- 重复预测ID与行数不匹配会明确报错；不同环境中的同名ghost按环境索引隔离。
- 非有限输入在MLP前按行屏蔽，正常预测不会被坏行污染。
- 覆盖率分母为本步全部唯一ghost查询，包括上下文未就绪的查询；与旧模式按候选行计算的覆盖率不能直接等同。
- 无门控与不同特征空间之间的余弦不再伪装成该模块的诊断量。保留查询、实际融合数量、增量范数和梯度统计。

## 文件与配置

- `vlnce_baselines/nwm/ghost_concat_fusion.py`：模块及跨环境批量gather/MLP/回填。
- `ss_trainer_ETP_R1.py`：预测与融合解耦，先更新纯观测图，再在导航前融合；训练、评测、轨迹推理共享这一rollout。
- `run_r2r/iter_train_rae_dino_ghost_concat.yaml`：独立配置、独立产物根、E24关闭。
- `scripts/ghost_concat_job.py`：独立train/eval入口，复用现有环境、资源检查、同步及manifest验证，不加入先前A/B/C队列。
- `scripts/check_ghost_concat_rollout.py`：真实导航期间检查原始RGB/图未变、每步一次全景和最多一次MLP批量前向。检查中的同步开销仅属于诊断工具，不进入生产运行。

关键配置：

```yaml
IL:
  freeze_navigation_backbone: True
MODEL:
  RAENWM:
    rgb_fusion_type: ghost_concat
    ghost_concat_hidden_dim: 1536
    rgb_fusion_zero_init: True
    rgb_fusion_alpha: 1.0
    rgb_fusion_trainable: True
    rgb_fusion_align_navigation_cls: False
  ACTIVE_LOOKAHEAD:
    enabled: False
```

隐藏宽度可以增大，但模块拒绝低于1536的宽度。旧 `residual_gate` 仍为全局默认模式。

检查点新增结构/维度/输入语义/临时存储合同，新旧adapter不能混载。新模式从干净原始基线初始化；恢复训练需要相同合同及alpha，评测允许显式覆盖alpha。没有NWM配置的旧模型保存以及旧结构恢复继续兼容。

当前只支持SS-ETP-R1路径，GRPO尚未接入新结构，入口会明确拒绝该模式，避免静默运行旧融合方式。

## 命令

代码已同步到训练机、测评机。以下正式训练命令只是使用说明，本次没有执行长训练：

```bash
ssh server 'cd /home/gwl/project/etpr1/ETP-R1 && python3 scripts/ghost_concat_job.py train --gpus 0,1 --batch 8 --iters 2000 --log-every 200 --sync --output data/logs/ghost_concat_formal_<run_id>'
```

中断后按完全相同参数追加 `--resume`。不同配置/轮次使用新的输出目录。

测评机宿主机入口（内部使用专用容器和环境）：

```bash
python3 scripts/ghost_concat_job.py eval --machine eval --gpus 0 --environments 8 --checkpoint <模型路径> --output data/logs/ghost_concat_eval_<run_id>
```

正数 `--episodes` 仅用于冒烟，不是固定ID调参子集。

## 已执行验证

所有行为测试在对应目标机器执行，没有以笔记本环境结果替代。

| 检查 | 结果 |
|---|---|
| 训练机广泛CPU回归，不含integration及测评机专属runtime_behavior | 529 passed, 2 skipped |
| 测评机融合/冻结/性能/检查点相关CPU回归 | 119 passed, 1 skipped |
| 训练机单卡1环境、2次真实更新 | 正常退出，模型/训练状态保存并原子同步 |
| 训练机双卡、每卡8环境、2次真实更新 | 正常退出 |
| 单卡/双卡检查点冻结审计 | 643个已保存导航张量全部与基线相等；编码器来源相同；MLP真实更新 |
| 测评机2环境、4条真实路线的结构检查 | 26个步骤、26次全景前向、23次MLP批量前向，最大批次9个ghost；114个临时输入行被修改，原始RGB/图及观测计数未改写 |

测试期间发现并修复了旧配置不含RAENWM时的保存兼容问题，以及新模式覆盖率分母遗漏未就绪查询的诊断问题。修复后上述回归全部通过。

训练环境：Python3.10.14、PyTorch2.2.2+cu121、Transformers4.49.0、Habitat/Habitat-Sim0.3.3；测评环境Python3.11.15，其余上述版本一致。CUDA构建12.1。单/双卡job日志均打印实际版本。

本次小规模验证产物位于两机工程的 `data/logs/ghost_concat_validation_20260908/`，含single、dual、冻结审计、性能基准及invariants结果。模型与训练日志没有复制回笔记本。

训练机已执行的核心验收命令（退出码均为0）：

```bash
CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q tests --ignore=tests/integration --ignore=tests/test_runtime_behavior.py
python3 scripts/ghost_concat_job.py train --gpus 0 --batch 1 --iters 2 --log-every 2 --sync --output data/logs/ghost_concat_validation_20260908/single
python3 scripts/ghost_concat_job.py train --gpus 0,1 --batch 8 --iters 2 --log-every 2 --output data/logs/ghost_concat_validation_20260908/dual
```

测评机在专用容器运行的CPU命令（退出码0，CUDA_VISIBLE_DEVICES置空）：

```bash
bash scripts/rgb_only_optimization_runtime.sh eval -m pytest -q tests/test_ghost_concat_fusion.py tests/test_ghost_concat_trainer.py tests/test_rgb_fusion_frozen_navigation.py tests/test_raenwm_rgb_fusion.py tests/test_online_performance_optimizations.py tests/test_rae_checkpoint.py
```

真实导航结构检查通过`check_ghost_concat_rollout.py`运行SS-ETP-R1 eval，使用本轮single/ckpt.iter2.pth、2环境、4episode，所有输出写入独立invariants目录；退出码0，检查结果为`ok=true, checked_steps=26, panorama_calls=26, mlp_calls=23, max_ghost_batch=9, fused_ghosts=114, changed_input_rows=114`。

## 性能基准

训练机另一张空闲RTX A6000，推理模式、autocast FP16；比较同一个非零初始化adapter的单批次前向与逐行调用拼接。预热3次，测量10次平均，前后CUDA同步。这里只测融合模块，不包含世界模型、全景编码和图元数据收集。

| ghost数量 | 批量/ms | 逐行/ms |
|---|---:|---:|
| 16 | 0.149 | 2.308 |
| 64 | 0.155 | 9.091 |
| 128 | 0.147 | 18.108 |

批量与逐行在FP16容差内一致，最大绝对误差不超过0.0002442。64行时模块批量前向约快59倍，主要减少小算子调用开销；不能把它当成整套导航训练或评测的提速倍数。

这些检查验证的是实现、冻结与批处理行为，不是导航性能提升。是否超过基线仍需后续正式训练和完整对照。
