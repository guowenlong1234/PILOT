# 多步递归前瞻接入（2026-10-09）

用户要求将独立递归前瞻合入当前工程，并指定在RTX 3090测评机验证。当前主线为`feature/e24-joint-sft`。本次采用三方移植解决主线演进带来的冲突，没有切换或覆盖远端工作目录；独立递归工作区保留。

## 为什么不能只合并旧分支

旧分支`9a97727`接的是联合训练/GRPO的候选q0缓存；主线Stage2使用持久节点的全景q0快照，原来固定生成一次q1。直接调用旧函数会改变q0朝向、世界模型移动跨度和噪声身份。

`5e9be0b`迁入旧递归实现并保留主线低层历史和Stage2导航状态；`49d20f8`接通当前二阶段，`8453867`补充各深度整段跳过的回归检查。

## 当前行为

- 第一阶段仍预测候选到达q0；二阶段默认只展开q1，可配置q1→q2或q1→q2→q3。这里深度不包含q0，也不等于Top-K候选数量。
- `stage2_collect.py`保持原q1计算、种子及FP16缓存语义；`stage2_rollout.py`只负责q2/q3。后层提议读取上一层预测的原始latent，位姿和移动跨度逐层累计。
- 每层世界模型都使用同一个真实源历史快照，预测不会追加成真实观测，递归不调用模拟器读取未来RGB/深度。
- 评分头仍接收每候选一个`[257,768]`端点特征，无需修改网络结构。不是把每层分数相加，也不是展开多分支搜索树。
- 后层提议无效/无后继、移动跨度越界、预测非有限时，保留最深有效端点；首层无效则不启用候选修正。未知运行异常与OOM仍报错，不伪装成正常回退。
- 新层使用独立且稳定的逐查询种子，不消耗第一阶段随机数。q1原始字段保留；新增`endpoint_conditions`、`endpoint_metadata`、逐层记录、实际深度和回退掩码。
- 整段剪枝在所有递归前执行。深度由配置固定，失败可提前终止；没有实现根据逐层分数自动决定继续或停止。严格证书仍依赖最终±1修正界；经验阈值1仅在先前一步实验上分析过，多步不能继承其不漏改选结论。

## 训练与部署

配置分别为`MODEL.STAGE2_COLLECT.lookahead_horizon_steps`和`MODEL.STAGE2_ONLINE.lookahead_horizon_steps`；统一启动器参数为`--lookahead-horizon-steps 1|2|3`。

旧数据/检查点缺少深度信息时按一步解释。新数据来源记录深度、端点评分和最深有效回退策略；写入和全量校验核对行深度，两个训练读取器均拒绝混合不同深度的数据目录。检查点已有的完整数据来源继续携带这些记录。

默认拒绝评分头训练深度与部署深度不同。调试性迁移须指定`--allow-rollout-depth-transfer`，在线报告明确写入训练深度、推理深度以及`head_training_depth_matches_inference=false`。已有`head_retrained`表示9200基座是否有原生评分头，不能用它推断多步已重训。

因此，代码接入并不要求重训世界模型；但正式比较多步方法，需要采集对应深度的训练数据、重新训练/适配评分头，再做成对评测。不能把一步旧头的多步短测视为已证明多步收益。

## 测评机验证

执行位置为`/home/a6000/gwl/ETP-R1`，专用容器`gwl-etpr1-rae`、环境`etpr1_rae`。只读检查确认GPU空闲、ETPNav容器停止；未使用训练机GPU。

回归命令（在专用容器内执行）：

```bash
bash scripts/rgb_only_optimization_runtime.sh eval -m pytest -q \
  tests/test_stage0_dino_cwp_future.py tests/test_stage0_e24_joint.py \
  tests/test_e24_joint_workflow.py tests/test_rxr_native_cls_e24_workflow.py \
  tests/test_grpo_frozen_lookahead.py tests/test_online_performance_optimizations.py \
  tests/test_stage2_*.py
```

源码`8453867`结果退出0，183 passed，3条依赖弃用/meshgrid提醒。日志`data/logs/recursive_integration_20261009/integrated_tests_v2.log`。此前发现两项旧测试夹具未包含主线Stage2条件/新增深度配置，已修正；首次误在宿主调用环境脚本因GLIBC导入失败，不计为有效测试，未修改环境。

真实导航入口`scripts/validate_stage2_recursive_navigation.py`：一步16路线与合并前完整计算组逐步动作/逐路线指标对照；两步、三步各4路线验证真实深度；随后三步训练集4路线采集、完整性校验与评分头前向/损失检查。每阶段保存命令、退出码和运行时版本，全部产物在测评机新目录`data/logs/recursive_integration_20261009/navigation_v1/`，不覆盖先前实验。

实际版本：Python 3.11.15、PyTorch 2.2.2+cu121、CUDA运行时12.1、Transformers 4.49.0、Habitat/Habitat-Sim 0.3.3；宿主驱动535.230.02。复现时在测评机宿主工程目录运行（换用新输出目录）：

```bash
python3 scripts/validate_stage2_recursive_navigation.py \
  --output data/logs/recursive_integration_20261009/navigation_v1 \
  --checkpoint /home/a6000/gwl/ETP-R1-stage2-e24/stage2_assets/ckpt.iter9200.pth \
  --head /home/a6000/gwl/ETP-R1-stage2-e24/data/logs/stage2_9200_future_ablation_20260921/formal_v1/full/train/head_step_002000.pt \
  --h1-reference data/logs/bounded_skip_20261009/navigation_v1/smoke_off
```

真实验证于北京时间15:06:15完成，监督及全部子阶段退出0：

| 检查 | 结果 |
|---|---|
| 一步真实导航 | 16路线、125次决策；对合并前参考动作零差异，逐路线指标完全相同 |
| 两步真实导航 | 4路线、31次决策；65个候选时刻达到第二层，12个回退第一层，73个无有效未来（包括STOP等） |
| 三步真实导航 | 4路线、31次决策；56个达到第三层，9个回退第二层，12个回退第一层，73个无有效未来 |
| 三步训练集采集 | 4路线、28条决策、18条可训练记录，75/138个未来槽有效；紧凑数据约30.19MB，全量SHA/身份/深度校验通过 |
| 评分头输入检查 | 两条有效记录组成`[2,5,257,768]`输入，零初始化残差为0、损失0.334685742855072且有限；没有反向传播或创建优化器 |
| 权重冻结 | 三组在线及采集的导航/评分模块与世界模型检查均exact_match |

一步仍为299次q1查询；两步77次q1加65次深层查询，三步77次q1加121次深层查询。后两组为旧一步头跨深度推理，报告明确记录训练深度1，不能据此宣称多步收益。它们也不是与一步16路线组匹配的性能对照。

最终只读检查GPU为0MiB、0%负载，本次监督/导航进程均已结束。产物全部留在测评机，没有复制模型、数据或大日志回笔记本；训练机工作目录未更新。
