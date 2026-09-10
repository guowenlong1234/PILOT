# 持久化融合候选节点状态：实现与验证

## 范围与当前状态

2026-09-10 按已确认的计划，从 `e4aa771` 建立 `feature/persistent-ghost-state`。
笔记本工作区 `/home/sia/project/ETP-R1-persistent-ghost`；训练机工作区 `/home/gwl/project/etpr1/ETP-R1-persistent-ghost`。
原工作区的未提交修改保留；论文正文未改。本分支未合入主工作区。

实现已接通训练、评测和轨迹推理的共享 rollout。真实 GPU 验收尚未完成：训练机两张 A6000 正在执行既有训练，按计划不结束或并行干扰现有任务。CPU 测试不能替代真实单卡、双卡训练与短评测。

## 状态更新

`MODEL.RAENWM.ghost_concat_memory_mode` 默认 `current_step_only`，保留旧行为；新值 `persistent_node_state` 启用持久状态。

候选保存融合状态 h 及真实观测次数 n。新关联的 m 次观测先全部聚合：

```text
u = (n*h + sum(new_observations)) / (n+m)
h_new = u + alpha * MLP(concat(u, prediction))
```

`GraphMap.ghost_embeds` 以 `[h_new*(n+m), n+m]` 存储，使现有观测聚合自然实现上述递推。预测不增加次数。同一步多个候选映射到同一 ghost，仍只融合一次。

没有有效预测时保留已聚合状态；本步未出现的历史候选不重复融合。位置继续按现有规则更新，不清空历史融合影响。ghost 删除时清理状态标记，真实到达节点使用真实观测。每个 rollout 新建图，episode 结束移除图，不保存跨 episode 状态。

批处理后仅对实际有效行写回。写回替换张量而不做原地修改、不 detach；后续损失能回传至之前融合网络。整段 rollout 统一反向传播，优化器更新之间不复用状态计算图。非有限预测、无效行和加权累加溢出局部回退。持久写回需要一次批量有效掩码从 GPU 传至 CPU，实际额外耗时和显存待 GPU 验收。

## 入口与兼容

- 新配置：`run_r2r/iter_train_rae_dino_ghost_concat_persistent.yaml`。默认目标朝向上下文、direct 渲染、FP16、编码/预测 batch 上限 64，关闭世界模型编译。
- 启动参数：`scripts/ghost_concat_job.py --memory-mode persistent_node_state`（完整别名 `--ghost-concat-memory-mode`），可结合 `--train-policy` 联合更新策略。
- 实验名含 `_persistent`，默认输出为 `data/logs/ghost_concat_persistent`；路径检查仅额外允许本次训练机工作区。
- 仍从原始 `pretrained/active_lookahead/base_iter14200.pth` 初始化，末层零初始化。第二阶段关闭，GRPO 仍拒绝 ghost concat。
- 检查点记录 memory、`weighted_observation_then_residual_v1` 更新版本、`full_rollout` 梯度策略。旧检查点缺 memory 字段按临时模式解释。带融合层的检查点在不同记忆模式间禁止加载，恢复还核对完整训练契约。
- 不新增 episode 中途图状态恢复；检查点仍保存模型和训练状态。

## 诊断与运行资源

新增 `RGB_fusion_persistent_writebacks`、`RGB_fusion_persistent_retained_states`、`RGB_fusion_persistent_state_norm_mean`；已有增量范数、融合梯度、耗时和峰值显存统计继续生效。retained 表示本步未再次写回但已有融合历史的节点数；范数均值统计本步写回状态。

训练机运行时 `.runtime/server_sft` 在新工作区单独复制，继续使用受保护环境以外的既有 `etpnav_unified`，未安装或修改包。DINO 模型按路径所有权要求独立复制，其他输入资产复用现有只读来源。

验证产物只在训练机：

```text
data/logs/persistent_ghost_validation_20260910/
  -> /mnt/data2tb/ETP-R1_data/experiments/persistent_ghost_validation_20260910/
```

实际版本：Python 3.10.14，PyTorch 2.2.2+cu121，CUDA 12.1，Transformers 4.49.0，Habitat / Habitat-Sim 0.3.3。版本导入命令退出码 0，记录 `versions.log`。

## CPU 验证

所有测试在训练机独立工作区执行，前缀均为：

```bash
env CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q
```

首轮覆盖 persistent/fusion/trainer/joint/job、graph candidate preview、frozen navigation 与 online checkpoint：73 passed，退出码 0，日志 `unit.log`。
额外旧路径回归覆盖 raenwm RGB fusion、RGB workflow、NWM prediction runtime、panorama runtime、joint optimizer、RGB projection：45 passed，退出码 0，日志 `regression.log`。其中 joint 的5项与首轮重复，不能把两轮相加作为独立测试总数。

补充验收测试覆盖第二步无预测的跨步梯度、同一融合网络连续两次优化且使用新图、暂停环境后的索引重排，以及真实到达节点不继承 ghost 状态。最终合并去重执行 117 项通过（`final_tests.log`，退出码 0）。另外真实审计钩子的两项 CPU 测试通过（`audit_tests.log`，退出码 0）：既能确认跨步保留，也能抓住“导航输入已融合但图未写回”的故意缺陷。共 119 项不同测试通过；模拟 rollout 审计不代替真实 Habitat/GPU 验收。

最终回归命令（工作区为训练机独立工作区）：

```bash
env CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q \
  tests/test_ghost_concat_persistent.py tests/test_ghost_concat_fusion.py \
  tests/test_ghost_concat_trainer.py tests/test_ghost_concat_joint.py \
  tests/test_ghost_concat_job.py tests/test_graph_map_candidate_preview.py \
  tests/test_rgb_fusion_frozen_navigation.py tests/test_online_checkpoint.py \
  tests/test_raenwm_rgb_fusion.py tests/test_rgb_fusion_only_workflow.py \
  tests/test_nwm_prediction_runtime.py tests/test_panorama_runtime.py \
  tests/test_rgb_projection.py
env CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q \
  tests/test_persistent_rollout_audit.py
```

实现提交 `68d8469`，行为审计与生命周期测试提交 `b20d501`，审计故障注入测试提交 `d4906d1`，均已推送中央裸仓库并快进同步训练机。

`scripts/check_ghost_concat_rollout.py` 支持新旧模式。它重新读取图字典，核对写回状态等于导航融合输入、计数不变、已访问节点不变；报告无预测保留、写回次数与范数。若真实轨迹未出现跨步无预测情况，报告 `cross_step_retention_exercised=false`，不能将其当作该情形已验收。

## GPU 待验收步骤

GPU 空闲后，在训练机上述工作区使用独立目录执行。先单卡两次更新，再保存/恢复验证、双卡两次更新与短评测；使用相同批量的临时模式短跑比较开销。恢复必须保持原计划总步数及优化器/调度器契约，不能改 iters 后冒充同实验恢复。

单卡两次更新示例（不启动世界模型编译、不同步测评机）：

```bash
python3 scripts/ghost_concat_job.py train --machine server --gpus 0 \
  --memory-mode persistent_node_state --train-policy --batch 1 \
  --iters 2 --log-every 1 --port 24961 \
  --output data/logs/persistent_ghost_validation_20260910/single
```

双卡使用 `--gpus 0,1 --batch 1 --port 24962` 并换到 `.../dual` 输出。短评测使用对应新检查点、同一记忆模式、`--machine server --environments 1 --episodes 2`。

这些真实验收尚未执行，不据此声称数值稳定、性能提升或可启动长训练。
