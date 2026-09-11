# 持久候选状态：训练机真实验证

用户要求先测试 `feature/persistent-ghost-state`，通过后合入笔记本主工作区，再以此前正在测评的实验参数启动长训练。

## 位置与环境

验证在训练机 `/home/gwl/project/etpr1/ETP-R1-persistent-ghost`，使用既有 `etpnav_unified` 和本工作区 `.runtime/server_sft`。Python 3.10.14、PyTorch 2.2.2+cu121、CUDA runtime 12.1、Transformers 4.49.0、Habitat/Habitat-Sim 0.3.3；两张 RTX A6000。

产物根 `data/logs/persistent_ghost_validation_20260911/` 链接到 `/mnt/data2tb/ETP-R1_data/experiments/persistent_ghost_validation_20260911/`。未修改受保护环境。预训练测试所需的 HDF5、R2R 标注和文本模型通过符号链接复用训练机已有资产。

## 已完成检查

- 环境导入退出 0，见 `versions.log`。
- 持久状态、融合、图更新、检查点及 GRPO 冻结相关回归：127 passed，退出 0，见 `tests.log`。
- 编码器真实 GPU 数值一致性、HDF5 读写合同及训练阶段冻结检查：68 passed；同轮四个预训练检查因独立工作区缺资产路径报错。补齐上述链接后单独重跑预训练文件：5 passed，退出 0，见 `pretrain_retry2.log`。失败日志保留，不计为通过结果。
- 从实际完整特征文件取三个视点写入独立小型 HDF5，再打开核对，均为有限的 `[36, 768]` 数据；退出 0，见 `tiny_hdf5.json`。
- 单卡、单环境、两次真实联合更新，开启直接渲染、FP16、64 批次上限及 Inductor；退出 0，模型与训练状态成对保存，见 `single/train/ghost_concat_v1_joint_persistent_train/manifest.json`。
- 双卡每卡批量 4，总批量 8，总步数固定为 4、每步保存。在第 2 步完整保存后由审计工具主动退出 75，再仅改变 `IL.is_requeue=True`，恢复完成第 3、4 步，退出 0。见 `dual_command.json`、`dual_resume_command.json`、`dual_initial.log`、`dual_resume.log`。
- 恢复后的两卡审计均通过：冻结视觉、路点和世界模型权重哈希不变，导航 CLS 映射与融合层均更新。第 2、4 步模型参数有限，均有 485 份优化器状态，调度器 `last_epoch` 分别为 2、4；两卡环境遍历状态成功恢复。见 `dual_audit/` 与 `checkpoint_audit.json`。
- R2R 四条真实路线的图状态审计退出 0：41 个导航步骤、468 次图状态核对、116 次融合写回、282 次跨步无新预测保留。真实观测节点未改，预测不增加观测次数，图中状态与导航输入一致。见 `r2r_rollout.json`、`r2r_command.json`、`r2r_audit.log`。
- RxR 单条真实路线审计退出 0：7 个导航步骤、72 次图状态核对、16 次融合写回、38 次跨步保留。只在短评测中叠加 RxR 任务配置，正式长训仍为 R2R。见 `rxr_rollout.json`、`rxr_command.json`、`rxr_audit.log`。

审计包装器只用于短验证，不用于正式长训练。首轮受控中断只由 rank 0 保存检查点，启动器会终止另一 rank；外层验证程序原先错误要求另一 rank 也有保存报告，已调整验收并补充逐段落盘。恢复完成后的两个 rank 都有完整通过报告。该问题属于审计工具，不是训练器故障。

## 参数对照

用 `get_config` 展开旧实验实际训练 manifest 和新长训命令，逐字段比较。除输出目录、同步目标路径和命令记录外，唯一差异是：

```text
MODEL.RAENWM.ghost_concat_memory_mode:
  current_step_only -> persistent_node_state
```

对照退出 0，完整记录见 `long_parameter_parity.json`。两者均使用 14200 基座、10000 步、每 200 步保存、双卡总批量 8、策略学习率 2e-6、融合学习率 1e-5、种子 100、直接渲染、FP16、64 批次上限和 Inductor；第二阶段关闭。

短验证只证明实现、梯度、恢复和运行链路可用，不代表导航性能提升。正式训练不使用短验证模型或优化器状态。
