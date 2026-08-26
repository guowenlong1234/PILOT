# 原生 CLS RAE-NWM 导航前瞻实施计划

**目标：** 在不改变旧 patch-only NWM/E24 实验语义的前提下，为 R2R SFT 与评测新增原生 257-token CLS+patch 世界模型链路，并完成保存、恢复和真实环境技术验收。

**总体做法：** 先迁移并验证最小原生序列推理核心，再把在线 DINO 输出、四帧缓存、RGB 融合和 Top-5 future 逐层接通；最后替换新模式下的 E24 replay 损失并升级 checkpoint。每一层先写失败测试，旧路径始终保留回归测试。

**执行边界：** 源码在笔记本工作区修改。真实 NWM 数值一致性、单/双卡 SFT 冒烟在训练机 `/home/gwl/project/etpr1/ETP-R1` 的实际环境运行；单 episode 评测在测评机 `gwl-etpr1-rae` 容器和 `etpr1_rae` 环境运行。源码只通过 Git 同步。不开长训练，不跑完整 `val_unseen`，不修改受保护工程、环境或 ETPNav 容器。

---

## Task 1：迁移原生 257-token 推理核心

**文件：**

- 修改 `vlnce_baselines/nwm/raenwm_core/models.py`
- 修改 `vlnce_baselines/nwm/raenwm_core/infer_compat.py`
- 修改 `vlnce_baselines/nwm/predictor.py`
- 新增/修改 `tests/test_nwm_prediction_runtime.py`

- [ ] 只读核对训练机指定 RAE-NWM 提交、配置、CDiT 原生序列结构、EMA 键和 10 步 Euler 采样。
- [ ] 添加 257-token 噪声、上下文、采样输出和严格 EMA 加载失败测试。
- [ ] 迁移最小原生序列能力，严格拒绝 missing/unexpected keys 和 patch-only 回退。
- [ ] 保持旧 patch-only predictor 测试通过。

## Task 2：建立 CLS/patch 数据契约

**文件：**

- 修改 `vlnce_baselines/nwm/types.py`
- 修改 `vlnce_baselines/nwm/runtime.py`
- 修改 `vlnce_baselines/nwm/etp_adapter.py`
- 修改 `vlnce_baselines/nwm/predictor.py`
- 修改 `tests/test_nwm_prediction_runtime.py`

- [ ] 添加 `[CLS+patch]` 打包/拆包往返测试。
- [ ] 添加 CLS spatial-stat 归一化/反归一化测试。
- [ ] 扩展 `NwmPrediction` 为 tokens、normalized/raw CLS 和 patch 的明确输出。
- [ ] 新模式缓存/snapshot/batch 使用 `[4,257,768]`，不足四帧保持无效。
- [ ] 新模式不创建也不调用旧外置 heads。

## Task 3：接通在线 DINO 三类输出和 RGB 注入

**文件：**

- 修改 `vlnce_baselines/models/encoders/rae_dinov2_encoder.py`
- 修改 `vlnce_baselines/models/R1Policy.py`
- 修改 `vlnce_baselines/nwm/rgb_fusion.py`
- 修改 `vlnce_baselines/ss_trainer_ETP_R1.py`
- 修改 `tests/test_online_rae_projection.py`
- 修改 `tests/test_raenwm_rgb_fusion.py`

- [ ] 先测试 `raw_cls/nav_cls/raw_patch` 输出边界和 DINO 冻结。
- [ ] policy 保留前向视图 raw CLS+patch，导航仍使用 nav CLS。
- [ ] 新模式用反归一化 native CLS 和余弦 agreement；gate 末层零初始化。
- [ ] 验证初始恒等、非零梯度、只修改 new/existing ghost。

## Task 4：增加 Top-5 CLS adapter 与原生 future

**文件：**

- 新增 `vlnce_baselines/nwm/active_lookahead/native_cls_adapter.py`
- 修改 `vlnce_baselines/nwm/active_lookahead/dino_cwp_future.py`
- 修改 `vlnce_baselines/nwm/active_lookahead/joint_e24.py`
- 修改 `tests/test_stage0_dino_cwp_future.py`
- 修改 `tests/test_stage0_e24_joint.py`

- [ ] 测试 5 维条件固定为 `[condition.dx, condition.dy, sin(dtheta), cos(dtheta), rel_t]`。
- [ ] 实现零残差初始化的独立 Top-5 CLS adapter。
- [ ] q0 只把 native patch 给 DINO-CWP；q1 直接使用 native 257 tokens。
- [ ] 只替换 token 0，256 个 patch 逐值不变。
- [ ] 联合 DDP wrapper 同时持有 Top-5 adapter 和 E24。

## Task 5：实现 adjusted full-logits replay 损失

**文件：**

- 修改 `vlnce_baselines/nwm/active_lookahead/joint_e24.py`
- 修改 `vlnce_baselines/ss_trainer_ETP_R1.py`
- 修改 `tests/test_stage0_e24_joint.py`

- [ ] replay 保存 raw q1 tokens、condition、完整基础 logits/mask、Top-5 映射和教师动作。
- [ ] 实现 STOP、Top-5 内/外教师、padding 的完整交叉熵测试。
- [ ] adjusted loss 只更新 Top-5 adapter/E24；base loss 更新导航和 RGB adapter。
- [ ] 无有效 row 时建立覆盖所有联合参数的 DDP 零值 dummy graph。
- [ ] 保留旧配置的 decision-aware loss，不做无关删除。

## Task 6：配置、资产和 checkpoint v2/v4

**文件：**

- 新增 `configs/nwm/raenwm_mp3d_fresh_cls.yaml`
- 新增 `run_r2r/iter_train_rae_dino_native_cls_e24_joint.yaml`
- 修改 `vlnce_baselines/config/default.py`
- 修改 `vlnce_baselines/ss_trainer_ETP_R1.py`
- 修改 `tests/test_rae_checkpoint.py`
- 新增/修改工作流测试

- [ ] 新模式严格校验 75k EMA、推理配置 SHA 和全部外部资产身份。
- [ ] 四个优化器参数组命名固定为 navigation decay/no-decay、Top-5 adapter、E24。
- [ ] 模型 checkpoint 使用 `etpr1-native-cls-e24-joint-v2`。
- [ ] training-state v4 保存/恢复 optimizer、scheduler、scaler、iteration、episode、全部 RNG 和每 rank NWM generator。
- [ ] new-run 只允许 base 缺少新状态；评测/requeue 必须严格完整。

## Task 7：真实数值一致性和启动脚本

**文件：**

- 新增 `scripts/validate_native_cls_nwm_parity.py`
- 新增 native CLS SFT smoke/评测 wrapper
- 新增对应脚本测试

- [ ] 固定真实四帧、condition 和 `[1,257,768]` noise，对比上游与兼容层完整输出。
- [ ] 新增独立 smoke 入口，不参数污染旧 70k+heads 实验。
- [ ] 新增固定单 checkpoint、单 episode、独立输出目录的测评入口。

## Task 8：分层验收

- [ ] 笔记本运行不依赖 GPU 的聚焦单测和静态检查。
- [ ] 提交源码后推送中央仓库；确认训练机工作区干净后 ff-only 更新。
- [ ] 训练机记录 Python/PyTorch/Transformers/CUDA/Habitat 版本并完成真实数值一致性。
- [ ] 训练机单环境两次更新、保存；新进程恢复并再更新一次。
- [ ] 训练机双卡短 smoke，审计 DINO/NWM/DINO-CWP/waypoint 冻结和 DDP 梯度。
- [ ] 提交并推送；确认测评机工作区干净后 ff-only 更新。
- [ ] 测评机先检查 GPU、容器和 ETPNav 任务，再完成固定单 episode。
- [ ] 在实际目标环境运行相关回归与完整 `pytest -q tests`。
- [ ] 最小更新 `research.md`，保留用户现有未提交内容，记录最终命令、版本和结果。
