# 原生 CLS RGB 注入与 Top-5 冻结式 GRPO 操作说明

## 运行边界

本阶段从对应数据集的 native-CLS joint SFT checkpoint 启动。RGB adapter、
Top-5 CLS adapter、E24、NWM、DINO-CWP、DINOv2 和 waypoint predictor 全部
冻结；GRPO 只更新原有导航模块。

行为分布固定为 `etpr1-frozen-lookahead-stop-mass-v1`：基础 STOP 概率保持
不变，Top-5 residual 只重新分配 MOVE 条件概率。rollout 保存调整后的完整
概率和冻结 residual；current/reference update 使用同一份 residual。

SFT、评测和冻结 GRPO 还必须共享
`r1_low_level_move_rgb_anchor_v1`：reset/teleport 清空轨迹并记录落点锚帧，之后
只接收 `MOVE_FORWARD` 后的非静止正前方 RGB。GRPO 在每次 `envs.step()` 后、
暂停环境前复用和 SFT 相同的批量编码同步器；低级模式禁止写入当前高层全景。
源 checkpoint 必须同时带有完全匹配的 `raenwm_context_metadata` 和 joint
provenance。旧高层 checkpoint 不能直接进入这个低级冻结 GRPO 流程。

## 训练机入口

先选择已经完成技术验收的 joint SFT checkpoint，并通过环境变量显式传入：

```bash
export ETPR1_R2R_ACTIVE_GRPO_SOURCE_CHECKPOINT=/absolute/path/to/r2r/ckpt.iterN.pth
bash scripts/manage_native_cls_e24_grpo_server.sh r2r start

export ETPR1_RXR_ACTIVE_GRPO_SOURCE_CHECKPOINT=/absolute/path/to/rxr/ckpt.iterN.pth
bash scripts/manage_native_cls_e24_grpo_server.sh rxr start
```

管理动作包括 `start`、`resume`、`status`、`logs`、`tail` 和 `stop`。恢复时仍
必须提供同一个源 SFT checkpoint；启动器会重新计算 SHA-256，GRPO 的
reference policy 始终从该不可变文件加载，不从当前 GRPO checkpoint 复制。

快速技术验证可覆盖：

```bash
export ETPR1_ACTIVE_GRPO_SAMPLE_NUM=2
export ETPR1_ACTIVE_GRPO_ITERS=1
export ETPR1_ACTIVE_GRPO_LOG_EVERY=1
export ETPR1_ACTIVE_GRPO_NUM_ENVIRONMENTS=1
export ETPR1_ACTIVE_GRPO_BATCH_SIZE=1
```

smoke 必须使用独立输出目录，不能接续正式训练。

## 验收顺序

1. 配置、资产 SHA、joint provenance 和 reference SHA 严格加载。
2. R2R/RxR 各完成单环境纯采样，确认 RGB fused count 和 q0/q1 诊断非零。
3. 各完成一次更新并保存模型/训练状态。
4. 新进程严格恢复并再更新一次，核对环境、随机状态与 NWM generator。
5. 双卡短测检查导航梯度，以及所有冻结模块的参数/梯度状态。
6. 在对应实际环境完成 R2R/RxR 单 episode 评测后，才考虑长训练。

`sample_num=8` 会把 NWM 与 DINO-CWP 推理量放大八倍，正式参数必须在短测后
根据吞吐和显存决定。第一阶段强制 `update_epochs=1`，不允许用同一份冻结
rollout context 做多轮参数更新。

统一 Q0 缓存版本还要求源 checkpoint 的 provenance 包含：

- `q0_contract: r1_post_update_ghost_mean_cached_v1`；
- `q0_cache_precision: cpu_fp16`；
- `q0_recompute_forbidden: true`。
- `context_contract: r1_low_level_move_rgb_anchor_v1`；
- `context_sampling_action: MOVE_FORWARD`；
- `context_teleport_anchor_policy: clear_then_record_landing_rgb`；
- `low_level_encode_batch_size: 64`。

GRPO rollout 继续执行一次第一阶段 Q0，并把同一缓存交给 Top-5；日志中的
`lookahead_q0_requested` 必须为 0。保存后应逐张量比较源 SFT 与 GRPO
checkpoint 的 RGB 融合层及 E24/Top-5 状态，确认冻结权重完全相同。
