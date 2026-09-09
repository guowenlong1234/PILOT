# 已观测全景的世界模型推理接口

新工作区提供独立的`PanoramaPredictionRuntime`，输出仍为原`NwmPrediction`：预测的原始CLS、标准化图块和按候选ID排列的元数据。输入只接受已保存图像及位姿，没有模拟器或目标真值回调。

本轮受测推荐模式为`world_exact_select`：四帧统一到目标的精确世界朝向，按历史运动和目标相对位置选择参考端点。配置为`configs/nwm/panorama_context_quality_best.yaml`，预测质量结果见`nwm-context-quality-results-20260909.md`。这是本轮受测方案中的选择，不是所有可能方案的全局最优。

## 缓存观测

每个环境维护一个`PanoramaHistory`，最多4个真实时间点。每个时间点保存：

- `position`：原模拟器地面参考点(x,y,z)；相机中心仍按原标定加1.25米高度。
- `body_yaw`：当时机体朝向。
- `cube_rgb`：6张90度方图，世界yaw0/90/180/270及pitch+90/-90，必须具有相同光心。
- `native_world_rgb12`：世界yaw0/30/…/330的12张原生224方图。30度对齐模式必须提供，才与正式验证的图像条件一致。
- `front_rgb`：当时直接记录的前置图，只有基线模式使用。
- `frame_id`与`segment_id`：真实时间点和连续移动段。重复时间点不能补齐历史；跨段自动清空，段内位置突跳会报错。

球面立方体面覆盖任意查询方向；匹配原生方向时直接复用原图，避免重投影损伤图像特征。原始RGB由帧对象复制并设为只读；生成的方向特征只缓存在该帧内部，帧移出后自然释放。原生12图与立方体面共约10.34MiB/时间窗口（四个时间点，未做重复正交面的共享优化），不含特征缓存和推理工作内存。

## 调用示例

```python
from vlnce_baselines.nwm.panorama_runtime import (
    ObservedPanoramaFrame, PanoramaHistory,
    PanoramaTarget, PanoramaPredictionRuntime,
)

history = PanoramaHistory()
for item in four_observed_time_points:
    history.append(ObservedPanoramaFrame(
        frame_id=item.frame_id,
        segment_id=item.segment_id,
        position=item.position,
        body_yaw=item.body_yaw,
        cube_rgb=item.cube_rgb,
        native_world_rgb12=item.native_world_rgb12,
        front_rgb=item.front_rgb,
    ))

runtime = PanoramaPredictionRuntime(
    encoder=frozen_navigation_dino_encoder,
    normalizer=existing_rae_normalizer,
    predictor=existing_frozen_native_cls_predictor,
    mode=selected_mode,  # 使用质量报告/配置中的受测推荐值
)
prediction = runtime.predict(
    [PanoramaTarget(0, "g7", tuple(target_position), target_absolute_yaw)],
    {0: history},
    generator=dedicated_nwm_generator,
)
```

`target_position`和`target_absolute_yaw`必须先固定，算法改变参考起点后不会改写它们。`prediction.meta['sources']`记录所选源位置、源朝向和是否回退。少于4帧时返回空预测和原因，调用方保留纯观测导航。传入`initial_noise`时，它必须与有效查询的输入顺序一致。

## 验证和复现

训练机新工作区使用`bash scripts/rgb_only_optimization_runtime.sh server ...`；环境没有安装或升级。完整实验入口和选择规则见`nwm-context-quality-plan-20260909.md`。`scripts/check_panorama_runtime.py`用真实全景、目标与固定噪声，把可复用接口和离线基准的同一输入对照，同时检查缓存第二次调用无需重新编码且结果相同。

本轮没有将该接口切换为生产导航默认，也没有重新训练策略或报告导航SR/SPL。将它接入导航事件流时，需要真正采集低级历史全景，保留段边界，并在训练检查点中新增上下文格式/模式/相机布局元数据；不能把当前单前置上下文的训练状态当作相同输入合同直接续训。
