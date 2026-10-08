# RxR相机扩大样本检查（2026-10-08）

承接小样本报告，用户要求扩大样本。四套相机、75k权重、后继预测器、3噪声和质量/几何指标不变，不训练。

## 固定设计

- 固定种子20261008；现存RxR val_unseen密集轨迹库满足长度≥45的10场景、439轨迹。
- 覆盖全部10场景，每场景随机最多20条，共160条轨迹。7场景各20条，其余为3、7、10条；不按结果筛选。
- 每条轨迹1/3、2/3两个位置各取4帧历史，目标后4/8帧，共640查询，是前轮48查询的13.33倍。
- 四相机×三噪声，共7680条世界模型输出；CWP真实图像2560条、预测图像7680条。
- 原生直接渲染与相同位置/方向/噪声成对；仍为固定轨迹诊断，不是导航候选分布测评。
- 训练机GPU0/1各200查询，测评机240查询，每个对照留在同卡。正式源码cb1d5a8。
- 世界模型按场景等权；CWP同时报告整体计数和场景等权比例，避免不同场景样本量主导结论。保留场景配对重采样区间。不能把三种噪声当独立位置。
- 全部输入ID×相机×噪声按manifest逐项验收；CWP真实/预测输入组合也要求完整唯一。

## 命令与产物

各机项目下`data/logs/rxr_camera_quality_expanded_20261008/`，含manifest、formal0/1/2目录及对应launch.json/log/exit。原图和特征留实际运行机器。

```bash
# 训练机生成清单
bash scripts/rgb_only_optimization_runtime.sh server scripts/rxr_camera_quality.py prepare --scenes 0 --trajectories 20 --output data/logs/rxr_camera_quality_expanded_20261008/manifest.json
# GPU0/1，SHARD=0或1
CUDA_VISIBLE_DEVICES=$SHARD bash scripts/rgb_only_optimization_runtime.sh server scripts/rxr_camera_quality.py run --manifest data/logs/rxr_camera_quality_expanded_20261008/manifest.json --output data/logs/rxr_camera_quality_expanded_20261008/formal$SHARD --shard $SHARD
# 测评机专用容器、项目环境
bash scripts/rgb_only_optimization_runtime.sh eval scripts/rxr_camera_quality.py run --manifest data/logs/rxr_camera_quality_expanded_20261008/manifest.json --output data/logs/rxr_camera_quality_expanded_20261008/formal2 --shard 2
```

新版汇总脚本在训练机原环境对旧48查询的三份结果重算，退出0；旧数据的计数合同仍通过。新的manifest完整性验收随正式汇总执行。环境版本同小样本报告，每份metadata仍独立记录实际版本和权重SHA。
