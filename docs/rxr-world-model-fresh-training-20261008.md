# RxR世界模型从头训练与旧检查点清理（2026-10-08）

最新现场：10月8日11:23（北京时间）自动接力进入正式双卡训练。4220轨迹、427685帧及CLS/patch预计算全部完成，完整性与真实数据读取验收均退出0。日志11:23:42明确FRESH_START，11:23:44开始epoch0；每轮1510次更新、共50轮。此次核查时两卡利用率100%、显存各约20.7GB；尚未出现首个每100步进度日志，正式检查点目录仍为空，首份按计划在1000步生成。下文“等待数据”为启动前记录。

用户最终约定：RxR相机0.88m/63°/224²，世界模型从随机初始化训练；其余沿用原CLS世界模型。双A6000训练，有效batch96，50轮，seed42，新实验每1000步全部检查点保留。允许按指标清理旧世界模型与旧导航实验。

## 实施入口

世界模型独立Git分支`feature/rxr-camera-fresh`，中央仓库`server:/home/gwl/git/raenwm.git`。笔记本工作区`/home/sia/project/raenwm-rxr-fresh`，训练机`/home/gwl/project/RAE-NWM/raenwm-rxr-fresh`；原主工作区及用户未提交修改不动。

完整操作记录见该工作区`docs/RXR_FRESH_TRAINING_20261008.md`。配置`config/raenwm_rxr_scratch_20261008.yaml`；数据生成`tools/prepare_rxr_camera.py`；自动接力`tools/run_rxr_fresh_pipeline.py`。

运行根：训练机`/mnt/data2tb/ETP-R1_data/experiments/rxr_world_model_fresh_20261008/`。

- 原4220轨迹427685帧、位姿及划分名单不变，重新渲染JPEG100/无色度降采样，再用原冻结编码器和统计生成原生CLS+patch特征。
- 双卡每卡8×累积6=96，学习率2e-4→2e-6线性，模型结构、损失、优化器与旧实验一致。评估双卡每卡4，保持旧全局batch8。
- 从头训练守卫禁止已有latest/from_checkpoint；所有新模型保留。latest采用硬链接原子切换，避免重复占用。
- 初次数据生成JPEG色度降采样设置不符，已在正式训练前停止自有采集进程、归档失败标记并从头按正确参数重生成；原数据不动。
- 双卡2步训练、保存/评估，以及跨进程恢复到3步通过。恢复调度器采用显式读取保存状态，避免旧默认重推导致少一步；正式从头学习率曲线不变。
- 真实全量数据结构预检：训练144945、内部验证13267，退出0；完整特征和JPEG计数/SHA校验须等采集结束后执行。
- 自动接力等待全量数据完整、短测通过及磁盘容量足够，再启动50轮正式训练。任何门槛失败会记录failed并停止，不绕过检查。

查看`pipeline.json`判断当前阶段；`preparing_data`不表示正式训练已开始。正式训练日志`training.log`，模型目录`training/rxr_h088_fov63_224_96_e50_scratch_20261008/checkpoints/`。

## 已执行的旧检查点清理

用户明确授权包括旧导航实验后，按完成测评数据挑选各5个代表点：保留SR、SPL、nDTW、SDTW最佳，再按SR+SPL补齐。仅针对以下三个已完成实验：

| 旧实验 | 保留的代表点 |
|---|---|
| ghost_concat_direct_compiled_10k_20260909 | 600、2000、4600、6600、7400 |
| ghost_concat_persistent_compiled_10k_20260911 | 6200、6400、6600、7600、9200 |
| rxr_dino_baseline_20260920 | 13600、15200、21600、29200、29600 |

RxR原2200步两份历史对照模型额外保留，训练恢复状态不删。总计保留17个模型文件，删除234个已测完且不入选的中间模型，逻辑大小335.21GiB。执行后数据盘空闲约479GiB（后续数据、短测与编译缓存继续占用）。

清理脚本`scripts/prune_completed_navigation_20261008.py`只允许三个指定根目录，执行前核对文件大小/inode/修改时间、符号链接依赖，并记录保留模型SHA。日志和全量测评结果没有删除，原14200基座不在清理范围。旧世界模型已各剩3—5份，不再额外删除。

完整审计：训练机`/mnt/data2tb/ETP-R1_data/experiments/checkpoint_cleanup_20261008/`，含plan.json、各组selection.json、kept_sha256.json、deleted.jsonl、apply.log（DELETED 234 KEPT 17）。没有清理新实验检查点。
