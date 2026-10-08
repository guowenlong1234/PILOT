# 训练机旧低机位世界模型查找（2026-10-08）

用户回忆曾因参数配置错误训练低机位、窄视角模型。本次只读查找，发现高度吻合的旧低机位模型；窄视角尚未证实，不将用户回忆直接当作63°的证据。

## 找到的权重

训练机逻辑目录：`/home/gwl/project/RAE-NWM/raenwm/logs/raenwm_mp3d_96_20260416_172946/checkpoints/`。65000、70000、75000及latest均存在，每份约5.5GiB。训练日志始于4月16日，权重65000于4月19日、70000/75000于4月20日保存。

旧研究记录和`raenwm/docs/MP3D_H125_TRAINING_TIME_EVAL_COMPARISON_20260616.md`将该轮标记为old-0.88。历史对照通常采用70000步。

在ETP-R1训练机既有环境用`torch.load(...,map_location='cpu',mmap=True)`读取70000步文件成功。记录含model、ema、opt、args、epoch、train_steps、scaler、scheduler；epoch=46。EMA有314键，无CLS相关参数，`pos_embed`形状(5,256,768)，确认为4历史帧+目标帧的patch-only模型。未运行推理，未改环境。

## 相机和预处理证据

- 旧训练为640×480图像，旧`misc.py`执行CenterCropAR(4/3)→Resize224×224→归一化，与当前原生224方图不同。
- Git提交0ffbe77及bfb0709的`tools/collect_mp3d_from_rxr.py`默认640×480、相机高0.88m、HFOV90°。
- 70000 checkpoint只保存临时YAML路径`/tmp/raenwm_mp3d_96_20260416_172946.pnjV.yaml`，未内嵌完整训练YAML；不能用今天已修改的`config/raenwm_mp3d.yaml`当作当年准确配置。
- 原研究记录注明旧0.88数据根已经删除。当前未找到该轮实际采集summary或启动参数明确覆盖HFOV为63°的记录；现有可追溯历史代码也未搜到对应63°更改。因此“低机位”证据充分，“窄视角63°”仍未证实。4:3裁剪/缩放也不能直接等同于63°相机。

## 对后续工作的含义

这是值得测试的现成低机位候选，不是已经确认匹配RxR的替代权重。当前75k版本预测CLS+patch；旧模型仅patch，不能直接放进要求257 token的当前评分脚本。若继续比较，应使用patch兼容推理和相同相机条件，比较图块预测及CWP；如需整体特征，需另行明确解码再编码或适配方案，不能用平均patch冒充原生CLS。

本次没有启动训练或修改RAE-NWM/dino_cwp工程。结果只说明找到了旧权重和已知边界，不说明其在RxR上优于当前模型。
