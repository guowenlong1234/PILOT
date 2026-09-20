# 本轮62.86%基线与历史最佳的关系

用户追问后只读核对历史manifest、eval_summary、本轮launch/provenance和计划。未启动新的训练或GPU评测。

## 确认的模型身份

本轮基座是`ghost_concat_persistent_compiled_10k_20260911`的`ckpt.iter6400.pth`。历史训练机manifest与本轮测评机provenance均记录SHA256 `4c729c84bf4338452da4d459fc82734dcbb5f72ac6a2b574ee8f20e1080bc2fe`。本轮从`stage2_assets/ckpt.iter6400.pth`加载；该路径是同权重部署副本，并非改用较早模型。

9月16日计划固定6400的理由是该持久组SPL以及SR+SPL最高，并非全工程或该组SR最高。该组SR最高是9200（SHA `87bf7ad691a93abfe2d5030c2c314ef4e630c41d3abbe61fea3b38872686055e`）。本轮缓存和E24头都以6400为基座生成/训练，因此没有在E24比较时换9200。

| 记录 | SR | SPL |
| --- | ---: | ---: |
| 6400历史，训练机 | 63.6215% | 55.2559% |
| 9200历史，训练机 | 64.0566% | 54.3754% |
| 6400本轮重新评测，测评机 | 62.8603% | 54.8048% |
| 6400＋E24本轮，测评机 | 63.6215% | 55.3100% |

更早的DINO无世界模型14200起点历史记录为SR63.7303%、SPL55.6054%；不同阶段的GRPO还有更高SR。因此62.86%不应被表述为项目历史最佳基线。

## 已查出的运行差异与尚未定位的原因

历史6400实际在训练机A6000上评测，源码1f36528，`etpnav_unified`；本轮在测评机RTX3090上，源码1c0fb20，专用容器/`etpr1_rae`。二者均为完整1839条val_unseen、8环境、seed100、持久状态、world_exact_select/direct/FP16、Inductor、编码与预测批次64。

历史命令沿用`--train-policy`对应的加载开关：`IL.freeze_navigation_backbone=False`、`MODEL.RAENWM.rgb_fusion_trainable=True`；本轮为了冻结采集/离线头部署，分别为True/False。这些开关在eval中不代表发生了训练更新，但会经过不同的模型加载/包装及冻结路径。历史未显式传`EVAL.EPISODE_ID`；本轮显式列出全部1839条ID。上述差异已确认，但尚未通过单因素复现实验证明哪一项造成分数下降。

检查过融合网络本身是Linear/GELU，没有Dropout或BatchNorm，不能仅凭其trainable开关就声称找到性能下降原因。也不能未经实验就归因于GPU不同或普通随机波动。本轮两次完整基线的逐路线指标相同，62.86%是本轮协议下可重复的结果；这并不能替代历史63.62%的复现。

## 结论边界

同机同版本的本轮成对结果仍是62.86%→63.62%，但不能据此宣称超过历史最佳。与6400历史相比，E24的SR只持平；与9200历史相比，SR仍低0.4350个百分点。跨历史记录不能替代同条件配对实验。

此前最终报告没有突出说明6400历史成绩比本轮基线高0.7613个百分点，说明不完整，现补充本核对。后续若继续做归因，应先复现训练机历史6400的运行合同，再分别检查冻结加载路径与运行机器；本次只读核对不自动启动这些长实验，也不直接更换E24训练基座。

## 证据位置

- 计划：`docs/plans/2026-09-16-stage2-e24-training-plan.md`第1节。
- 训练机主工作区：`data/logs/ghost_concat_persistent_compiled_10k_20260911/eval_summary.json`，以及`eval/ghost_concat_v1_joint_persistent_eval_iter6400/manifest.json`，含实际SHA、命令与完整结果。
- 本轮测评机独立工作区：`data/logs/stage2_online_20260920/navigation_v2_geometry/full_base/{launch.json,provenance.json}`及`comparison.json`。
- 14200及其他历史阶段：`docs/latest-evaluation-status-20260914.md`、`docs/rgb-injection-navigation-audit-20260905.md`。
