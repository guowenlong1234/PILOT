# 两机最近任务核查（2026-09-28）

核查时间：北京时间2026-09-28上午。通过指定SSH入口，只读检查远端身份、进程、GPU、日志、状态文件和产物；未重启、恢复、停止任务或同步代码。以下时间统一为北京时间。

## 训练机

- 最新主任务：`rxr_dino_baseline_20260920`，9月21日10:28从2200步继续RxR DINO基座训练，双A6000、总批量24。
- 9月23日12:55:15正常结束，日志明确`exit_code=0`，完成30000步，末次IL_loss=0.743。
- `train_fast_resume_20260921/checkpoints/rxr_dino_baseline_20260920/`内2200—30000步每200步一份，共140份模型，无缺号；30000步模型和训练状态均存在。此次只核查文件及日志，没有重新加载大模型验证内容。
- 完整续训日志未检出Traceback、RuntimeError、CUDA error、OOM、段错误及独立nan/inf词。存在线程默认值、meshgrid接口、expandable_segments支持和模型类型兼容性提示，未导致任务失败。
- 此前2200步全量评测于9月20日19:24结束，独立退出码文件为0。未发现30000步模型的后续全量测评结果。
- 当前无训练进程，两卡GPU利用率0%。根分区99%、余33G；数据盘83%、余322G。

证据根：`/mnt/data2tb/ETP-R1_data/experiments/rxr_dino_baseline_20260920/`。主要证据：`train_fast_resume_20260921/supervisor_server/latest.log`、检查点目录，以及`eval_iter2200_dual_20260920_retry1/{finished_at,exit_code}`。

## 测评机

- 最新主任务：9200基座的未来特征内容对照，分别训练读取未来内容的full组与将该内容置零的none组，再做完整导航评测。
- 两组各6000步训练成功、退出码0；最终base/full/none三组各完整覆盖1839条路线，各自状态均completed、退出码0。最终导航流水线9月22日15:13:33完成。
- 历史失败保留：首轮8环境短采曾发生主机内存不足；旧导航流程报`ValueError: zero-gain logit difference 0.004474759101867676 exceeds baseline repeat 0.0`。`preparation.json`及`formal_v1/pipeline.json`仍保留failed，不能单独据此判定最终任务失败。
- `navigation_latest.json`明确指向`formal_v1/navigation_full_20260922`，其`pipeline.json`为completed/0。后续验收政策为`actions_metrics_zero_residual`，明确未强制旧logit差值上限（`logit_bound_enforced=false`）；两组短测动作、路线指标和零残差一致性通过。不能将其描述为原严格浮点分数门槛通过。

| 组别 | 导航成功率SR | 路径效率SPL | 路径一致性nDTW |
| --- | ---: | ---: | ---: |
| 9200基线 | 64.1653% | 54.7599% | 66.9786% |
| full | 63.5128% | 54.1772% | 66.4462% |
| none | 63.7847% | 54.2611% | 66.4116% |

任务运行完成，但本轮两组均未超过基线，不能视作效果提升。

其他近期任务：9月20日9200成对导航评测正常完成；6400旧导航首轮在358/1839报`ValueError: base_logits must not contain NaN`，其后`navigation_v2_geometry`成功结束（9月20日13:46）。

当前无训练、采集或导航评测进程，GPU空闲；仅旧TensorBoard和指标整理进程仍在运行。项目容器`gwl-etpr1-rae`运行，ETPNav容器已停止。磁盘93%、余67G。指定专线入口实际报告GPU为RTX3090 24GB，非约定俗称的4090；9月21日实验记录也已记载3090。

证据根：`/home/a6000/gwl/ETP-R1-stage2-e24/data/logs/stage2_9200_future_ablation_20260921/`。主要证据：`navigation_latest.json`、`formal_v1/navigation_full_20260922/pipeline.json`、各组`status.json`、`final_report/report.md`与`summary.json`。旧实验根：`stage2_online_9200_20260920/`及`stage2_online_20260920/`。
