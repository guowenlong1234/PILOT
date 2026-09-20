# E24 最佳离线头接回在线导航（2026-09-20）

用户授权把当前表现最好的方案接入导航并比较导航指标。固定方案为一阶段6400步基座、原离线4750步E24头、倍率1.5，并逐候选裁剪到±1。权重SHA分别为`4c729c84bf4338452da4d459fc82734dcbb5f72ac6a2b574ee8f20e1080bc2fe`和`598986525cb3ac4696743b0480ac6733b918e3d4407d647c4cddb0a498ae286e`。

## 接入方式

新增默认关闭的`MODEL.STAGE2_ONLINE`，仅允许eval，禁止与采集/旧ACTIVE_LOOKAHEAD/教师可视化同时启用。所有参数冻结，无优化器。`stage2_collect.py::predict_step`由采集和在线共用，只生成未来预测、不调用教师、不访问标签写入器。在线使用原q0快照、原预测路点、相同显式噪声流和raw CLS + normalized patch，不渲染未来真值。评分使用FP16缓存往返和BF16头推理，与原离线评价一致。仅有效future候选可改分，基础STOP、强制STOP与无可执行候选不改分；移动时仍在全部候选中选择。

## 固定评测协议

测评机`/home/a6000/gwl/ETP-R1-stage2-e24`，容器`gwl-etpr1-rae`、环境`etpr1_rae`；启动前GPU空闲，受保护ETPNav容器停止。现场卡为RTX3090 24GB，仍使用约定的专线入口。完整R2R val_unseen共1839路线，8环境；基线和最佳头相同机器、配置、随机种子、数据清单、编译及精度。每项使用独立新输出目录，不覆盖旧结果。本轮不训练、不扩展为RxR评测。

顺序：真实缓存重放 → 16路线纯基线 → 同16路线倍率0 → 同16路线倍率1.5 → 完整纯基线 → 完整倍率1.5。零倍率要求逐步一阶段logits/动作与逐路线导航指标完全相同。任一检查失败就停止，不自动把未完成结果当成功。

评价指标：success（路线成功率）、SPL（成功且路径高效的程度）、nDTW/SDTW（与参考路径接近程度）、目标距离、路径长度、碰撞及耗时。耗时分别记录整个进程和实际rollout，两组初次预测编译均计入rollout，不能当稳态单步延迟。开发集已参与离线选点，不作为独立测试集。

## 已完成验收及运行位置

计算接入源码`73afe93`，有限评测队列`76c0312`。测评机相关测试53项退出0，日志`data/logs/stage2_online_20260920/tests.log`。真实开发缓存前4路线共34行，在线评分与离线BF16推理逐位相同；移除全部teacher字段后仍可执行，倍率0修正严格为0；重放worker退出0，记录`replay.json`和`replay_job/`。

队列输出根`data/logs/stage2_online_20260920/navigation_v1`，父进程启动命令和PID在同级`navigation_v1_launch.json`。`pipeline.json`记录当前阶段及精确命令，各子目录`launch.json`/`provenance.json`/`status.json`/`run.log`保存配置、数据、退出码和环境版本。短程trace只写小型决策日志，不写大体积future缓存。正式评价使用`full_base`、`full_best`，逐episode结果与汇总保存在各自`results/`；在线冻结审计、改动作数量和运行耗时在`online/`。

当前：输入重放及CPU回归通过，已启动真实短程检查；完整指标待队列通过并完成后填写。
