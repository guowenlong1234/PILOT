# 第二阶段预测future离线采集实施记录

本次用户授权：新建工作区/分支，训练机与测评机均可使用，止于离线数据可训练；不启动E24训练。

## 工作区及数据位置

- 分支：`feature/stage2-e24-offline`，基于主工作区`9d723c4`。
- 笔记本：`/home/sia/project/ETP-R1-stage2-e24`。
- 训练机：`/home/gwl/project/etpr1/ETP-R1-stage2-e24`，使用`etpnav_unified`和独立复制的`.runtime/server_sft`。
- 测评机：`/home/a6000/gwl/ETP-R1-stage2-e24`，使用`gwl-etpr1-rae`/`etpr1_rae`与独立复制的`.runtime/etpr1_habitat`。复制运行时内的8个绝对软链接已改为新工作区自有路径，未放宽所有权检查。
- 训练机`data/logs/stage2_e24`链接到`/mnt/data2tb/ETP-R1_data/experiments/stage2_e24_persistent6400_v1_20260916`；测评机数据保留在新工作区相同相对目录。
- 6400基座SHA256：`4c729c84bf4338452da4d459fc82734dcbb5f72ac6a2b574ee8f20e1080bc2fe`。测评机仅交付该单模型，校验后由`.partial`发布到`stage2_assets/ckpt.iter6400.pth`。
- 原主工作区、论文未提交修改、ETPNav工程/环境/容器均未修改。

## 实现边界

`MODEL.STAGE2_COLLECT.enabled`是只读采集模式，仅允许`run-type=eval`。旧`ACTIVE_LOOKAHEAD`保持关闭，采集器观察一阶段的融合后节点与分数，不接管动作。全阶段参数冻结并逐episode核验张量版本，结束时比较参数哈希；没有E24优化器、反向传播或训练更新。

q0快照在全景运行时实际构建输入的位置捕获，包含每查询实际顺序、源位姿、目标朝向和FP16历史token。q1复用该历史，采用独立、按episode/step/ghost确定的噪声；运行时逐步断言世界模型一阶段随机流未被q1消耗。CWP读normalized patch；E24数据空间保持现有原生链的`raw CLS + normalized patch`，不是把全部257token统一当raw。教师生成在预测future后进行，且保存/恢复Python随机状态。基础STOP、无可执行节点及达到最大步数的强制STOP均不请求q1，不产生移动重排训练行。

已观测历史位置的direct渲染仍属于一阶段原逻辑。q1从latent接口计算，不去未来位置渲染或查询真实目标；模拟器真实位置/目标距离仅用于教师标签，不能流入未来特征。

## 数据格式与后续读取

每条完整episode一个原子`.pt`文件及校验`.json`；指令特征按episode去重，future `[K,257,768]`为FP16，mask/教师/索引完整保存。文件只在episode完成后发布，崩溃留下的未提交文件移入`uncommitted/`保留，再重跑该episode。恢复只保证完整episode去重，不承诺剩余路线的一阶段随机流与未中断任务逐位一致；每次启动保留`attempts/`。

`Stage2Dataset`支持多个同split分片根，拒绝重复episode、不同基座或train/dev混读；惰性读取、最多缓存2条episode。`collate_stage2`直接输出E24网络和既有移动决策损失需要的字段。未来训练可使用：

```python
from vlnce_baselines.nwm.active_lookahead.stage2_data import Stage2Dataset, collate_stage2
train = Stage2Dataset(["<train_part0>/episodes", "<train_part1>/episodes"])
# 交给DataLoader(collate_fn=collate_stage2)，正式训练入口尚未实施。
```

只有完整`validate_dataset`通过才发布`dataset_manifest.json`。`check_stage2_train_ready.py`只做真实E24前向、零初始化输出及损失有限性检查，不做backward。`run_stage2_collection.py`固定依次采集→全量校验→只前向验证→退出，没有train子命令。

## 已完成检查

- 训练机相关测试61项通过；测评机数据/全景/TopK/E24相关测试36项通过，均退出0（详细命令见本轮执行记录，后续最终报告补充日志位置）。
- 2条真实train路线：19行决策、92槽、42有效future，13可训练行；971个导航/融合/路点状态张量逐项一致。数据已校验，60854785参数的真实E24前向和损失计算成功，loss约0.60145，没有反向或优化器。
- 相同2条路线，采集开/关的19步动作、所有基础有限logits逐位相同。
- 训练机4环境编译模式16条路线：132行，644槽，318有效future，约0.239GiB。与同配置关闭采集的132步动作、基础logits全部一致。导航等971张量、世界模型314张量前后一致。
- 已确认正式train有10819条、val_unseen有1839条，全部都有对应GT。
- 8环境预检已完成，详见下方正式启动记录。早期按ID取前16/64条val样本时场景不足8，环境明确拒绝启动；已将正数`--episodes`短测子集改为按场景轮流选样。正式全量不裁剪数据，不受此改动影响。

## 运行环境与资源

训练机：Python3.10.14、PyTorch2.2.2+cu121、CUDA12.1、Transformers4.49.0、Habitat/Habitat-Sim0.3.3。两张A6000均无其他计算任务，数据盘启动前约777GiB可用。沿用经核验的项目内580.173.02库解决当前系统NVML用户态/内核版本不匹配，不改系统驱动：

```bash
export LD_LIBRARY_PATH="$PWD/.runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu"
```

测评机现场GPU为RTX3090 24GB，启动前空闲、根盘约218GiB可用；受保护`gwl-etpnav`容器已停止，未操作它。运行时版本写入每次任务的`run.log`。所有GPU任务经过项目共享锁与占用检查；单分片发布前保留至少20GiB磁盘。

## 正式采集（已完成）

实际训练机GPU0/GPU1按episode ID交错分配train各5410/5409条；测评机负责全部1839条val_unseen作为开发缓存。统一8环境，独立输出目录。每次完整命令及源码版本写入`launch.json`和`provenance.json`，监督链状态写入`pipeline.json`。

预检通过后执行`run_stage2_collection.py collect ...`，自动完成校验和前向验收后退出。任何阶段失败即报告failed，不会自动启动训练或把未校验数据标为ready。

### 2026-09-16 正式启动

源码版本`1de707d`。测评机17:28:26启动`formal_dev`，监督PID2609768；训练机17:30:57启动`formal_train_part0`/`formal_train_part1`，监督PID509708/509709。以上PID只作为历史记录，后续须核对实际命令。三份任务由独立会话托管，采集、完整校验、真实E24只前向验收依次执行，不进入训练。

最终预检：训练机8环境64条路线，494行、2400槽、1108有效future、355可训练行，956548974字节；测评机8环境跨场景16条路线，113行、543槽、260有效future、76可训练行，216435773字节。两机全部校验和实际E24读取通过；导航等971张量、世界模型314张量不变。新增分片统计故障注入测试在两机均通过（6项数据测试）。预检数据不混入正式数据。

完整命令在三个正式根的`supervisor.json`/`launch.json`。正式训练数据范围固定为10819条，两份无交叉；开发数据为1839条。文件数增长不等于ready：必须等`pipeline.json.status=ready`、`dataset_manifest.json.status=validated`及`train_ready.json.status=ready`都成立，再做跨分片覆盖核验。

### 验证命令索引

训练机实际工作区：

```bash
bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q \
  tests/test_stage2_data.py tests/test_panorama_runtime.py \
  tests/test_ghost_concat_persistent.py tests/test_ghost_concat_trainer.py \
  tests/test_stage0_e24_joint.py
# 61 passed，退出0；随后新增分片计数故障注入后，单独重跑数据测试6 passed，退出0。

bash scripts/rgb_only_optimization_runtime.sh server scripts/check_stage2_train_ready.py \
  data/logs/stage2_e24/preflight_final64/episodes \
  --report data/logs/stage2_e24/preflight_final64/ready.json
# 494行可读取；真实E24 forward/loss通过，退出0，无backward和optimizer。
```

测评机上述运行时入口改为`eval`，必须在`gwl-etpr1-rae`内执行。其36项回归命令的测试文件为`test_stage2_data.py`、`test_panorama_runtime.py`、`test_stage0_topk_query.py`、`test_stage0_e24_joint.py`；数据测试补充后另6项通过。

两机预检的精确导航命令保存在各自`launch.json`，退出码在`status.json`，环境版本和导航关键输出在`run.log`。全量数据校验读取该次`provenance.json.episode_ids`为预期清单；历史短测曾采用ID前缀，新短测改为按场景采样，因此校验旧短测不能按新版采样规则重建另一份清单。

### 开发集完成

2026-09-16约19:05，测评机`formal_dev`流水线自动完成并退出，GPU显存归零。另行核验全部1839个episode ID与正式val_unseen完全一致，覆盖11个场景，无重复/缺失。数据15703行，10997行满足训练损失有效条件；75360个Top-5槽位中36992个future有效；大小30024107716字节（约27.96GiB）。此处有效率分母包含STOP及历史不足行，不应直接与旧Oracle覆盖率比较。

- `episodes/dataset_manifest.json` SHA256：`25d9666e89e482186b9d6f7436dcf5c9934f82cd9265fa0658c0f1ec632ba93d`。
- `episodes/freeze_report.json`：971个状态张量完全一致；`world_freeze.json`：314个世界模型张量完全一致。
- `train_ready.json`：真实E24只前向及损失检查通过，未调用backward、未创建optimizer、未开始训练。
- `verified_summary.json`：额外核验后的紧凑总表。

开发集数据留在测评机独立工作区`data/logs/stage2_e24/formal_dev/episodes`。当时训练集仍在采集中；最终两份训练分片也已通过验收，见下方最终汇总。

开发集额外标签审计（`label_coverage.json`，脚本`scripts/summarize_stage2_labels.py`，退出0）：有效教师且基础/教师均MOVE的决策13389条；教师在Top-5中12052条（90.0142%），教师候选future有效7586条（56.6585%）；基础选择错误且教师future可用1984条。后者尚未考虑残差上限能否跨越分数差，不能称为可实现收益。所有15703条教师标签均有有效动作；有效损失行10997与主manifest完全一致。标签统计只读缓存，无优化或训练。

收尾代码补充：`1516498`对旧模型配置缺少采集字段的情况关闭trace，测评机62项相关回归通过；它不改变本次采集输入/输出。`751c2d9`新增独立标签审计脚本。正式数据的采集源码仍记录为`1de707d`。


## 最终交付：数据就绪，训练未启动

2026-09-16 21:42（北京时间）完成三份数据的流水线验收，另行核对训练分片并集与完整train清单完全一致。训练集61个场景与开发集11个场景无交集。所有采集监督进程均退出，训练机两卡只剩桌面显示占用、计算利用率0；测评机计算进程已退出、显存归零。没有启动E24训练。

| 数据 | 路线数 | 决策行数 | 有效损失行 | 有效future槽 | 大小 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train part0 | 5410 | 41282 | 29173 | 97383 | 73.37GiB |
| train part1 | 5409 | 41392 | 29383 | 98140 | 73.65GiB |
| train合计 | 10819 | 82674 | 58556 | 195523 | 147.02GiB |
| dev（val_unseen） | 1839 | 15703 | 10997 | 36992 | 27.96GiB |

“有效损失行”指教师有效、基础与教师均继续移动、仍有可执行候选且至少一个预测future有效；包含教师未在有效Top-5中的负向/少改动监督。其余行保留用于完整轨迹和覆盖率审计，不假装全部可用。

实际数据目录：

- 训练机：`/mnt/data2tb/ETP-R1_data/experiments/stage2_e24_persistent6400_v1_20260916/formal_train_part0/episodes/`。
- 训练机：同根下`formal_train_part1/episodes/`。
- 测评机：`/home/a6000/gwl/ETP-R1-stage2-e24/data/logs/stage2_e24/formal_dev/episodes/`。

每个目录含逐episode的`.pt`、SHA/覆盖率侧文件及完整`dataset_manifest.json`。训练机`verified_train_summary.json`记录并集、场景隔离、模型冻结及所有计数；测评机`formal_dev/verified_summary.json`保存对应开发集证明。三个`pipeline.json`均为ready，三个`train_ready.json`均证明真实E24前向和损失可计算，且没有backward或optimizer。

两个训练manifest SHA256：

- part0：`9ce7d8b55dc467679ce8e420bbf81964e4201def9f35267089a9abd645d24b5e`。
- part1：`f242510d30dbed8e893a744dc608e513fb43bf8fc9f9630d02af9cd8fdbc5b7b`。

三个任务均核验导航/融合/路点971个状态张量、世界模型314个状态张量前后完全一致。全部数据完成逐文件SHA、逐行shape/有限值/索引/标签映射及覆盖计数验证。两机`input_fingerprints.json`额外记录train/dev源JSON、GT及DINO配置/预处理配置的SHA与mtime，两机对应文件完全一致；视觉模型权重SHA也在每次加载基座时被检查。

正式数据采集源码为`1de707d`。后续兼容性保护和只读标签审计不改变已生成数据；最后的62项相关回归在测评机专用环境退出0。完整命令、版本、退出码、逐路线指标与验收结果均保留在各实验目录。本轮不把数据或模型复制回笔记本，不将大产物提交到Git。

训练集合并审计也已完成（`train_label_coverage.json`）：10819条路线、82674行均可读取，70761条基础/教师均MOVE的有效决策中，教师Top-5覆盖率99.5902%，教师future覆盖率70.8794%；其中基础选择错误且教师future可用3791条。教师无效行0，有效损失行58556与两个manifest之和完全一致。合并读取器和实际E24只前向检查退出0，记录在`combined_train_ready.json`；没有训练更新。这些是数据诊断，不是E24性能收益。

后续如要训练，应使用两份完整train缓存，开发集保持独立；不混入任何预检目录。实际训练入口及优化器/训练恢复验收仍属于下一阶段，本轮按用户要求止于数据就绪。
