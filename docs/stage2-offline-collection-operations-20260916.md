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
- 8环境预检进行中。早期按ID取前16/64条val样本时场景不足8，环境明确拒绝启动；已将正数`--episodes`短测子集改为按场景轮流选样。正式全量不裁剪数据，不受此改动影响。

## 运行环境与资源

训练机：Python3.10.14、PyTorch2.2.2+cu121、CUDA12.1、Transformers4.49.0、Habitat/Habitat-Sim0.3.3。两张A6000均无其他计算任务，数据盘启动前约777GiB可用。沿用经核验的项目内580.173.02库解决当前系统NVML用户态/内核版本不匹配，不改系统驱动：

```bash
export LD_LIBRARY_PATH="$PWD/.runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu"
```

测评机现场GPU为RTX3090 24GB，启动前空闲、根盘约218GiB可用；受保护`gwl-etpnav`容器已停止，未操作它。运行时版本写入每次任务的`run.log`。所有GPU任务经过项目共享锁与占用检查；单分片发布前保留至少20GiB磁盘。

## 正式采集（启动及完成状态待补充）

计划训练机GPU0/GPU1按episode ID交错分配train各5410/5409条；测评机负责全部1839条val_unseen作为开发缓存。统一8环境，独立输出目录。每次完整命令及源码版本写入`launch.json`和`provenance.json`，监督链状态写入`pipeline.json`。

预检通过后执行`run_stage2_collection.py collect ...`，自动完成校验和前向验收后退出。任何阶段失败即报告failed，不会自动启动训练或把未校验数据标为ready。
