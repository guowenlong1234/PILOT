# 有序未来证据与有界渐进修正

2026-10-09。本次是代码与 CPU 单元/模拟测试交付；没有启动正式采集、训练或导航评测，没有新的导航性能结论。默认仍为旧路径。新模型名称 `progressive_e24_v1`；旧 E24 权重不能直接恢复为新模型，未提供隐式 warm start。

## 实际接通的调用链

`stage2_e24_job.py collect --progressive none` → trainer 创建 `ProgressiveCollector` → 复用第一阶段 q0 缓存 → `ProgressivePrediction.initialize_decision` → 调用 D 次 `predict_next_depth` → `finalize_decision` → **之后**调用原 teacher 标签函数 → 独立 episode writer。

`progressive_stage2.py validate/train` → 新数据验证/动态补齐 → `ProgressiveE24Head.initialize_state` → 同一个 `step` 因果循环 → 原决策/候选对损失读取累计修正 → 保存新 checkpoint 与优化器/RNG/数据游标。

`stage2_e24_job.py online --progressive certified` → `ProgressiveOnline` 校验 head SHA、基座 SHA、世界模型/统计/CWP/世界模型配置 SHA、预测设置、特征及几何尺度 → `ProgressiveController` → d=0 证书 → 预测一层 → 评分一层 → 证书 → 必要时下一层。trainer 的贪心动作直接采用控制器结果，**不会经过旧 gain/clip，也不会在低精度下重加残差**。

q0 是候选到达位置，已用于第一阶段增强。d=1、2 是候选之后的物理后继，不是 `fusion_layers`。`owner_embeddings` 来自实际导航输出的 `gmap_embeds`，其输入已包含第一阶段候选增强。第二阶段 memory 只在一次 `score_step` 内创建和使用；不回写拓扑节点、q0 缓存或真实历史。

## 数学和保证范围

初始化 `S[0]=base_logits`。共享融合模块读取指令、当前 CLS/patch 和上一层候选 memory，然后进行候选间比较及层内反馈。

```
z_tilde = visual_projection(z) + depth_embedding(d) + geometry_mlp(g)
u[d] = tanh(score(memory[d], base_log_probability, cumulative_delta, S[d-1], depth))
beta[d] = total_residual_bound * budget_fractions[d]
delta[d] = beta[d] * u[d]
S[d] = S[d-1] + delta[d]
```

没有候选槽位专属编码，没有跨候选减均值，没有预算回收。末层零初始化：初始模型零修正；第一步通常只有最终评分层获得非零梯度，更新后上游融合/记忆才获得非零梯度。新 memory 不 detach；导航、语言和预测输入 detach。物理深度共享整个融合/比较模块，层内融合仍有独立轮次参数。

只有本层确实得到有效预测的候选更新 memory 和分数。其他真实候选保留已有 memory，仍作为候选比较上下文。固定 Top-K 外所有可执行候选仍参与最终比较，但预算为 0。STOP 保留现有隔离规则。

未终止的 Top-K 分支剩余预算 `R[i,d]=sum(beta[d+1:])`；已确定终止的分支后续不再提交增量，R=0。尚未查询的深层不能当作终止。

仅在 `S[w,d]-R[w,d] > max(S[j,d]+R[j,d]) + guard` 时认证提前停止。比较使用 FP64，guard 至少 `max(1e-6,16*eps_fp32*max(1,max(abs(S)+R)))`；实际累加和动作分数使用 FP32，严格并列或临界分差继续计算。性质测试覆盖合法增量下区间逐层包含。

保证只针对**相同 progressive 模型、相同最大深度的贪心 argmax**；不保证分数、概率分布相等，也不保证导航正确。`pruning_mode=none` 不使用任何证书或经验门控提前退出，但保留原 STOP、自然终止。`certified` 也不进行按候选胜率删证据。

## 特征、坐标和预测契约

特征常量集中在 `progressive_contract.py`：257 个 token，768 维；第一个是 raw CLS，后面是 normalized patch。写盘 FP16、评分读取 FP32；不做 RGB 重建，不改变原特征归一化。q0 从第一阶段缓存读取，不重新预测。

第 d 层 CWP 读取 d-1 层 normalized patch，提出路点。世界模型每次仍读取该候选 q0 保存的**原始真实观测上下文快照**，仅目标位置/朝向/horizon 改变。没有把预测追加为真实历史。horizon 按路段长度/原运行时路点间距累计，严格遵循原 min/max 范围。

局部坐标遵循现有路点函数：令 `h=q0_yaw-pi`，世界水平位移为 `(dx,dz)`，

```
local_dx = dx*cos(h) - dz*sin(h)  # 正路点角方向
local_dy = dx*sin(h) + dz*cos(h)  # q0前向
relative_yaw = target_yaw - q0_yaw
g = [local_dx/scale, local_dy/scale,
     sin(relative_yaw), cos(relative_yaw), cumulative_length/scale]
```

Habitat yaw=0 朝向世界 -z；正 90° 路点转向 -x。世界 y 是竖直轴，不能直接用世界 x/y 作水平坐标。累计路径从 q0 开始，不包含智能体到 q0 的距离。原 `candidate_q0_geometry` 单独保存并投影。几何定义与 distance_scale 均进入数据/checkpoint 校验。

请求噪声采用 SHA256(JSON(seed,scene,episode,真实步,candidate稳定ID,q0来源步,物理深度))，私有 `torch.Generator` 生成。无 Python hash、无候选槽位依赖。首版 CWP 和世界模型固定单候选请求批次，避免其他环境提前停止改变计算批次形状；这牺牲部分吞吐，尚未做真实测速。全局 RNG 用 fork 隔离，第一阶段 runtime.generator 另做前后检查；冻结参数做版本和最终 tensor manifest 校验。第二阶段不保留跨决策预测缓存。

## 数据与 mask

新 schema 为 `etpr1-progressive-episode-v1`，独立于旧 `stage2-panorama-predicted-v1`，不会改写旧数据。张量深度索引 0 对应物理 d=1。批量形状：

| 字段 | 形状/含义 |
|---|---|
| future_tokens | `[N,K,D,T,C]` |
| future_geometry | `[N,K,D,5]` |
| candidate_present_mask | `[N,K]`，真实候选槽存在 |
| future_valid_mask | `[N,K,D]`，已得到合法预测 |
| future_token_mask | `[N,K,D,T]`，该预测内部 token 有效 |
| future_terminal_mask | `[N,K,D]`，本层之后分支已确定不能继续 |
| future_queried_mask | `[N,K,D]`，实际尝试该层（包括 CWP 后未调用世界模型的失败） |
| topk_base_indices | `[N,K]`，固定到完整可执行候选集合的映射 |

同时存完整候选 ID/base logits、teacher 映射、文本、q0 几何及来源、目标位姿/horizon、请求身份和终止原因。预测层另有 `unqueried_mask`，未执行层不会伪装成完整预测。全量采集不使用评分证书；缺 q0、CWP none、horizon 超范围、非有限预测等自然终止有显式原因。无 q0 的来源明确记录为 `source_unavailable`，不伪造来源。

无效预测不进入带偏置视觉投影；无效 token 在投影前清理并有 attention mask。合法未来要求 CLS 和至少一个 patch。无效占位可含 NaN，不污染其他候选。完整数据验证拒绝终止后恢复、缺元信息、不完整深度伪装为完整采集、分片 SHA 变化和 teacher 映射错误。

训练只创建新评分头优化器。最终监督权重 1.0、浅层前缀权重 0.2（多个浅层取均值），每个前缀监督同一次真实导航决策，传入累计修正。增量平方正则默认 0.001，只计一次。teacher 不在 Top-K、STOP、无有效未来复用原目标过滤规则。没有每层必须改善/赢家保持不变的损失。AdamW 使用固定学习率；保存优化器/RNG/确定性数据游标，scheduler 为显式 None。恢复使用 `--resume`，不覆盖已有 checkpoint。

## 配置和旧新切换

全局 `MODEL.PROGRESSIVE.enabled=False`。启用时默认 D=2、总预算 1、比例 `[0.5,0.5]`、共享深度权重、深度/几何编码均开、distance_scale=1、`pruning_mode=certified`。支持 D=1/2/3，D 改变必须使用匹配数据和新 checkpoint，并显式给出对应长度预算；不提供旧数据自动复制/补深层适配。

新路径要求 `STAGE2_ONLINE.gain=1`、`margin_threshold=-1`，否则明确报错；没有重复 clip。旧模式 gain/clip 和 SHA 校验未改。

- `configs/progressive/legacy.yaml`：旧 native-9200 E24 示例。
- `configs/progressive/full.yaml`：新 D2 完整展开。
- `configs/progressive/certified.yaml`：只将上一文件的 `pruning_mode` 改为 certified。
- `configs/progressive/model.json`：离线模型结构。

YAML 是叠加配置，应与 `run_r2r/iter_train_rae_dino_ghost_concat_persistent.yaml` 一起传给 `run.py --exp-config 基础.yaml,叠加.yaml`，并提供 head、head_sha256 和输出路径。下面的作业入口会自动设置等价覆盖项，并计算真实 head SHA。`--progressive` 作业入口当前固定 D2；更改 D 使用显式 run.py 配置及匹配 provenance/model JSON。训练辅助损失通过 CLI `--prefix-loss-weight` 设置，配置中的训练默认值为文档化对应值。

## 后续实际运行命令

代码尚未提交或同步；应先审查，再按项目 Git 规则交付到**实际运行机器**。以下例子在测评机宿主机 `/home/a6000/gwl/ETP-R1` 执行；作业入口自动进入 `gwl-etpr1-rae`/`etpr1_rae` 并检查 GPU/受保护任务。若选择训练机，应在其工程目录改用 `--machine server` 和该机器实际环境，不能把本机 CPU 测试当成远端验收。

须先填写实际基座路径。当前作业入口沿用原工程批准的 9200 SHA `87bf7ad691a93abfe2d5030c2c314ef4e630c41d3abbe61fea3b38872686055e`（也支持原 6400）；不能用其他 checkpoint 冒充。以下其余路径相对于目标工程：

- `pretrained/raenwm_native_cls/checkpoint_step_75000.pth.tar`
- `pretrained/raenwm_stage0/stat.pt`
- `pretrained/active_lookahead/dino_cwp_best.pt`
- `configs/nwm/raenwm_mp3d_fresh_cls.yaml`
- 原 R2R/MP3D 数据、导航环境与视觉编码器资产。

若这些资产不存在，先提供真实文件并核对原配置 SHA；不要填假路径或放宽校验。head 路径由新训练产生，不能填旧 E24 head。输出必须是新实验目录。

```bash
cd /home/a6000/gwl/ETP-R1
BASE=/填写真实路径/ckpt.iter9200.pth
RUN=data/logs/progressive_d2_v1

# 先做4条路线全深度采集；验收后再去掉 --episodes 4 扩大采集。
python scripts/stage2_e24_job.py collect --machine eval --gpu 0 \
  --environments 1 --base-step 9200 --checkpoint "$BASE" --split train \
  --episodes 4 --output "$RUN/train_smoke" --progressive none \
  --gain 1 --margin-threshold -1 --no-compile

# 该命令在新容器/专用环境中做完整数据校验。
python scripts/stage2_e24_job.py validate --machine eval --base-step 9200 \
  --checkpoint "$BASE" --split train --episodes 4 --output "$RUN/train_smoke" \
  --progressive none --gain 1 --margin-threshold -1
```

离线命令在实际任务环境中执行；测评机入口例如：

```bash
# 从宿主机进入专用环境；以下训练例子只做2个更新。
docker exec -it -w /home/a6000/gwl/ETP-R1 gwl-etpr1-rae bash
source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh
conda activate etpr1_rae
RUN=data/logs/progressive_d2_v1
python scripts/progressive_stage2.py train --data "$RUN/train_smoke/episodes" \
  --model-config configs/progressive/model.json --output "$RUN/train" \
  --steps 2 --batch-size 2 --device cpu --prefix-loss-weight 0.2
python scripts/progressive_stage2.py train --data "$RUN/train_smoke/episodes" \
  --model-config configs/progressive/model.json --output "$RUN/train" \
  --resume "$RUN/train/state_step_000002.pt" --steps 1 --batch-size 2 --device cpu
python scripts/progressive_stage2.py evaluate --data "$RUN/train_smoke/episodes" \
  --checkpoint "$RUN/train/head_step_000003.pt" --output "$RUN/offline_smoke.json"
python scripts/progressive_stage2.py replay --data "$RUN/train_smoke/episodes" \
  --checkpoint "$RUN/train/head_step_000003.pt" --pruning-mode certified \
  --output "$RUN/replay_certified.json"
python scripts/progressive_stage2.py replay --data "$RUN/train_smoke/episodes" \
  --checkpoint "$RUN/train/head_step_000003.pt" --pruning-mode none \
  --output "$RUN/replay_full.json"
exit
```

以上同训练集评价只是接口烟测。正式离线评价须另外用 `collect --split val_unseen` 生成冻结基座的完整深度数据，再用 `evaluate/replay --data` 指向该目录；不能将训练集准确率当成泛化结果。正式训练扩大 `--steps`，GPU 使用前按项目约定检查资源，再使用 `--device cuda:0`，本任务没有自动执行。

同一新 head 的小规模导航对照（宿主机）：

```bash
HEAD="$RUN/train/head_step_000003.pt"
python scripts/stage2_e24_job.py online --machine eval --gpu 0 --environments 1 \
  --base-step 9200 --checkpoint "$BASE" --head "$HEAD" --split val_unseen \
  --episodes 4 --output "$RUN/nav_full" --progressive none \
  --gain 1 --margin-threshold -1 --trace --profile --no-compile
python scripts/stage2_e24_job.py online --machine eval --gpu 0 --environments 1 \
  --base-step 9200 --checkpoint "$BASE" --head "$HEAD" --split val_unseen \
  --episodes 4 --output "$RUN/nav_certified" --progressive certified \
  --gain 1 --margin-threshold -1 --trace --profile --no-compile
```

完整评测将两条命令的 `--episodes 4` 改为 `--episodes -1`，输出分别改为新的 `nav_full_all`、`nav_certified_all`。先完成真实环境导入、冻结检查、小采集/恢复/重放及单路线验收，再启动长任务。本文不授权自动启动。

## 日志与验证结果

`online/decisions.jsonl` 记录 executed_depth、stop_reason、per_depth_delta、cumulative_delta、base/prefix/final winner、remaining_budget、每层 certificate、certificate_margin、numerical_guard、世界模型请求数、有效未来数、prediction/scoring/total_stage2_seconds。winner 是完整可执行移动候选集合的局部索引，可用 ghost_ids 解析；executed_action 是真实导航索引。`current_scores` 明确标为已执行前缀，不声称是未生成深层的最终完整分数。STOP、自然终止、认证、最大深度、异常失败分开记录。

GPU 计时前后同步。各行 prediction_seconds 包含其活跃深度的整批预测耗时，不能把多行相加当墙钟时间；total_stage2_seconds 是整批耗时。固定单请求策略尚未测速，平均执行深度降低不代表实际加速。

本机 CPU：Python 3.13.13，PyTorch 2.11.0+cu130（CUDA build 13.0），pytest 9.1.1；未安装 Transformers、Habitat、Habitat-Sim、yacs。`python -m pytest -q tests/test_stage2_inference_gate.py` 因缺 yacs 在收集阶段失败。专用 CPU 入口只跳过顶层 Habitat 注册，所有被测工程模块均为真实源码，没有替代实际实现。

```bash
python scripts/test_progressive_cpu.py -q tests/test_progressive* tests/test_stage2* tests/test_stage0_topk_query.py
```

最终完整运行：**142 passed，6 条旧路径弃用警告，退出码 0（9.51 秒）**。日志补充后控制器/预测/入口针对性复测 **18 passed，退出码0**。`git diff --check` 和两个命令行入口 `--help` 均退出 0。

该命令涵盖模型/数学/数据/控制器/模拟预测、真实 CLI 合成数据训练2步→恢复第3步→评价/重放、新旧配置/作业生成器及旧路径回归。测试中的世界模型和导航资产明确为 mock，不能当作真实导航结果。曾发现逐列表精确比对因 PyTorch 有/无梯度 attention 路径出现约 1e-8 差异，已改为明确 1e-6 容差；动作认证仍使用独立保守余量。

未执行：真实 Habitat 导入与运行、真实 q0/CWP/世界模型预测、GPU 前向/训练、真实多深度数据采集、真实冻结 manifest 验收、完整导航和性能/耗时对照。没有虚构 SR/SPL 提升。


## 本次文件清单

| 范围 | 实际文件 |
|---|---|
| 新模型和数学 | `vlnce_baselines/nwm/active_lookahead/progressive_contract.py`、`progressive_core.py`、`progressive_head.py` |
| 新数据与训练 | 同目录 `progressive_data.py`、`progressive_training.py` |
| 新预测与部署 | 同目录 `progressive_prediction.py`、`progressive_controller.py`、`progressive_collect.py`、`progressive_online.py` |
| 现有接入点修改 | `vlnce_baselines/config/default.py`、`vlnce_baselines/ss_trainer_ETP_R1.py`、`scripts/stage2_e24_job.py` |
| 新命令行 | `scripts/progressive_stage2.py`、`scripts/test_progressive_cpu.py` |
| 新配置 | `configs/progressive/legacy.yaml`、`full.yaml`、`certified.yaml`、`model.json` |
| 新测试 | `tests/test_progressive_core.py`、`test_progressive_head.py`、`test_progressive_data_training.py`、`test_progressive_prediction.py`、`test_progressive_controller.py`、`test_progressive_integration.py` |
| 文档 | `docs/progressive_lookahead.md`、`research.md`（仅追加本次入口记录，保留原修改） |

原 `residual_head.py`、`stage2_collect.py`、`stage2_data.py`、`stage2_training.py`、`stage2_online.py`、`inference_gate.py`、世界模型和第一阶段实现保持原代码；没有提交、推送或切换分支。
