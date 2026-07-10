# ETP-R1 与原版 ETPNav 差异分析

## 1. 对比基准

本文档对比的是当前工程 `ETP-R1` 与本地 ETPNav 工程早期提交中的原版 ETPNav 代码。

- ETP-R1 路径：`/home/gwl/project/etpr1/ETP-R1`
- 原版 ETPNav 本地仓库：`/home/gwl/project/ETPNav/ETPNav`
- 原版对比提交：`85bed52 chore: reset repository to clean baseline with step log viewer`
- 原版临时 worktree：`/tmp/etpnav-baseline-85bed52`

这里的“原版 ETPNav”指早期 ETPNav 代码结构：`SS-ETP` 训练器、`PolicyViewSelectionETP` 策略、以 R2R 为主的预训练配置。当前 `/home/gwl/project/ETPNav/ETPNav` 的后续分支已经混入 DINO、NWM 等实验，因此本文没有把当前 HEAD 当作原版。

## 2. 总体结论

ETP-R1 没有推翻 ETPNav 的核心导航骨架。两者都保留了同一条主线：

1. 当前视角输入 RGB/depth。
2. waypoint predictor 预测附近可走候选点。
3. `GraphMap` 把已访问节点和候选 ghost 节点组织成拓扑图。
4. 多模态 planner 根据语言指令和拓扑图选择下一步图节点。
5. 低层控制器把图上选择转成连续环境动作。

真正的主要差异在这几层：

1. 训练范式：从原版单一 `train`，扩展成 `dagger` 监督微调 + `grpo` 强化微调。
2. 预训练数据：从 R2R 为主，扩展成 R2R + RxR + Gemini 增强数据联合预训练。
3. 语言底座：从 BERT/LXMERT 风格配置，改成 XLM-RoBERTa 风格配置。
4. 任务感知输入：新增文本任务类型编码和图节点任务类型编码，用来统一 R2R/RxR。
5. 图上决策头：从直接对图节点打分，改成“图节点重新查询文本后再融合打分”。
6. 工程训练细节：新增 warmup 学习率调度、optimizer 分组、checkpoint 轻量保存和 GRPO 相关配置。

可以把 ETP-R1 理解成：保留 ETPNav 的拓扑规划器，但用更大规模预训练、更强语言底座、更统一的 R2R/RxR 建模，以及在线 GRPO 强化微调来提升它。

## 3. 入口和训练阶段差异

### 原版 ETPNav

原版入口只区分：

- `train`
- `eval`
- `inference`

代码位置：

- `/tmp/etpnav-baseline-85bed52/run.py:30`
- `/tmp/etpnav-baseline-85bed52/run_r2r/main.bash:4`

原版 R2R 脚本里，训练阶段直接使用 `--run-type train`，trainer 名称来自配置里的 `SS-ETP`。

### ETP-R1

ETP-R1 把训练拆成：

- `dagger`：在线监督微调，也可以理解成原来 imitation learning 路线的延续。
- `grpo`：在线强化微调，是 ETP-R1 新增的关键阶段。
- `eval`
- `inference`

代码位置：

- `run.py:25`
- `run.py:97`
- `run_r2r/main_server.bash:20`
- `run_r2r/main_server.bash:41`
- `run_rxr/main_server.bash:13`
- `run_rxr/main_server.bash:34`

R2R 脚本中可以清楚看到两阶段训练：

- DAgger/SFT 从联合预训练权重开始：`run_r2r/main_server.bash:20`
- GRPO 从 DAgger checkpoint 继续：`run_r2r/main_server.bash:41`

这一点是和原版最大的流程差异。原版主要是“预训练后监督微调”，ETP-R1 变成“联合预训练后监督微调，再强化微调”。

## 4. 新增 GRPO 强化微调

GRPO 可以先用通俗方式理解：同一个环境起点，模型自己采样多条路线。走得好的路线相对加分，走得差的路线相对降权。这样模型不只是模仿专家轨迹，还会根据实际导航结果调整策略。

### 新增文件

ETP-R1 新增：

- `vlnce_baselines/GRPO_trainer_ETP_R1.py`

原版 ETPNav 没有对应文件。

### 关键代码

GRPO trainer 注册：

- `vlnce_baselines/GRPO_trainer_ETP_R1.py:65`

GRPO 参数初始化：

- `vlnce_baselines/GRPO_trainer_ETP_R1.py:72`
- `vlnce_baselines/GRPO_trainer_ETP_R1.py:80`
- `vlnce_baselines/GRPO_trainer_ETP_R1.py:83`

采样多条轨迹：

- `vlnce_baselines/GRPO_trainer_ETP_R1.py:637`
- `vlnce_baselines/GRPO_trainer_ETP_R1.py:867`

按同组轨迹 reward 计算相对优势：

- `vlnce_baselines/GRPO_trainer_ETP_R1.py:656`
- `vlnce_baselines/GRPO_trainer_ETP_R1.py:666`

使用概率比值、裁剪项和 KL 约束更新策略：

- `vlnce_baselines/GRPO_trainer_ETP_R1.py:762`
- `vlnce_baselines/GRPO_trainer_ETP_R1.py:764`
- `vlnce_baselines/GRPO_trainer_ETP_R1.py:771`
- `vlnce_baselines/GRPO_trainer_ETP_R1.py:786`

### 对模型行为的影响

原版监督训练主要学习“专家在这里会选哪个图节点”。ETP-R1 的 GRPO 阶段会让模型实际闭环运行，并根据最终导航质量调整策略。这更接近真实评测时的状态分布，因为模型会遇到自己走错后产生的新状态。

代价是训练更复杂：需要多次采样、保存采样时概率、计算 reward、控制策略不要偏离太远。

## 5. 监督训练器差异

### 原版

原版监督 trainer：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/ss_trainer_ETP.py:66`

训练循环：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/ss_trainer_ETP.py:774`
- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/ss_trainer_ETP.py:809`

原版训练循环比较直接：rollout、算 loss、反向传播、保存 checkpoint。

### ETP-R1

ETP-R1 监督 trainer 改名并注册为：

- `vlnce_baselines/ss_trainer_ETP_R1.py:62`

主要差异包括：

1. checkpoint 可轻量保存，只保存 `state_dict/config/iteration`，减少中间 checkpoint 体积。
   - `vlnce_baselines/ss_trainer_ETP_R1.py:76`

2. optimizer 分组，给不同参数设置 weight decay。
   - `vlnce_baselines/ss_trainer_ETP_R1.py:229`

3. 新增 warmup 和最小学习率比例。
   - `vlnce_baselines/ss_trainer_ETP_R1.py:240`
   - `run_r2r/iter_train.yaml:55`
   - `run_rxr/iter_train.yaml:50`

4. rollout 时统一生成任务类型，用于区分 R2R/RxR。
   - `vlnce_baselines/ss_trainer_ETP_R1.py:874`
   - `vlnce_baselines/ss_trainer_ETP_R1.py:876`
   - `vlnce_baselines/ss_trainer_ETP_R1.py:882`

5. 构建图输入时新增 `gmap_task_embeddings`。
   - `vlnce_baselines/ss_trainer_ETP_R1.py:416`
   - `vlnce_baselines/ss_trainer_ETP_R1.py:465`
   - `vlnce_baselines/ss_trainer_ETP_R1.py:487`

原版 `_nav_gmap_variable` 没有 `task_type` 参数，也没有 `gmap_task_embeddings`：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/ss_trainer_ETP.py:671`
- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/ss_trainer_ETP.py:739`

## 6. 策略网络接口差异

### 原版策略

原版策略类：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/Policy_ViewSelection_ETP.py:35`

语言输入只有：

- `txt_ids`
- `txt_masks`

位置：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/Policy_ViewSelection_ETP.py:157`
- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/Policy_ViewSelection_ETP.py:166`

导航输入里也没有图任务编码：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/Policy_ViewSelection_ETP.py:351`

### ETP-R1 策略

ETP-R1 策略类改为：

- `vlnce_baselines/models/R1Policy.py:35`

新增输入：

- `txt_task_encoding`
- `gmap_task_embeddings`
- `dropout_rate`

位置：

- `vlnce_baselines/models/R1Policy.py:41`
- `vlnce_baselines/models/R1Policy.py:160`
- `vlnce_baselines/models/R1Policy.py:169`
- `vlnce_baselines/models/R1Policy.py:355`

这说明 ETP-R1 的模型不是只看“这句话是什么、这张图是什么”，还会显式知道“当前样本属于哪个任务类型”。这对 R2R/RxR 联合训练很重要，因为 R2R 和 RxR 的语言长度、语言来源、评测习惯都不完全一样。

## 7. 核心多模态模型差异

### 7.1 语言 embedding 加入任务类型

原版 `BertEmbeddings` 只包含：

- word embedding
- position embedding
- token type embedding

位置：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/etp/vilmodel_cmt.py:51`
- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/etp/vilmodel_cmt.py:70`

ETP-R1 新增：

- `task_type_encoding`
- 训练时对 task embedding 做 dropout
- 最终 embedding 里加上 task embedding

位置：

- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:51`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:56`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:80`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:85`

这让同一个语言模型可以在输入层就知道任务来源。

### 7.2 图节点 embedding 加入任务类型

ETP-R1 在全局图编码器里新增：

- `gmap_task_embeddings`

位置：

- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:575`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:623`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:629`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:636`

原版图输入只包括图节点、步数、位置和距离关系，没有任务 embedding：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/etp/vilmodel_cmt.py:635`

### 7.3 图上决策头改为图文再融合

原版图上动作预测基本是直接对图节点特征打分：

- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/etp/vilmodel_cmt.py:671`
- `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/etp/vilmodel_cmt.py:742`

ETP-R1 新增了图节点查询文本的过程：

- `graph_query_text`
- `graph_attentioned_txt_embeds_transform`
- 将 `gmap_embeds` 和文本注意力后的特征拼接成 `fusion_input`
- 再用新的 `global_sap_head` 打分

位置：

- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:716`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:717`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:804`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:806`
- `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py:807`

通俗理解：原版更像“每个图节点自己和语言融合后给分”；ETP-R1 在最终打分前，又让每个图节点重新去语言里找和自己相关的信息，然后再打分。这样有利于更细地对齐长指令和图节点。

## 8. 预训练差异

### 原版预训练

原版预训练配置：

- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_pretrain_habitat.json`

特点：

- `max_txt_len: 100`
- `num_train_steps: 100000`
- `init_pretrained: lxmert`
- 训练数据主要是 R2R train 和 R2R Prevalent augmented train
- 验证集是 R2R val seen/unseen

代码位置：

- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_pretrain_habitat.json:6`
- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_pretrain_habitat.json:14`
- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_pretrain_habitat.json:28`
- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_pretrain_habitat.json:33`

### ETP-R1 预训练

ETP-R1 预训练配置：

- `pretrain_src/run_pt/mix_pretrain_server.json`

特点：

- `max_txt_len: 250`
- `num_train_steps: 500000`
- `init_pretrained: roberta`
- 混合 R2R、RxR、Prevalent、Gemini 增强数据
- 同时验证 R2R unseen 和 RxR unseen

代码位置：

- `pretrain_src/run_pt/mix_pretrain_server.json:6`
- `pretrain_src/run_pt/mix_pretrain_server.json:14`
- `pretrain_src/run_pt/mix_pretrain_server.json:28`
- `pretrain_src/run_pt/mix_pretrain_server.json:33`
- `pretrain_src/run_pt/mix_pretrain_server.json:38`

这里是论文中“更大规模、更高质量、Gemini 标注、R2R/RxR 联合预训练”的代码落点。

### 预训练启动脚本

原版：

- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/run_r2r.bash`

ETP-R1：

- `pretrain_src/run_pt/run_mix_server.bash`

ETP-R1 的输出目录也从原版 R2R 预训练变成联合预训练：

- `pretrained/r2r_rxr_ce/mlm.sap_habitat_depth`

## 9. 语言底座差异

原版模型配置：

- `num_l_layers: 9`
- `layer_norm_eps: 1e-12`
- `max_position_embeddings: 512`
- `vocab_size: 30522`
- `lang_bert_name: bert-base-uncased`

位置：

- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_model_config_dep.json:17`
- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_model_config_dep.json:20`
- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_model_config_dep.json:32`
- `/tmp/etpnav-baseline-85bed52/pretrain_src/run_pt/r2r_model_config_dep.json:36`

ETP-R1 模型配置：

- `num_l_layers: 12`
- `layer_norm_eps: 1e-5`
- `max_position_embeddings: 514`
- `vocab_size: 250002`
- `lang_bert_name: ./bert_config/xlm-roberta-base`
- `max_txt_task_embeddings: 4`
- `max_gmap_task_embeddings: 3`

位置：

- `pretrain_src/run_pt/mix_model_config_dep.json:17`
- `pretrain_src/run_pt/mix_model_config_dep.json:20`
- `pretrain_src/run_pt/mix_model_config_dep.json:32`
- `pretrain_src/run_pt/mix_model_config_dep.json:36`
- `pretrain_src/run_pt/mix_model_config_dep.json:37`

这说明 ETP-R1 的语言侧明显向多语言和长文本场景靠拢，尤其是为了 RxR。

## 10. 配置和路径组织差异

原版默认输出路径更多使用 `ETPNav_data`：

- `/tmp/etpnav-baseline-85bed52/run.py:83`
- `/tmp/etpnav-baseline-85bed52/run_r2r/iter_train.yaml:9`

ETP-R1 输出路径改成项目内 `data/logs/...`：

- `run.py:77`
- `run.py:78`
- `run_r2r/iter_train.yaml:10`
- `run_rxr/iter_train.yaml:9`

原版 R2R 配置里 trainer/policy 是：

- `TRAINER_NAME: SS-ETP`
- `policy_name: PolicyViewSelectionETP`

ETP-R1 R2R/RxR 配置里变成：

- `TRAINER_NAME: SS-ETP-R1`
- `ENV_NAME: R1Env`
- `policy_name: R1Policy`
- 新增完整 `GRPO` 配置块

位置：

- `run_r2r/iter_train.yaml:6`
- `run_r2r/iter_train.yaml:7`
- `run_r2r/iter_train.yaml:72`
- `run_r2r/iter_train.yaml:106`
- `run_rxr/iter_train.yaml:5`
- `run_rxr/iter_train.yaml:62`
- `run_rxr/iter_train.yaml:102`

## 11. 保留不变的核心骨架

虽然 ETP-R1 改了很多训练和模型细节，但下面这些 ETPNav 的核心思想仍然保留：

1. `waypoint_pred/TRM_net.py` 仍是候选路点预测器。
2. `GraphMap` 仍负责维护拓扑图、真实节点、ghost 节点和图距离。
3. policy 仍分为 `language`、`waypoint`、`panorama`、`navigation` 几种模式。
4. rollout 仍然是“编码语言 -> 预测候选点 -> 更新图 -> 图上选点 -> 环境执行”。

ETP-R1 的主要创新不是把 ETPNav 换成另一个导航器，而是把原有拓扑规划器放进更强的数据和训练体系里。

## 12. 建议阅读路线

如果你主要想看差异，不建议从头读完整工程。推荐这样读：

1. 先看入口差异：
   - `run.py`
   - `run_r2r/main_server.bash`
   - 对照 `/tmp/etpnav-baseline-85bed52/run.py`
   - 对照 `/tmp/etpnav-baseline-85bed52/run_r2r/main.bash`

2. 再看新增强化训练：
   - `vlnce_baselines/GRPO_trainer_ETP_R1.py`

3. 再看监督训练器怎么给模型喂任务类型：
   - `vlnce_baselines/ss_trainer_ETP_R1.py:416`
   - `vlnce_baselines/ss_trainer_ETP_R1.py:863`

4. 再看 policy 接口变化：
   - `vlnce_baselines/models/R1Policy.py`
   - 对照 `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/Policy_ViewSelection_ETP.py`

5. 最后看核心模型变化：
   - `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py`
   - 对照 `/tmp/etpnav-baseline-85bed52/vlnce_baselines/models/etp/vilmodel_cmt.py`

6. 如果想理解论文里“大规模预训练”的落点，再看：
   - `pretrain_src/run_pt/mix_pretrain_server.json`
   - `pretrain_src/run_pt/mix_model_config_dep.json`
   - `pretrain_src/pretrain_src/train_r2r.py`

## 13. 一句话总结

原版 ETPNav 重点是提出“在线拓扑建图 + 图上规划 + 连续控制”的导航框架。ETP-R1 的主要工作是在这个框架上叠加了更强的联合预训练、更适合 R2R/RxR 的任务感知模型，以及在线 GRPO 强化微调，让原来的拓扑规划器具备更强的数据利用和闭环优化能力。
