# Git 工作区盘点（2026-09-21）

## 最终状态

本轮累计归档15个工作区，三机各保留3个，共9个：`ETP-R1`、`ETP-R1-stage2-e24`、`ETP-R1-recursive-future-rollout`。递归前瞻由用户明确要求保留不动。

最后一批训练机`ETP-R1-etpnav59-r1`已归档，固定提交`dfcd23d1d7b9f119407c95bcf812f5586e1c0a67`保留在`refs/archive/workspaces/20260921/ETP-R1-etpnav59-r1`，亦被保留的world-model-migration分支包含。归档目录为`/home/gwl/project/etpr1/ETP-R1-workspace-archives/20260921/ETP-R1-etpnav59-r1/`，78项额外文件/链接及精确提交已核验；共享数据、权重、运行环境链接的目标没有移动。移除前未发现工作目录占用或入向工作区链接。

最终三机worktree清单已复核；训练机RxR父进程及两个训练进程、测评机stage2采集主进程仍存活。全部归档保留原分支、非Git内容及恢复清单，没有推送或改变中央仓库。恢复方法见各机`ETP-R1-workspace-archives/20260921/README.md`及各目录`manifest.json`；有未提交文档的migration归档还包含全文与补丁。

用户明确授权的后续整理原则：对明确已被主线完整包含或明确决定不采用的历史工作区，核对依赖、保存必要内容后自主归档，不再逐个询问；当时未决定是否采用、且现场证据也无法判定的，再询问。仍在使用的主目录、二阶段目录和明确要求保留的递归前瞻不适用自动归档。

以下保留各批次操作和归档前盘点，旧计数与“待确认”描述不代表当前状态。

## 已执行归档（用户授权后）

最新用户决定：三机`ETP-R1-recursive-future-rollout`明确保留，暂不改动；当前转入讨论训练机`ETP-R1-etpnav59-r1`，尚未获归档指令。该工作区固定于2026-08-18的`dfcd23d`，无未提交内容，现场未发现进程以其为工作目录；提交已被保留的`feature/world-model-migration`分支包含。

第四批：用户确认后，笔记本`ETP-R1-world-model-migration`已归档，准确提交`a0793426696f1e3d549442644e8240ae41b98a35`及原分支保留，并建立同名归档引用。归档目录为`/home/sia/project/ETP-R1-workspace-archives/20260921/ETP-R1-world-model-migration/`。32项额外文件已按设备、inode、大小核验；未提交的`research.md`完整保存于`modified/research.md`并核验SHA256，同时保存`uncommitted.patch`，在原提交上通过`git apply --check`。恢复代码后应用补丁即可恢复未提交修改。未发现工作目录占用或入向工作区链接；远端未做修改。

**第四批后最新计数：累计归档14个工作区，剩余10个（笔记本3、训练机4、测评机3）。** 尚待逐项讨论：递归前瞻、ETPNav59验证。以下批次计数均为操作历史。

第三批：用户确认后，三机`ETP-R1-grpo-stop-fix`归档完成。笔记本/训练机/测评机分别保留`b404900c`、`9032d310`、`3e6e445c`，各机`fix/grpo-stop-accounting`分支未改变，并建立`refs/archive/workspaces/20260921/ETP-R1-grpo-stop-fix`。各机归档根下同名目录包含清单及恢复命令，测评机归档根为`/home/a6000/gwl/ETP-R1-workspace-archives/20260921/`。

额外条目分别13、80、1502项；测评机嵌套依赖另含27592项，均核验保存。共享data链接目标未移动；测评机运行环境中指向旧工作区的绝对链接已调整到归档目录，原链接记录在清单的`link_original`或嵌套`contents[].link`中。归档前未发现工作目录占用或其他ETP-R1目录的入向链接；归档后训练机RxR父/子进程和测评机stage2采集主进程仍存活。未修改中央仓库。

**第三批后最新计数：累计归档13个工作区，剩余11个（笔记本4、训练机4、测评机3）。** 尚待逐项讨论：世界模型迁移、递归前瞻、ETPNav59验证。下方第一、二批计数保留为操作历史。

第二批：用户逐项确认后，笔记本和训练机的`ETP-R1-perf`也已归档，精确提交`89008184be4bc52384bceaea09fec18c0e3449c4`，原`perf/panorama-training`分支及7个未合入提交全部保留。新增归档引用为`refs/archive/workspaces/20260921/ETP-R1-perf`，归档位置为下面两机归档根中的`ETP-R1-perf/`。笔记本无额外文件；训练机保存1612项额外条目，其中一个嵌套依赖目录另核验27591项，独立运行环境约5.1GB亦保留。清单、文件元数据、链接、分支及归档引用核验通过；RxR训练主进程仍存活。

**最新计数：累计归档10个工作区，剩余14个（笔记本5、训练机5、测评机4）。** GRPO修复、世界模型迁移、递归前瞻和ETPNav59验证尚未归档；后文第一批计数及盘点表为历史快照。

已归档五类共8个工作区：笔记本与训练机的`persistent-ghost`、`rxr-perf`、`wm-context`，以及训练机`.worktrees/legacy452500-hfov60`、`.worktrees/rae-dinov2-encoder`。
下方24个工作区的表格保留为归档前快照；归档后剩余16个：笔记本6、训练机6、测评机4。

- 笔记本归档根：`/home/sia/project/ETP-R1-workspace-archives/20260921/`。
- 训练机归档根：`/home/gwl/project/etpr1/ETP-R1-workspace-archives/20260921/`。
- 原分支全部保留；各原仓库新增`refs/archive/workspaces/20260921/<目录名>`保留准确提交。没有推送或更改中央仓库。
- 各归档目录的`manifest.json`记录原路径、分支、提交、额外文件、原始链接和恢复代码命令，`extras/`保存全部未跟踪及忽略文件。源码可由Git恢复，没有额外复制源码树。
- 笔记本保存16项额外文件；训练机保存7618项额外文件/链接。文件使用同文件系统重命名，归档后逐项核对设备、inode与大小；链接保留原始记录并调整归档目标。训练机约8.79GB实际文件仍留在训练机，共享链接指向的数据未移动。
- 移除前确认选定提交是主线祖先、没有已跟踪文件改动、没有进程以这些目录为工作目录、未发现其他ETP-R1工作区链接依赖它们；保存额外内容后使用普通`git worktree remove`，没有使用force。
- 验证8份归档清单、全部额外文件和归档引用通过；训练机原RxR torchrun及两个训练进程仍存活。测评机未做修改。

剩余历史线按用户要求逐个讨论：`perf`、`grpo-stop-fix`、`world-model-migration`、`recursive-future-rollout`、`etpnav59-r1`；尚未对这些目录执行归档。

现场只读核查三机 `git worktree list --porcelain`、每个工作区状态、提交祖先关系、中央仓库分支及远端进程。没有 fetch/pull、切分支、删除目录或停止任务。以下“闲置”表示未发现当前实验在该目录运行，不代表用户已决定永久弃用。

## 当前布局

中央裸仓库：训练机 `/home/gwl/git/ETP-R1.git`，共有13个分支。
实际集成主线为 `feature/e24-joint-sft`，中央提交 `01424a2`；`main` 停在2026-08-17的 `b2fbbcc`。

下表“笔/训/测”的父目录分别是 `/home/sia/project`、`/home/gwl/project/etpr1`、`/home/a6000/gwl`。

| 工作区目录 | 机器 | 当前判断 | 依据与保留事项 |
|---|---|---|---|
| ETP-R1 | 笔/训/测 | 保留，正式主目录 | 笔记本有论文、图件及文档改动；训练机正在运行RxR；测评机存放资源并有TensorBoard及日志整理服务 |
| ETP-R1-stage2-e24 | 笔/训/测 | 保留，当前二阶段开发线 | 测评机正在运行stage2_collect；笔记本两份文档有未提交修改；训练机保存前期实验 |
| ETP-R1-persistent-ghost | 笔/训 | 已完成，优先归档候选 | 两端干净，提交均已包含在当前主线 |
| ETP-R1-rxr-perf | 笔/训 | 已完成，优先归档候选 | 两端干净，e5cb7bb已包含在主线；正式RxR已从主目录运行 |
| ETP-R1-wm-context | 笔/训 | 已完成，先保存产物再归档 | 2b5f777已包含在主线；笔记本有未跟踪的docs/diagnostics/nwm-context-quality-20260909/ |
| ETP-R1-perf | 笔/训 | 历史实验，保留独立提交 | 主线只合入选定版本；分支末端还有7个提交，包括静态缓存与SDPA试验 |
| ETP-R1-grpo-stop-fix | 笔/训/测 | 历史实验，待归档 | 均干净；三机分别b404900、9032d31、3e6e445；分支末端相对主线有27个独立提交 |
| ETP-R1-world-model-migration | 笔 | 历史实验，待归档 | a079342相对主线有31个独立提交；research.md有未提交修改；训练/测评机仍有同名分支但无对应工作区 |
| ETP-R1-recursive-future-rollout | 笔/训/测 | 历史实现参考，建议暂留 | 9a97727有12个主线未包含的提交；当前论文图仍引用其递归实现；远端为固定提交检出 |
| ETP-R1-etpnav59-r1 | 训 | 历史验证，待归档 | 固定提交dfcd23d，干净；相对主线22个独立提交，应先确认持久引用 |
| ETP-R1/.worktrees/legacy452500-hfov60 | 训 | 早期历史工作区，优先归档候选 | 固定提交7431266，干净且主线已包含 |
| ETP-R1/.worktrees/rae-dinov2-encoder | 训 | 早期历史工作区，优先归档候选 | 9cb9577，干净且主线已包含 |

共24个已注册工作区：笔记本9、训练机11、测评机4。表中独立提交数按Git祖先关系计算，不等于这些功能都没有通过其他提交进入主线。

## 同步与未提交状态

- 主线：笔记本和训练机为01424a2；测评机为839fa97，相对中央落后38个提交。
- 二阶段：笔记本和测评机为62577b2；训练机为f750ecf，相对中央实际落后22个提交。训练机本地origin缓存显示落后17，说明缓存不是最新远端状态。
- 笔记本主目录有147条简略状态记录（目录条目可能包含多个文件），包括`.gitignore`、旧计划删除、未跟踪新计划、research.md、论文和大量图件。
- 笔记本二阶段工作区修改了`docs/stage2-9200-future-ablation-20260921.md`与`research.md`。
- 其余未提交内容为上表wm-context的诊断目录、world-model-migration的research.md。远端所有已注册工作区Git状态干净。
- Git状态干净不表示目录没有数据：远端多处存在被忽略的`.runtime`、`data`、`pretrained`及符号链接。本次未逐项核算体积或数据归属，不能据此直接删除。

## 其他目录与建议

测评机还有`etpr1-prefetch-test.O6nOAt`和`ETP-R1-recursive-future-rollout-validation`，不在上述worktree清单中，属于另待检查的临时/验证目录。笔记本`PILOT`拥有独立.git，不属于ETP-R1的关联工作区，本次不将其列作可清理对象。

`ETP-R1-prediction-adapter`已按9月16日既有记录删除，当前三机worktree清单中均不存在；实验产物另存，不计入现存工作区。

建议先保留三机主目录和二阶段目录；优先整理已合入主线的五类旧工作区；其他历史分支先保存未提交文档、给固定提交建立明确归档引用，并核对忽略文件与产物链接，再决定是否移除工作区。删除工作区与删除分支是两件事，可以移除旧源码副本而保留分支历史。
