# AGENTS.md instructions for /home/gwl/project/etpr1/ETP-R1

请在和用户讨论、讲解代码、流程、算法、架构或技术方案时，尽量使用简单、自然、容易理解的中文。默认先用通俗说法解释，不要一上来就使用大量英文、缩写或专业名词。能用中文说明的地方，优先用中文说明。

如果确实需要使用英文词、缩写或专业术语，请先用中文说明它是什么、用来做什么、为什么这里需要它，再继续讲细节。不要连续抛出一串没有解释的英文术语。

讲解时尽量按“这是什么、要解决什么问题”“它怎么工作”“为什么这样设计、有什么利弊”的顺序展开。涉及较多背景时先讲核心，再分小步骤补充细节。

## 各机器职责

- 训练机工作区是 `/home/gwl/project/etpr1/ETP-R1`，测评机工作区是 `/home/a6000/gwl/ETP-R1`，笔记本工作区是 `/home/sia/project/ETP-R1`。
- 训练、测试和验证在哪台机器实际运行，就使用该机器上本工程对应的工作区和环境完成；训练机任务直接在训练机测试，测评机任务直接在测评机测试，不再要求所有测试统一转移到测评机。
- 不要仅凭任务名称自行选择执行机器；以用户当次要求和任务实际运行位置为准。
- 本机现有 `/home/gwl/miniconda3/envs/etpnav` 只作为旧 CLIP/ETP-R1 环境的历史参考，不把它的结果当作新方案验证结果。
- 本机 `/home/gwl/miniconda3/envs/raenwm` 属于受保护环境，不安装、卸载、升级或修改其中的 `.pth` 文件，避免影响正在使用该环境的 ETPNav 工程。

## 测评机身份约定

- 用户提到“4090”或“测评机”时，固定指主机 `eval-4090`。
- 从笔记本访问测评机统一使用 `ssh eval`；从训练机访问测评机使用 `ssh eval-4090`。
- 从笔记本访问训练机统一使用 `ssh server`。
- SSH 别名负责在局域网与 Tailscale 之间选择稳定入口，不硬编码或猜测动态 IP。
- 本文件中的 `ssh eval` 命令按笔记本入口书写；如果当前已经在训练机上，将入口替换为 `ssh eval-4090`。
- 测评机 SSH 用户、主机名都可能显示为 `a6000`，不能仅凭提示符判断机器身份。
- 每次连接后先运行 `hostname` 和 `whoami` 确认机器身份。
- 测评机工作根固定为 `/home/a6000/gwl`。

## 本工程的测评机容器

- 本工程专用 Docker 容器固定命名为 `gwl-etpr1-rae`。
- 容器基础镜像固定为 `gwl-etpnav:etpnav-runtime-20260701185256`，只复用镜像，不复用现有容器的进程空间。
- 容器绑定 `/home/a6000/gwl` 到相同容器路径，默认项目目录为 `/home/a6000/gwl/ETP-R1`。
- 本工程不得进入现有 `gwl-etpnav` 容器安装依赖、修改文件、生成数据或运行实验。该容器属于 ETPNav，并且可能有长期任务正在运行。
- 非交互检查优先使用：

```bash
ssh eval 'docker exec gwl-etpr1-rae bash -lc "<command>"'
```

- 需要交互终端时使用：

```bash
ssh eval 'docker exec -it gwl-etpr1-rae bash'
```

- 只有安装系统包等确实需要管理员权限时，才临时使用 `docker exec -u root`。

## 本工程的测评机 conda 环境

- 本工程专用环境固定为 `/home/a6000/gwl/miniconda3/envs/etpr1_rae`，环境名是 `etpr1_rae`。
- 该环境从测评机已有 `raenwm` 环境只读克隆，克隆后所有包变更都只发生在 `etpr1_rae`。
- 不得修改 `/home/a6000/gwl/miniconda3/envs/raenwm`，不得修改它的 `etpnav-local-deps.pth`。
- 新环境克隆后必须移除新环境中继承的 ETPNav 路径绑定，并改用 ETP-R1 自有依赖目录。
- 容器内激活方式：

```bash
source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh
conda activate etpr1_rae
cd /home/a6000/gwl/ETP-R1
```

- 测试和实验日志必须记录实际的 Python、PyTorch、Transformers、CUDA、Habitat 和 Habitat-Sim 版本。

## 受保护的测评机资源

- `/home/a6000/gwl/ETPNav`：ETPNav 工程，只允许读取或复制必要参考，不允许原地修改。
- `/home/a6000/gwl/RAE-NWM`：RAE-NWM 工程，只允许读取或复制模型配置、权重和统计文件，不允许原地修改。
- `/home/a6000/gwl/dino_cwp`：其他实验工程，除非用户明确授权，否则只读。
- `/home/a6000/gwl/miniconda3/envs/raenwm`：ETPNav 正在使用的环境，禁止修改。
- `gwl-etpnav`：ETPNav 容器，禁止停止、删除、重建或用于本项目实验。
- 默认不得终止不属于本项目的 `torchrun`、`run.py`、`train.py` 或 Docker 进程。只有用户针对当前任务中的具体进程明确说明可以停止时，才允许停止；该授权只对当次明确指定的进程有效，不能延伸到后续任务或其他进程。

## GPU 使用约定

- 测评机只有一张 RTX 4090 24GB。运行任何 GPU 命令前先检查：

```bash
ssh eval 'nvidia-smi; docker ps --format "{{.Names}}|{{.Status}}"'
```

- 同时检查 `gwl-etpnav` 中是否仍有 ETPNav 任务：

```bash
ssh eval 'docker exec gwl-etpnav bash -lc "ps -eo pid,ppid,stat,etime,cmd | grep -E '\''torchrun|run.py|train.py'\'' | grep -v grep || true"'
```

- 如果 ETPNav 正在占用 GPU，不并行启动本项目的全量特征生成、预训练、SFT、GRPO 或完整评测，也不得擅自结束 ETPNav；应等待资源释放或向用户说明现场情况。只有用户明确授权停止该具体任务时，才能按上一节的授权边界处理。
- 在测评机执行的轻量 CPU 检查也必须在新容器中进行，但不得因为“只是检查”而修改现有 ETPNav 环境。

## 代码同步与产物位置

- 本工程中央裸仓库固定为训练机的 `/home/gwl/git/ETP-R1.git`。
- 三个工作目录分别是：训练机 `/home/gwl/project/etpr1/ETP-R1`、笔记本 `/home/sia/project/ETP-R1`、测评机 `/home/a6000/gwl/ETP-R1`。
- 笔记本通过 `origin = server:/home/gwl/git/ETP-R1.git` 访问中央仓库；训练机和测评机也以该中央仓库为 `origin`。原 GitHub 仓库保留为 `upstream`，只用于获取上游历史。
- 常规同步先检查 `git status --short --branch`，提交后使用 `git push origin HEAD`。推送只更新中央裸仓库，不会自动更新其他工作目录。
- 更新目标工作目录前必须确认没有未提交修改，再执行 `git fetch origin --prune --tags` 和 `git pull --ff-only`。如果目录有改动、分支分叉或无法快进，立即停止并说明，不得强制覆盖。
- 同步全部本地分支和标签只在用户明确要求时使用 `git push origin --all` 和 `git push origin --tags`。
- 已由 Git 管理的源码只通过 Git 同步，不用 `scp`、`rsync` 或直接复制目录替代 Git。
- 中央仓库拒绝新提交中单个超过 10 MiB 的文件。模型、数据集、HDF5、checkpoint、日志、缓存和运行结果不得提交。
- 源码优先在笔记本工作区修改并提交，再同步到本次任务实际运行的目标工作目录。
- 同步时保留远端生成的数据、checkpoint 和日志，不使用带删除语义的命令覆盖这些目录。
- 全量 DINO HDF5、预训练权重、SFT/GRPO checkpoint、评测结果和日志保存在实际运行任务的机器上对应的 ETP-R1 项目目录或其明确的数据目录中。
- 除非用户明确要求，不把大体积 HDF5、checkpoint 或训练日志复制回本机。
- CLIP 和 RAE/DINOv2 使用独立配置、特征文件和输出目录，禁止互相覆盖。

## 测试与长任务约定

- 测试必须在实际运行训练或评测的目标机器及其对应工程环境中执行，测试结果要记录命令、退出码和关键输出。测评机任务使用 `gwl-etpr1-rae` 和 `etpr1_rae`；训练机任务使用训练实际采用的环境。
- 在长训练前依次完成：环境导入检查、RAE CLS 数值一致性、小规模 HDF5、预训练单 batch、SFT 单环境、GRPO 冻结检查、R2R/RxR 单 episode。
- 长任务使用最薄的启动方式；需要后台托管时优先使用目标机器已有的 `tmux` 或项目管理脚本，保持实际训练命令本体不变。
- 启动后必须确认父进程和子进程存活、日志持续增长、GPU 有计算负载、checkpoint 写入新实验目录。
- 恢复训练时必须同时核对实验名、配置、checkpoint 路径、iteration、优化器和调度器状态，避免接错实验。
