# 世界模型朝向合并复查（2026-09-09）

复查版本：笔记本和训练机均为 `84e6081`，分支 `feature/e24-joint-sft`。训练机工作区干净；笔记本已有 `.gitignore` 修改保持原样。本次没有修改模型代码、启动长训练或完整评测。

## 结论

默认 `world_exact_select` 已在真实导航预测中生效。检查的目标朝向、参考端点切换、相对条件重算及历史 reset/pause 链路没有发现阻断问题。短检查不能证明完整导航成功率或路径效率提升。

发现一个旧实验入口兼容问题：`scripts/ghost_concat_job.py` 新增朝向参数并默认 `world_exact_select`，但 `scripts/manage_ghost_concat_joint.py` 仍绑定旧 `ghost_concat_joint_bs8_20260908` 目录，未固定 `front`。真实旧 manifest 没有该参数，新生成命令含新模式；`scripts/rgb_only_optimization.py` 按命令完整比较，因此旧的未完成任务经管理器重启/恢复会被拒绝，提示 `Existing manifest has different command; choose a new --output`。已经完成的管理任务会提前返回，不受影响。保护能避免覆盖，但旧任务不能直接续跑。修复需要同时固定旧模式并兼容旧 manifest 中省略的默认参数；新全景实验应使用新输出目录。

## 本轮测试

实际机器：训练机 `gwl-sever`，用户 `gwl`，目录 `/home/gwl/project/etpr1/ETP-R1`。使用现有 `etpnav_unified` 与项目运行脚本，没有修改环境。

- Python 3.10.14，PyTorch 2.2.2+cu121，CUDA 构建 12.1，Transformers 4.49.0，Habitat/Habitat-Sim 0.3.3。
- CPU 回归：559 passed、2 skipped、2 warnings，30.03 秒，退出码 0。按已有训练机验证范围排除 integration 和 test_runtime_behavior.py。
- GPU 真实导航：双环境，4 次预测调用，其中 3 次有有效历史，19 个查询、10 个倒序参考端点；每个目标位置保持不变，目标绝对朝向符合当前位置指向目标的方向，相对转角为 0。退出码 0。
- 首次调用的 9 个查询因不足四帧跳过，是预期行为。检查在获得 3 次有效调用后主动结束，没有跑完两个 episode。
- 旧实验命令只做 dry-run 和 manifest 比较，未启动或恢复旧实验。

在训练机项目目录执行：

```bash
CUDA_VISIBLE_DEVICES= bash scripts/rgb_only_optimization_runtime.sh server -m pytest -q tests --ignore=tests/integration --ignore=tests/test_runtime_behavior.py
CUDA_VISIBLE_DEVICES=0 bash scripts/rgb_only_optimization_runtime.sh server scripts/verify_panorama_default.py --checkpoint data/logs/ghost_concat_joint_bs8_20260908/train/ghost_concat_v1_joint_train/checkpoints/ghost_concat_v1_joint_train/ckpt.iter1800.pth --output data/logs/panorama_review_20260909/real_navigation --environments 2
```

本轮训练机产物目录：`data/logs/panorama_review_20260909/`，包含 `tests.log`、`real_navigation.log`、`real_navigation/report.json`、`versions.json` 和 `legacy_command_review.json`。
