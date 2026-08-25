# ETP-R1 Top-5 前瞻与 E24 联合 SFT 操作说明

## 固定实验

本实验只运行 R2R。决策链为：基座 Top-5 ghost、q0 NWM、DINO-CWP g1、
q1 NWM、E24 残差重排、最终导航点。q1 不查询模拟器、不渲染 RGB，E24
只修改 Top-5 的分数；最终动作仍在全部可执行 ghost 中比较。

- 基座：第二轮 SFT `iter14200`，SHA256
  `1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61`。
- E24：`head.offline.avg3.center17250.pth`，SHA256
  `bae7a9664000235dfc6fb66b43a8a0a38e7645b876e6a6369f732eb7bf404ed8`。
- DINO-CWP：SHA256
  `6a45291219907dd027203d224f3f8400631651a83bd01c45b1dea55d93ec0979`。
- NWM body、heads、stat 的 SHA256 固定在
  `run_r2r/iter_train_rae_dino_e24_joint.yaml`。
- 正式训练：双 A6000、每 rank 4 环境、全局 batch 8、10,000 次更新、
  每 200 次保存，共 50 个 checkpoint。
- DAgger 使用绝对时间轴偏移 14,200；E24 动作权重在本地前 400 次更新
  从 0 线性增至 1。

## 稳定资源入口

训练机项目目录为 `/home/gwl/project/etpr1/ETP-R1`。忽略目录
`pretrained/active_lookahead/` 中使用三个稳定软链接：

- `base_iter14200.pth`：指向 ETP-R1 保留的第二轮 SFT 最佳点。
- `e24_avg3.pth`：指向
  `/mnt/data2tb/ETPNav_data/active_lookahead/g59_top5_v2/offline/final/e24_champion/head.offline.avg3.center17250.pth`。
- `dino_cwp_best.pt`：指向训练机数据盘上的 DINO-CWP `best.pt`。

测评机只需 `dino_cwp_best.pt`、NWM 三项资产以及联合 checkpoint；E24
从联合 checkpoint 恢复，不依赖旧 avg3 文件。所有启动入口都会校验固定哈希。

## 验收顺序

1. 在实际训练环境运行单元测试和真实资产数值对齐。
2. 运行两次更新 smoke，确认模型和训练状态落盘。
3. 用新进程 `resume` 到第 3 次更新，核对 optimizer、scheduler、scaler、
   iteration 和 episode 队列。
4. 从原基座重新运行 400 次 pilot；不得从 smoke 接续。
5. 在测评机完成固定 16 episode pilot 评测。
6. pilot 通过后，再从原基座和原 E24 avg3 重新启动正式 10,000 次训练。

训练机管理入口：

```bash
bash scripts/manage_rae_r2r_e24_joint_server.sh smoke
bash scripts/manage_rae_r2r_e24_joint_server.sh pilot
bash scripts/manage_rae_r2r_e24_joint_server.sh start
bash scripts/manage_rae_r2r_e24_joint_server.sh resume
bash scripts/manage_rae_r2r_e24_joint_server.sh status
```

`smoke` 和 `pilot` 使用各自带时间戳的新目录，不覆盖正式实验。正式恢复必须
找到成对的模型与 training-state，并严格恢复三组 optimizer 和 E24 状态。

## 测评与最佳点

测评机先按项目规则检查 RTX 4090、受保护的 ETPNav 容器和计算进程，再启动：

```bash
bash scripts/manage_e24_joint_eval_watch_host.sh start
bash scripts/manage_e24_joint_eval_watch_host.sh status
```

watcher 按 iteration 升序评测，要求每个 checkpoint 同时存在完整指标 JSON 和
前瞻诊断 JSON。首次发现 checkpoint 时按实际文件大小核算剩余同步空间，并在
预计剩余总量外保留至少 40 GiB；空间不足即停止。50 点完成后自动生成：

- `checkpoint_summary.json`：路径、SHA256、SR、SPL、SR+SPL、覆盖率和耗时。
- `best_selection.json`：按 SR+SPL、SPL、SR、iteration 依次选择唯一最佳点。

最后在训练机运行：

```bash
bash scripts/finalize_e24_joint_selection_server.sh
```

该脚本通过训练机到测评机的直连读取选择结果，重新核验训练机 checkpoint
SHA256，并在训练机建立最佳模型硬链接。它不会复制大 checkpoint，也不会删除
任何既有产物。
