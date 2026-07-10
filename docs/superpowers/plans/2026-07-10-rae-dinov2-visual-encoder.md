# RAE/DINOv2 视觉编码器实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留原 CLIP 分支的同时，为 ETP-R1 增加始终冻结的 RAE/DINOv2-B 编码器，用三层 MLP 将归一化后的 768 维 CLS 投影到 512 维，并接通离线联合预训练、在线 SFT、GRPO、特征生成和 checkpoint 流程。

**Architecture:** 测评机使用独立容器 `gwl-etpr1-rae`、独立 conda 环境 `etpr1_rae` 和 ETP-R1 自有 Habitat 运行目录。离线预训练读取 `[36,768]` 的 DINO CLS HDF5，在线导航实时计算同语义 CLS；两条链路在各自 `ImageEmbeddings` 中使用同名 `rgb_projection`，后续网络只接收 512 维特征。DINO 主干永远冻结且不写入在线 checkpoint，MLP 在预训练和 SFT 中训练、在 GRPO 中冻结。

**Tech Stack:** Python 3.11、PyTorch 2.2.2+cu121、Transformers 4.49.0、Habitat-Lab/Habitat-Sim 0.3.3、HDF5、pytest、Docker、conda。

---

## 执行边界

- 本机 `/home/gwl/project/etpr1/ETP-R1` 只编辑代码、文档和 Git。
- 所有依赖安装、测试、特征生成、训练和评测都在测评机执行。
- 每次测评机操作前检查 `hostname`、`eno1=10.10.10.2`、容器和 GPU。
- 默认禁止停止 ETPNav；只有用户针对具体进程明确授权时才能停止。本计划开始时，先前获授权停止的评测任务已经退出，`gwl-etpnav` 容器仍在运行。
- 实施开始前使用 `using-git-worktrees` 建立隔离工作区；每个任务完成后按计划提交。

统一的测评机命令形式为：

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh python scripts/inspect_etpr1_runtime.py"'
```

每次运行测评机测试前，必须先从当前本机工作树做一次非删除式代码同步；大数据目录由后面的专门步骤单独同步：

```bash
rsync -a --exclude='.git/' --exclude='.runtime/' --exclude='data/' --exclude='pretrained/' --exclude='pretrain_src/datasets/' --exclude='pretrain_src/img_features/' ./ 4090:/home/a6000/gwl/ETP-R1/
```

## 文件结构

- 新增 `scripts/build_etpr1_habitat.sh`：复制并本地化 ETPNav 已验证的 Habitat 0.3.3 构建逻辑。
- 新增 `scripts/etpr1_rae_runtime_exec.sh`：只为当前子进程设置 ETP-R1 自有运行路径。
- 新增 `scripts/inspect_etpr1_runtime.py`：打印 Python、PyTorch、Transformers、CUDA 和 Habitat 实际来源。
- 新增 `vendor/legacy_clip/clip/`：把已验证的旧 CLIP Python 包复制为 ETP-R1 自有代码，保证原 CLIP 分支不依赖 ETPNav 工程路径。
- 新增 `model_components/rgb_projection.py`：定义 CLIP 恒等投影和 RAE 三层 MLP，避免预训练包依赖 Habitat。
- 新增 `vlnce_baselines/models/encoders/rae_dinov2_encoder.py`：加载本地 DINO、关闭最终仿射层、应用 RAE latent 统计并输出 768 维 CLS。
- 新增 `vlnce_baselines/models/checkpoint_utils.py`：过滤冻结 DINO、保存/校验 RGB 编码器元数据。
- 修改 `pretrain_src/pretrain_src/data/dataset.py`：按 `raw_image_feat_size` 读取 768 维 HDF5。
- 修改两套 `ImageEmbeddings`：统一使用 `rgb_projection` 参数名。
- 修改 `R1Policy.py` 和在线模型初始化：按配置选择 CLIP 或 RAE，并只向后续网络提供 512 维特征。
- 新增 `precompute_img_features/extract_rae_dinov2_features.py` 和 `validate_rae_dinov2_features.py`：直接渲染、编码、断点续写和完整校验。
- 新增 DINO 专用预训练、R2R、RxR 配置，不覆盖原 CLIP 配置。
- 新增 `tests/` 下的单元和集成测试；测试只在测评机运行。

---

### Task 1：建立测评机独立运行底座

**Files:**
- Create: `scripts/build_etpr1_habitat.sh`
- Create: `scripts/etpr1_rae_runtime_exec.sh`
- Create: `scripts/inspect_etpr1_runtime.py`
- Create: `tests/test_runtime_contract.py`

- [ ] **Step 1：添加运行约定测试**

创建 `tests/test_runtime_contract.py`，固定环境名、运行目录和禁止引用 ETPNav 的规则：

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_wrapper_uses_etpr1_owned_paths():
    text = (ROOT / "scripts/etpr1_rae_runtime_exec.sh").read_text()
    assert ".runtime/etpr1_habitat" in text
    assert "vendor/legacy_clip" in text
    assert "ETPR1_RUNTIME_ACTIVE=1" in text
    assert "/ETPNav" not in text


def test_habitat_builder_uses_etpr1_runtime_root():
    text = (ROOT / "scripts/build_etpr1_habitat.sh").read_text()
    assert ".runtime/etpr1_habitat" in text
    assert "ETPR1_HABITAT_LAB_SOURCE" in text
    assert "ETPR1_HABITAT_SIM_SOURCE" in text
```

- [ ] **Step 2：从已验证脚本生成 ETP-R1 自有脚本**

以只读方式参考：

```text
/home/gwl/project/ETPNav/ETPNav/scripts/build_sidecar_habitat.sh
/home/gwl/project/ETPNav/ETPNav/scripts/etp_runtime_exec.sh
/home/gwl/project/ETPNav/ETPNav/scripts/inspect_etp_runtime.py
```

用 `apply_patch` 把内容加入本工程，执行以下完整替换，并保留来源复制、来源标记、Python 3.11 补丁、Habitat-Sim 构建和 manifest 校验逻辑：

```text
build_sidecar_habitat.sh -> build_etpr1_habitat.sh
ETPNAV_RUNTIME_ROOT -> ETPR1_RUNTIME_ROOT
.runtime/etp_habitat_legacy -> .runtime/etpr1_habitat
ETPNAV_HABITAT_LAB_SOURCE -> ETPR1_HABITAT_LAB_SOURCE
ETPNAV_HABITAT_SIM_SOURCE -> ETPR1_HABITAT_SIM_SOURCE
etp_runtime_exec.sh -> etpr1_rae_runtime_exec.sh
ETPNAV_SIDECAR_* -> ETPR1_RUNTIME_*
ETPNAV_SIDECAR_ACTIVE -> ETPR1_RUNTIME_ACTIVE
```

把 `/home/gwl/project/ETPNav/ETPNav/vendor/legacy_clip/clip` 的源文件复制到本工程 `vendor/legacy_clip/clip`，不复制 `__pycache__`。运行脚本只把本工程的 `vendor/legacy_clip` 加入 `PYTHONPATH`，不得加入 ETPNav、`dino_cwp` 或共享 `_deps`；共享 `_deps` 只允许作为构建时只读来源，实际运行必须指向 `.runtime/etpr1_habitat/prefix`。

- [ ] **Step 3：同步代码基线并创建专用容器**

在本机执行非删除式同步，然后在测评机创建容器：

```bash
ssh 4090 'mkdir -p /home/a6000/gwl/ETP-R1/.runtime'
rsync -a --exclude='.git/' --exclude='.runtime/' --exclude='data/logs/' --exclude='pretrain_src/img_features/*.hdf5' /home/gwl/project/etpr1/ETP-R1/ 4090:/home/a6000/gwl/ETP-R1/
ssh 4090 'docker run -d --name gwl-etpr1-rae --gpus all --ipc=host --shm-size=16g --user 1000:1000 -v /home/a6000/gwl:/home/a6000/gwl -w /home/a6000/gwl/ETP-R1 gwl-etpnav:etpnav-runtime-20260701185256 bash -lc "while true; do sleep 3600; done"'
```

Expected：`docker ps` 显示 `gwl-etpr1-rae|Up ...`，原 `gwl-etpnav` 也保持运行。

- [ ] **Step 4：只读克隆 conda 环境并清除继承绑定**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda list -p /home/a6000/gwl/miniconda3/envs/raenwm --explicit > /home/a6000/gwl/ETP-R1/.runtime/raenwm-conda-explicit.txt && conda run -p /home/a6000/gwl/miniconda3/envs/raenwm pip freeze > /home/a6000/gwl/ETP-R1/.runtime/raenwm-pip-freeze.txt && conda create -y -p /home/a6000/gwl/miniconda3/envs/etpr1_rae --clone /home/a6000/gwl/miniconda3/envs/raenwm"'
ssh 4090 'find /home/a6000/gwl/miniconda3/envs/etpr1_rae -name "*.pth" -type f -print -exec sha256sum {} \; > /home/a6000/gwl/ETP-R1/.runtime/etpr1-cloned-pth-before.txt'
ssh 4090 'rm -f /home/a6000/gwl/miniconda3/envs/etpr1_rae/lib/python3.11/site-packages/etpnav-local-deps.pth'
```

Expected：原环境的 `etpnav-local-deps.pth` 仍存在；新环境中该文件不存在。

- [ ] **Step 5：构建 ETP-R1 自有 Habitat 目录**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && ETPR1_HABITAT_LAB_SOURCE=/home/a6000/gwl/_deps/habitat-lab-v0.3.3 ETPR1_HABITAT_SIM_SOURCE=/home/a6000/gwl/_deps/habitat-sim-v0.3.3 bash scripts/build_etpr1_habitat.sh"'
```

Expected：`.runtime/etpr1_habitat/prefix/site-packages/habitat`、`habitat_sim` 和 `prefix/habitat-baselines/habitat_baselines` 都存在。

- [ ] **Step 6：运行环境检查**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh python scripts/inspect_etpr1_runtime.py"'
```

Expected：Python `3.11.15`、PyTorch `2.2.2+cu121`、Transformers `4.49.0`；三个 Habitat 模块都来自 ETP-R1 的 `.runtime/etpr1_habitat`，输出中不含 `/ETPNav`。

- [ ] **Step 7：运行约定测试并提交**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_runtime_contract.py"'
```

Expected：`2 passed`。

```bash
git add scripts/build_etpr1_habitat.sh scripts/etpr1_rae_runtime_exec.sh scripts/inspect_etpr1_runtime.py tests/test_runtime_contract.py
git add vendor/legacy_clip
git commit -m "build: add isolated ETP-R1 eval runtime"
```

---

### Task 2：补齐现代 Python 和 Habitat 兼容层

**Files:**
- Create: `vlnce_baselines/common/runtime_compat.py`
- Modify: `vlnce_baselines/__init__.py`
- Modify: `habitat_extensions/__init__.py`
- Modify: `vlnce_baselines/ss_trainer_ETP_R1.py`
- Modify: `vlnce_baselines/GRPO_trainer_ETP_R1.py`
- Modify: `pretrain_src/pretrain_src/data/common.py`
- Modify: `vlnce_baselines/waypoint_pred/TRM_net.py`
- Test: `tests/test_modern_runtime_imports.py`

- [ ] **Step 1：写失败的现代运行测试**

```python
import importlib
import numpy as np


def test_removed_numpy_bool_alias_is_not_used():
    for module_name in (
        "vlnce_baselines.ss_trainer_ETP_R1",
        "vlnce_baselines.GRPO_trainer_ETP_R1",
        "pretrain_src.pretrain_src.data.common",
    ):
        source = importlib.import_module(module_name).__loader__.get_source(module_name)
        assert "dtype=np.bool)" not in source


def test_main_entrypoints_import_in_modern_runtime():
    import run
    import habitat_extensions.task
    import vlnce_baselines.ss_trainer_ETP_R1
    import vlnce_baselines.GRPO_trainer_ETP_R1
    assert np.__version__.startswith("1.26")
```

- [ ] **Step 2：运行测试确认旧代码失败**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_modern_runtime_imports.py"'
```

Expected：因旧 Habitat 配置入口、`np.bool` 或 `pytorch_transformers` 兼容问题失败。

- [ ] **Step 3：移植最小兼容层**

只读参考 ETPNav 中已经验证的：

```text
vlnce_baselines/common/runtime_compat.py
vlnce_baselines/__init__.py 中 Habitat config bootstrap
habitat_extensions/__init__.py 中 DictConfig 兼容入口
```

用 `apply_patch` 将兼容逻辑复制进本工程，并保留 ETP-R1 原有 trainer/policy 注册。把三个旧别名改为：

```python
cand_idxes = np.zeros(12, dtype=np.bool_)
```

把 `pretrain_src/pretrain_src/data/common.py` 中的零长度 mask 改为：

```python
return np.zeros((len(seq_lens), 0), dtype=np.bool_)
```

把路点预测器配置导入改为：

```python
from transformers import BertConfig
```

- [ ] **Step 4：运行导入测试和入口帮助**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_modern_runtime_imports.py && scripts/etpr1_rae_runtime_exec.sh python run.py --help"'
```

Expected：测试通过，`run.py --help` 退出码为 0。

- [ ] **Step 5：提交兼容层**

```bash
git add vlnce_baselines habitat_extensions pretrain_src/pretrain_src/data/common.py tests/test_modern_runtime_imports.py
git commit -m "fix: support ETP-R1 in modern eval runtime"
```

---

### Task 3：实现共享的三层 RGB 投影

**Files:**
- Create: `model_components/__init__.py`
- Create: `model_components/rgb_projection.py`
- Test: `tests/test_rgb_projection.py`

- [ ] **Step 1：写失败测试**

```python
import torch

from model_components.rgb_projection import build_rgb_projection


def test_rae_projection_is_three_linear_layers():
    projection = build_rgb_projection("rae_dinov2", 768, 512, 768)
    linear = [m for m in projection if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [
        (768, 768), (768, 768), (768, 512)
    ]
    assert projection(torch.randn(2, 36, 768)).shape == (2, 36, 512)


def test_clip_projection_is_identity():
    projection = build_rgb_projection("clip", 512, 512, 768)
    x = torch.randn(2, 12, 512)
    assert projection(x) is x


def test_projection_rejects_wrong_dimensions():
    try:
        build_rgb_projection("rae_dinov2", 512, 512, 768)
    except ValueError as exc:
        assert "768" in str(exc)
    else:
        raise AssertionError("RAE raw dimension mismatch must fail")
```

- [ ] **Step 2：运行测试确认模块不存在**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rgb_projection.py"'
```

Expected：FAIL，提示 `model_components.rgb_projection` 不存在。

- [ ] **Step 3：实现最小投影工厂**

```python
import torch.nn as nn


def build_rgb_projection(encoder_type, raw_size, output_size, hidden_size):
    encoder_type = str(encoder_type).lower()
    raw_size = int(raw_size)
    output_size = int(output_size)
    hidden_size = int(hidden_size)
    if encoder_type == "clip":
        if raw_size != 512 or output_size != 512:
            raise ValueError("CLIP requires raw_size=output_size=512")
        return nn.Identity()
    if encoder_type != "rae_dinov2":
        raise ValueError(f"Unsupported RGB encoder type: {encoder_type}")
    if raw_size != 768 or output_size != 512 or hidden_size != 768:
        raise ValueError("RAE/DINOv2 projection requires 768->768->768->512")
    return nn.Sequential(
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 512),
    )
```

- [ ] **Step 4：运行测试并提交**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rgb_projection.py"'
```

Expected：`3 passed`。

```bash
git add model_components tests/test_rgb_projection.py
git commit -m "feat: add shared RAE RGB projection"
```

---

### Task 4：实现始终冻结的 RAE/DINOv2 CLS 编码器

**Files:**
- Create: `vlnce_baselines/models/encoders/rae_dinov2_encoder.py`
- Create: `tests/test_rae_dinov2_encoder.py`
- Create: `tests/integration/test_rae_dinov2_parity.py`

- [ ] **Step 1：写输入、冻结和统计量测试**

测试使用假的 DINO 主干，不加载大权重：

```python
import torch

from vlnce_baselines.models.encoders.rae_dinov2_encoder import normalize_rae_cls


def test_spatial_variance_is_averaged_for_cls():
    cls = torch.tensor([[2.0, 4.0]])
    var = torch.tensor([[[1.0, 3.0]], [[4.0, 12.0]]])
    out = normalize_rae_cls(cls, mean=None, var=var, eps=0.0)
    expected = cls / torch.sqrt(torch.tensor([[2.0, 8.0]]))
    torch.testing.assert_close(out, expected)


def test_encoder_rejects_non_224_rgb(fake_encoder):
    try:
        fake_encoder({"rgb": torch.zeros(1, 200, 224, 3, dtype=torch.uint8)})
    except ValueError as exc:
        assert "224x224" in str(exc)
    else:
        raise AssertionError("non-native RGB must fail")


def test_encoder_stays_frozen_after_train(fake_encoder):
    fake_encoder.train()
    assert not fake_encoder.backbone.training
    assert all(not p.requires_grad for p in fake_encoder.backbone.parameters())
```

- [ ] **Step 2：运行测试确认编码器不存在**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rae_dinov2_encoder.py"'
```

Expected：FAIL，提示模块不存在。

- [ ] **Step 3：实现 encoder-only 封装**

实现以下公开接口和固定行为：

```python
class RaeDinov2ClsEncoder(torch.nn.Module):
    output_size = 768

    def __init__(self, model_dir, stat_path, device):
        super().__init__()
        self.backbone = Dinov2WithRegistersModel.from_pretrained(
            model_dir, local_files_only=True
        )
        self.backbone.layernorm.elementwise_affine = False
        self.backbone.layernorm.weight = None
        self.backbone.layernorm.bias = None
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        processor = AutoImageProcessor.from_pretrained(
            model_dir, local_files_only=True
        )
        self.register_buffer("image_mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(processor.image_std).view(1, 3, 1, 1))
        stats = torch.load(stat_path, map_location="cpu")
        self.register_buffer("latent_var", stats["var"].float())
        mean = stats.get("mean")
        if mean is None:
            self.latent_mean = None
        else:
            self.register_buffer("latent_mean", mean.float())
        self.to(device)

    def train(self, mode=True):
        super().train(False)
        self.backbone.eval()
        return self

    @property
    def is_blind(self):
        return False

    @torch.no_grad()
    def forward(self, observations):
        rgb = observations["rgb"]
        if tuple(rgb.shape[1:3]) != (224, 224):
            raise ValueError(f"RAE/DINOv2 requires native 224x224 RGB, got {tuple(rgb.shape)}")
        rgb = rgb.permute(0, 3, 1, 2).contiguous().float().div(255.0)
        rgb = (rgb - self.image_mean) / self.image_std
        cls = self.backbone(rgb).last_hidden_state[:, 0].float()
        cls = normalize_rae_cls(cls, self.latent_mean, self.latent_var, eps=1e-5)
        if not torch.isfinite(cls).all():
            raise FloatingPointError("RAE/DINOv2 CLS contains NaN or infinity")
        return cls
```

`normalize_rae_cls()` 必须复现 RAE `_stat_for_shape()` 对二维 CLS 的规则：`[768,16,16]` 的统计量先在空间维求均值，再进行 `(cls-mean)/sqrt(var+1e-5)`。禁止联网回退。

- [ ] **Step 4：复制本地模型资产到测评机 ETP-R1 目录**

```bash
ssh 4090 'mkdir -p /home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base && cp -a /home/a6000/gwl/RAE-NWM/raenwm/models/encoders/dinov2-with-registers-base/config.json /home/a6000/gwl/RAE-NWM/raenwm/models/encoders/dinov2-with-registers-base/preprocessor_config.json /home/a6000/gwl/RAE-NWM/raenwm/models/encoders/dinov2-with-registers-base/model.safetensors /home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base/ && cp -a /home/a6000/gwl/RAE-NWM/raenwm/models/stats/dinov2/wReg_base/imagenet1k/stat.pt /home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base/stat.pt'
ssh 4090 'sha256sum /home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base/model.safetensors /home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base/stat.pt > /home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base/SHA256SUMS'
```

- [ ] **Step 5：做真实 RAE 数值一致性测试**

`tests/integration/test_rae_dinov2_parity.py` 用固定随机种子生成一张 `[1,3,224,224]` 的 `[0,1]` 图片。参考分支从测评机只读的 `/home/a6000/gwl/ETPNav/vlnce_baselines/nwm/raenwm_core/RAE` 构建原始 RAE，生产分支使用 `RaeDinov2ClsEncoder`。测试只在单独子进程临时设置参考路径，不把 ETPNav 写入环境 `.pth` 或正式运行脚本。

断言：

```python
assert reference.shape == actual.shape == (1, 768)
assert (reference - actual).abs().max().item() <= 1e-5
assert torch.nn.functional.cosine_similarity(reference, actual).item() >= 0.999999
```

运行：

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rae_dinov2_encoder.py tests/integration/test_rae_dinov2_parity.py"'
```

Expected：全部通过；真实一致性测试不得跳过。

- [ ] **Step 6：提交编码器**

```bash
git add vlnce_baselines/models/encoders/rae_dinov2_encoder.py tests/test_rae_dinov2_encoder.py tests/integration/test_rae_dinov2_parity.py
git commit -m "feat: add frozen RAE DINOv2 CLS encoder"
```

---

### Task 5：接通离线预训练的 768→512 投影

**Files:**
- Modify: `pretrain_src/pretrain_src/data/dataset.py`
- Modify: `pretrain_src/pretrain_src/train_r2r.py`
- Modify: `pretrain_src/pretrain_src/model/vilmodel.py`
- Create: `pretrain_src/run_pt/mix_model_config_rae_dino.json`
- Create: `pretrain_src/run_pt/mix_pretrain_rae_dino.json`
- Create: `pretrain_src/run_pt/run_mix_rae_dino.bash`
- Test: `tests/test_pretrain_rae_projection.py`

- [ ] **Step 1：写失败的数据维度和梯度测试**

```python
import torch
from transformers import PretrainedConfig

from pretrain_src.pretrain_src.model.vilmodel import ImageEmbeddings


def _config():
    return PretrainedConfig(
        rgb_encoder_type="rae_dinov2",
        raw_image_feat_size=768,
        image_feat_size=512,
        projection_hidden_size=768,
        hidden_size=768,
        depth_feat_size=128,
        angle_feat_size=4,
        obj_feat_size=0,
        hidden_dropout_prob=0.0,
        num_pano_layers=0,
        layer_norm_eps=1e-5,
    )


def test_pretrain_projection_has_expected_parameter_name_and_gradient():
    module = ImageEmbeddings(_config())
    names = dict(module.named_parameters())
    assert "rgb_projection.0.weight" in names
    x = torch.randn(2, 36, 768, requires_grad=True)
    y = module.project_rgb(x)
    assert y.shape == (2, 36, 512)
    y.sum().backward()
    assert names["rgb_projection.0.weight"].grad is not None
```

- [ ] **Step 2：运行测试确认旧模型没有投影**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_pretrain_rae_projection.py"'
```

Expected：FAIL，提示 `rgb_projection` 或 `project_rgb` 不存在。

- [ ] **Step 3：分离原始特征维度和后续维度**

给 `R2RTextPathData` 和基类新增 `raw_image_feat_size`，HDF5 数据只按原始维度截取：

```python
self.raw_image_feat_size = int(raw_image_feat_size)
...
"traj_view_img_fts": [x[:, :self.raw_image_feat_size] for x in traj_view_img_fts]
```

读取 dataset 时检查每个 key 的第二维至少等于 `raw_image_feat_size`，不足时抛出包含 key、实际 shape 和期望维度的 `ValueError`。`train_r2r.py` 三处数据集构造都传：

```python
raw_image_feat_size=model_config.raw_image_feat_size
```

RAE 配置初始化数据集时还必须打开一次 HDF5 根属性并逐项校验：

```python
expected = {
    "feature_extractor": "rae_dinov2_with_registers_base_cls",
    "feature_dim": 768,
    "dtype": "float32",
    "num_views": 36,
    "image_size": 224,
    "vfov": 60,
    "latent_normalized": True,
}
for key, value in expected.items():
    if handle.attrs.get(key) != value:
        raise ValueError(
            f"DINO HDF5 metadata mismatch for {key}: "
            f"expected {value!r}, got {handle.attrs.get(key)!r}"
        )
```

`tests/test_pretrain_rae_projection.py` 增加一个临时错误 HDF5，用 `feature_dim=512` 断言数据集初始化立即失败。

- [ ] **Step 4：在预训练 ImageEmbeddings 中加入投影**

构造函数加入：

```python
self.rgb_projection = build_rgb_projection(
    config.rgb_encoder_type,
    config.raw_image_feat_size,
    config.image_feat_size,
    config.projection_hidden_size,
)
```

加入方法并在 `img_linear` 前调用：

```python
def project_rgb(self, features):
    projected = self.rgb_projection(features)
    if projected.shape[-1] != 512:
        raise ValueError(f"RGB projection must output 512 dims, got {projected.shape[-1]}")
    if not torch.isfinite(projected).all():
        raise FloatingPointError("RGB projection contains NaN or infinity")
    return projected

traj_view_img_fts = self.project_rgb(traj_view_img_fts)
traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
```

- [ ] **Step 5：新增独立预训练配置**

`mix_model_config_rae_dino.json` 从现有文件复制，明确设置：

```json
"rgb_encoder_type": "rae_dinov2",
"raw_image_feat_size": 768,
"image_feat_size": 512,
"projection_hidden_size": 768,
"image_prob_size": 0
```

`mix_pretrain_rae_dino.json` 保留五类训练数据和两个验证集，只修改：

```json
"img_ft_file": "pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5"
```

启动脚本使用单卡测评机和独立输出：

```bash
NUM_GPUS=1
outdir=pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp
torchrun --nproc_per_node=1 --node_rank 0 --master_port="$1" \
  pretrain_src/pretrain_src/train_r2r.py --world_size 1 --vlnbert cmt \
  --model_config pretrain_src/run_pt/mix_model_config_rae_dino.json \
  --config pretrain_src/run_pt/mix_pretrain_rae_dino.json --output_dir "$outdir"
```

- [ ] **Step 6：运行投影测试并提交**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_pretrain_rae_projection.py"'
```

Expected：PASS，MLP 梯度非空。

```bash
git add pretrain_src tests/test_pretrain_rae_projection.py
git commit -m "feat: project offline DINO CLS features"
```

---

### Task 6：接通在线 SFT、GRPO 和评测链路

**Files:**
- Modify: `vlnce_baselines/config/default.py`
- Modify: `vlnce_baselines/models/R1Policy.py`
- Modify: `vlnce_baselines/models/etp/ETP_R1_vilmodel_cmt.py`
- Modify: `vlnce_baselines/models/etp/ETP_R1_vlnbert_init.py`
- Create: `run_r2r/iter_train_rae_dino.yaml`
- Create: `run_rxr/iter_train_rae_dino.yaml`
- Test: `tests/test_online_rae_projection.py`

- [ ] **Step 1：写失败的在线分支测试**

使用假的 RGB 编码器和假的路点预测器，断言 768 维只投影一次：

```python
def test_online_waypoint_returns_512_dim_features(rae_policy_with_fakes):
    outputs = rae_policy_with_fakes(mode="waypoint", observations=rae_policy_with_fakes.observations, in_train=False)
    assert outputs["pano_rgb"].shape[-1] == 512
    assert all(x.shape[-1] == 512 for x in outputs["cand_rgb"])


def test_dino_is_frozen_but_projection_is_trainable(rae_policy_with_fakes):
    assert all(not p.requires_grad for p in rae_policy_with_fakes.rgb_encoder.parameters())
    projection = rae_policy_with_fakes.vln_bert.img_embeddings.rgb_projection
    assert all(p.requires_grad for p in projection.parameters())
```

- [ ] **Step 2：运行测试确认当前只有 CLIP**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_online_rae_projection.py"'
```

Expected：FAIL，因为 `R1Policy` 固定构建 `CLIPEncoder`。

- [ ] **Step 3：扩展配置并按类型构建编码器**

默认配置保持 CLIP：

```python
_C.MODEL.RGB_ENCODER.type = "clip"
_C.MODEL.RGB_ENCODER.model_dir = ""
_C.MODEL.RGB_ENCODER.stat_path = ""
_C.MODEL.RGB_ENCODER.raw_output_size = 512
_C.MODEL.RGB_ENCODER.output_size = 512
_C.MODEL.RGB_ENCODER.projection_hidden_size = 768
```

`R1Policy.ETP.__init__()` 改为显式分支：

```python
if model_config.RGB_ENCODER.type == "clip":
    self.rgb_encoder = CLIPEncoder(self.device)
elif model_config.RGB_ENCODER.type == "rae_dinov2":
    self.rgb_encoder = RaeDinov2ClsEncoder(
        model_config.RGB_ENCODER.model_dir,
        model_config.RGB_ENCODER.stat_path,
        self.device,
    )
else:
    raise ValueError(f"Unsupported RGB encoder: {model_config.RGB_ENCODER.type}")
```

- [ ] **Step 4：给在线 ImageEmbeddings 加同名投影**

在线 `ImageEmbeddings` 也构造 `self.rgb_projection` 并提供相同 `project_rgb()`。`forward_panorama()` 继续只做 `img_linear(512→768)`，不重复投影。`R1Policy` waypoint 分支改为：

```python
raw_rgb_embedding = self.rgb_encoder(obs_view12)
rgb_embedding = self.vln_bert.img_embeddings.project_rgb(raw_rgb_embedding)
waypoint_heatmap_logits = waypoint_predictor(rgb_embedding, depth_embedding)
rgb_embed_reshape = rgb_embedding.reshape(batch_size, NUM_IMGS, 512, 1, 1)
```

- [ ] **Step 5：让在线模型配置知道投影类型**

`ETP_R1_vlnbert_init.py` 设置：

```python
vis_config.rgb_encoder_type = config.RGB_ENCODER.type
vis_config.raw_image_feat_size = config.RGB_ENCODER.raw_output_size
vis_config.image_feat_size = config.RGB_ENCODER.output_size
vis_config.projection_hidden_size = config.RGB_ENCODER.projection_hidden_size
```

加载预训练 checkpoint 前检查：DINO 配置必须存在 `bert.img_embeddings.rgb_projection.0.weight`；CLIP 配置若出现该键则拒绝加载。

- [ ] **Step 6：新增 R2R/RxR 专用配置**

两份配置均设置：

```yaml
MODEL:
  RGB_ENCODER:
    type: rae_dinov2
    model_dir: pretrained/rae_dinov2_with_registers_base
    stat_path: pretrained/rae_dinov2_with_registers_base/stat.pt
    raw_output_size: 768
    output_size: 512
    projection_hidden_size: 768
```

两份配置的预训练入口固定为：

```yaml
MODEL:
  pretrained_path: pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp/ckpts/model_step_500000.pt
```

R2R 保留 HFOV 90°，并使用：

```yaml
TENSORBOARD_DIR: data/logs/rae_dinov2/r2r/tensorboard/
CHECKPOINT_FOLDER: data/logs/rae_dinov2/r2r/checkpoints/
RESULTS_DIR: data/logs/rae_dinov2/r2r/results/
```

RxR 保留 HFOV 63°，并使用：

```yaml
TENSORBOARD_DIR: data/logs/rae_dinov2/rxr/tensorboard/
CHECKPOINT_FOLDER: data/logs/rae_dinov2/rxr/checkpoints/
RESULTS_DIR: data/logs/rae_dinov2/rxr/results/
```

深度配置、路点预测器和任务定义全部保持原值。

- [ ] **Step 7：运行在线测试并提交**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_online_rae_projection.py"'
```

Expected：PASS；DINO 全冻结，MLP 可训练，输出均为 512 维。

```bash
git add vlnce_baselines/config/default.py vlnce_baselines/models run_r2r/iter_train_rae_dino.yaml run_rxr/iter_train_rae_dino.yaml tests/test_online_rae_projection.py
git commit -m "feat: route online navigation through RAE DINOv2"
```

---

### Task 7：过滤冻结 DINO checkpoint 并校验类型

**Files:**
- Create: `vlnce_baselines/models/checkpoint_utils.py`
- Modify: `vlnce_baselines/ss_trainer_ETP_R1.py`
- Modify: `vlnce_baselines/GRPO_trainer_ETP_R1.py`
- Test: `tests/test_rae_checkpoint.py`

- [ ] **Step 1：写失败的过滤和映射测试**

```python
def test_dino_backbone_is_filtered_but_projection_is_saved(fake_rae_policy):
    state, meta = navigation_state_dict(fake_rae_policy, fake_rae_policy.config)
    assert not any("rgb_encoder.backbone" in key for key in state)
    assert any("img_embeddings.rgb_projection.0.weight" in key for key in state)
    assert meta["type"] == "rae_dinov2"
    assert len(meta["model_sha256"]) == 64
    assert len(meta["stat_sha256"]) == 64


def test_grpo_keeps_projection_frozen(fake_grpo_trainer):
    fake_grpo_trainer.setup_training_parts()
    projection = fake_grpo_trainer.policy.net.vln_bert.img_embeddings.rgb_projection
    assert all(not p.requires_grad for p in projection.parameters())
```

- [ ] **Step 2：运行测试确认当前 checkpoint 保存全部 DINO**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rae_checkpoint.py"'
```

Expected：FAIL，因为过滤工具不存在。

- [ ] **Step 3：实现 checkpoint 工具**

公开函数固定为：

```python
def navigation_state_dict(policy, config):
    state = policy.state_dict()
    if config.MODEL.RGB_ENCODER.type == "clip":
        return state, {"type": "clip"}
    filtered = {
        key: value for key, value in state.items()
        if ".rgb_encoder.backbone." not in key
    }
    metadata = {
        "type": "rae_dinov2",
        "model_dir": config.MODEL.RGB_ENCODER.model_dir,
        "model_sha256": sha256_file(Path(config.MODEL.RGB_ENCODER.model_dir) / "model.safetensors"),
        "stat_sha256": sha256_file(config.MODEL.RGB_ENCODER.stat_path),
        "raw_output_size": 768,
        "output_size": 512,
    }
    return filtered, metadata
```

加载工具必须拒绝：类型不一致、哈希不一致、维度不一致、DINO checkpoint 缺失 MLP；只忽略 `.rgb_encoder.backbone.` 对应的已知 missing keys，其他 missing/unexpected keys 继续完整打印。

- [ ] **Step 4：修改两个 trainer 的保存和加载**

两个 `save_checkpoint()` 都先调用：

```python
state_dict, rgb_encoder_meta = navigation_state_dict(self.policy, self.config)
```

保存对象加入：

```python
"state_dict": state_dict,
"rgb_encoder": rgb_encoder_meta,
```

加载时在 `load_state_dict()` 前调用 `validate_rgb_checkpoint_metadata()`。SFT checkpoint 中 MLP 保持可训练；GRPO 的 `setup_training_parts()` 不把 `img_embeddings` 加进解冻列表。

- [ ] **Step 5：运行 checkpoint 测试并提交**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rae_checkpoint.py"'
```

Expected：PASS；假的 DINO key 被过滤，MLP key 保留，GRPO 中 MLP 冻结。

```bash
git add vlnce_baselines/models/checkpoint_utils.py vlnce_baselines/ss_trainer_ETP_R1.py vlnce_baselines/GRPO_trainer_ETP_R1.py tests/test_rae_checkpoint.py
git commit -m "feat: isolate frozen DINO checkpoint state"
```

---

### Task 8：实现全量 DINO CLS HDF5 生成和校验

**Files:**
- Create: `precompute_img_features/extract_rae_dinov2_features.py`
- Create: `precompute_img_features/validate_rae_dinov2_features.py`
- Test: `tests/test_rae_feature_hdf5.py`

- [ ] **Step 1：写失败的几何、断点续写和元数据测试**

测试使用两个假视点、假的 simulator 和假的 encoder：

```python
def test_view_index_covers_three_elevations_and_twelve_headings():
    rotations = [view_index_to_rotation_quat(i) for i in range(36)]
    assert len(rotations) == 36
    assert len({repr(x) for x in rotations}) == 36


def test_render_pipeline_keeps_rgb_channel_order(fake_red_pixel_simulator):
    rendered = render_36_views(fake_red_pixel_simulator, [0.0, 0.0, 0.0])
    assert rendered[0, 0, 0].tolist() == [255, 0, 0]


def test_resume_keeps_complete_keys_and_replaces_bad_shape(tmp_path, fake_pipeline):
    output = tmp_path / "features.hdf5"
    fake_pipeline.write(output, keys=["scan_a"])
    fake_pipeline.write_bad_shape(output, "scan_b")
    fake_pipeline.run(output, resume=True)
    with h5py.File(output, "r") as handle:
        assert handle["scan_a"].shape == (36, 768)
        assert handle["scan_b"].shape == (36, 768)
        assert handle.attrs["latent_normalized"]
        assert handle.attrs["vfov"] == 60
```

- [ ] **Step 2：运行测试确认脚本不存在**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rae_feature_hdf5.py"'
```

Expected：FAIL，提示生成模块不存在。

- [ ] **Step 3：实现直接渲染和编码**

以只读方式参考 ETPNav 的 `pretrain_src/pretrain_src/extract_dino_features.py`，保留 connectivity 位置转换、36 个视角四元数和按 scan 复用 simulator 的结构，做以下固定修改：

```text
模型：RaeDinov2ClsEncoder
图像：224×224 RGB
离线视场角：60°
输出：每个 key 为 [36,768] float32
默认文件：pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5
默认 key 集：connectivity 中全部 10,567 个 included viewpoint
```

每个 dataset 使用 `dtype=np.float32` 和 `compression="gzip"`。每完成一个视点立即 `flush()`；已存在且 shape/dtype/有限值正确的 key 跳过，损坏 key 删除后重算。Habitat-Sim 输出直接按 RGB 使用，禁止再做 `[..., ::-1]`。根属性必须完整写入：

```text
feature_extractor=rae_dinov2_with_registers_base_cls
feature_dim=768
dtype=float32
num_views=36
image_size=224
vfov=60
sensor_height=1.25
latent_normalized=true
dino_weights_sha256=sha256_file(model.safetensors) 计算出的 64 位摘要
rae_stat_sha256=sha256_file(stat.pt) 计算出的 64 位摘要
preprocess_version=rae_native_224_rgb_v1
```

- [ ] **Step 4：实现独立校验器**

校验器必须一次扫描全部 key，输出 JSON 摘要并在任一条件不满足时退出非零：

```python
assert len(keys) == 10567
assert keys == connectivity_keys == clip_hdf5_keys
assert shape == (36, 768)
assert dtype == np.float32
assert np.isfinite(array).all()
assert not np.all(array == 0)
```

- [ ] **Step 5：运行单元测试并提交**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/test_rae_feature_hdf5.py"'
```

Expected：PASS。

```bash
git add precompute_img_features/extract_rae_dinov2_features.py precompute_img_features/validate_rae_dinov2_features.py tests/test_rae_feature_hdf5.py
git commit -m "feat: generate RAE DINOv2 CLS features"
```

---

### Task 9：生成并验证全量 1.2GB HDF5

**Files:**
- Generated on eval host only: `pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5`
- Generated on eval host only: `data/logs/rae_dino_feature_generation/*.log`

- [ ] **Step 1：同步必要数据，不覆盖远端生成目录**

先同步代码，再单独同步 connectivity、MP3D 软链接目标、现有 CLIP/深度 HDF5 和预训练 JSONL；全程不使用 `--delete`：

```bash
rsync -a /home/gwl/project/etpr1/ETP-R1/pretrain_src/datasets/ 4090:/home/a6000/gwl/ETP-R1/pretrain_src/datasets/
rsync -a /home/gwl/project/etpr1/ETP-R1/pretrain_src/img_features/CLIP-ViT-B-32-views-habitat.hdf5 /home/gwl/project/etpr1/ETP-R1/pretrain_src/img_features/ddppo_resnet50_depth_features.hdf5 4090:/home/a6000/gwl/ETP-R1/pretrain_src/img_features/
ssh 4090 'mkdir -p /home/a6000/gwl/ETP-R1/data/scene_datasets && if [ ! -d /home/a6000/gwl/ETP-R1/data/scene_datasets/mp3d ]; then cp -a /home/a6000/gwl/ETPNav/data/scene_datasets/mp3d /home/a6000/gwl/ETP-R1/data/scene_datasets/mp3d; fi'
```

场景文件从受保护工程只读复制一次，后续运行只访问 ETP-R1 自有副本，不修改或链接 ETPNav 工程文件。

- [ ] **Step 2：生成一个视点并在线复算**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh python precompute_img_features/extract_rae_dinov2_features.py --max_viewpoints 1 --output_file /home/a6000/gwl/ETP-R1/.runtime/rae_dino_one_view.hdf5"'
```

Expected：恰好一个 `[36,768] float32` key；抽取一个方向重新编码，最大绝对误差不超过 `1e-5`。

- [ ] **Step 3：确认 GPU 空闲后生成全量文件**

```bash
ssh 4090 'nvidia-smi; docker exec gwl-etpnav bash -lc "ps -eo pid,ppid,stat,etime,cmd | grep -E '\''torchrun|run.py|train.py'\'' | grep -v grep || true"'
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && mkdir -p data/logs/rae_dino_feature_generation && scripts/etpr1_rae_runtime_exec.sh python precompute_img_features/extract_rae_dinov2_features.py --output_file pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5 2>&1 | tee data/logs/rae_dino_feature_generation/full.log"'
```

若检查发现新的 ETPNav GPU 任务，停止本步骤并报告，不能使用之前那次停止授权。

- [ ] **Step 4：运行全量校验**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh python precompute_img_features/validate_rae_dinov2_features.py --features pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5 --clip_features pretrain_src/img_features/CLIP-ViT-B-32-views-habitat.hdf5 --connectivity pretrain_src/datasets/R2R/connectivity"'
```

Expected：`keys=10567`、`bad_shape=0`、`non_finite=0`、`all_zero=0`、`missing=0`、`extra=0`。

---

### Task 10：验证权重传递和三阶段训练规则

**Files:**
- Create: `tests/integration/test_pretrain_to_online_projection.py`
- Create: `tests/integration/test_training_stage_freeze.py`
- Create: `tests/integration/test_pretrain_tasks_smoke.py`
- Create: `scripts/smoke_rae_dino.sh`

- [ ] **Step 1：同步在线冒烟所需的只读资产**

```bash
rsync -a /home/gwl/project/etpr1/ETP-R1/data/datasets/ 4090:/home/a6000/gwl/ETP-R1/data/datasets/
rsync -a /home/gwl/project/etpr1/ETP-R1/data/wp_pred/ 4090:/home/a6000/gwl/ETP-R1/data/wp_pred/
rsync -a /home/gwl/project/etpr1/ETP-R1/bert_config/ 4090:/home/a6000/gwl/ETP-R1/bert_config/
```

Expected：R2R/RxR 数据集、两个路点预测器权重和本地 XLM-RoBERTa 权重都存在于 ETP-R1 自有目录。

- [ ] **Step 2：写预训练到在线的权重映射测试**

测试给预训练模型的 `bert.img_embeddings.rgb_projection.*` 写入固定递增值，保存 state dict，再通过 `get_vlnbert_models()` 构建在线模型，逐参数断言：

```python
for name, expected in pretrain_projection.state_dict().items():
    actual = online_model.img_embeddings.rgb_projection.state_dict()[name]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

- [ ] **Step 3：写三阶段冻结测试**

```python
def test_pretrain_and_sft_train_projection_but_never_dino(models):
    assert all(not p.requires_grad for p in models.dino.parameters())
    assert all(p.requires_grad for p in models.pretrain_projection.parameters())
    assert all(p.requires_grad for p in models.sft_projection.parameters())


def test_grpo_freezes_projection_and_dino(models):
    assert all(not p.requires_grad for p in models.grpo_projection.parameters())
    assert all(not p.requires_grad for p in models.dino.parameters())
```

- [ ] **Step 4：运行集成测试**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests/integration/test_pretrain_to_online_projection.py tests/integration/test_training_stage_freeze.py"'
```

Expected：全部通过。

- [ ] **Step 5：实现统一冒烟脚本**

`scripts/smoke_rae_dino.sh` 使用 `set -euo pipefail`，按顺序执行：

```text
1. 环境和模型导入
2. 一个 MLM batch 前向/反向
3. 一个 SAP batch 前向/反向
4. 单环境 SFT 前向/反向
5. GRPO 参数冻结检查
6. R2R 单 episode
7. RxR 单 episode
```

每一步写入 `data/logs/rae_dino_smoke/` 独立日志；任一步失败立即退出。MLM/SAP 和 SFT 后检查 `rgb_projection.0.weight.grad` 非空且有限；DINO 梯度始终为空；GRPO 检查 MLP `requires_grad=False`。

脚本中的核心命令固定为：

```bash
pytest -q tests/integration/test_pretrain_tasks_smoke.py -k mlm
pytest -q tests/integration/test_pretrain_tasks_smoke.py -k sap

torchrun --nproc_per_node=1 --master_port=23401 \
  pretrain_src/pretrain_src/train_r2r.py --world_size 1 --vlnbert cmt \
  --model_config pretrain_src/run_pt/mix_model_config_rae_dino.json \
  --config pretrain_src/run_pt/mix_pretrain_rae_dino.json \
  --output_dir pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp_smoke \
  --num_train_steps 1 --train_batch_size 1 --val_batch_size 1 \
  --valid_steps 1000 --log_steps 1 --n_workers 0

python run.py --exp_name rae_smoke_r2r_sft --run-type dagger \
  --exp-config run_r2r/iter_train_rae_dino.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  IL.iters 1 IL.log_every 1 IL.load_from_ckpt False \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp_smoke/ckpts/model_step_1.pt

python run.py --exp_name rae_smoke_r2r_grpo --run-type grpo \
  --exp-config run_r2r/iter_train_rae_dino.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  TRAINER_NAME GRPO-R1 GRPO.iters 1 GRPO.sample_num 2 GRPO.update_epochs 1 \
  GRPO.load_from_ckpt True GRPO.is_requeue False \
  GRPO.ckpt_to_load data/logs/rae_dinov2/r2r/checkpoints/rae_smoke_r2r_sft/ckpt.iter1.pth \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp_smoke/ckpts/model_step_1.pt

python run.py --exp_name rae_smoke_r2r_eval --run-type eval \
  --exp-config run_r2r/iter_train_rae_dino.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.CKPT_PATH_DIR data/logs/rae_dinov2/r2r/checkpoints/rae_smoke_r2r_sft/ckpt.iter1.pth \
  EVAL.EPISODE_COUNT 1 EVAL.SAVE_RESULTS False \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp_smoke/ckpts/model_step_1.pt

python run.py --exp_name rae_smoke_rxr_sft --run-type dagger \
  --exp-config run_rxr/iter_train_rae_dino.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  IL.iters 1 IL.log_every 1 IL.load_from_ckpt False \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp_smoke/ckpts/model_step_1.pt

python run.py --exp_name rae_smoke_rxr_eval --run-type eval \
  --exp-config run_rxr/iter_train_rae_dino.yaml \
  SIMULATOR_GPU_IDS [0] TORCH_GPU_IDS [0] GPU_NUMBERS 1 NUM_ENVIRONMENTS 1 \
  EVAL.CKPT_PATH_DIR data/logs/rae_dinov2/rxr/checkpoints/rae_smoke_rxr_sft/ckpt.iter1.pth \
  EVAL.EPISODE_COUNT 1 EVAL.SAVE_RESULTS False \
  MODEL.pretrained_path pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp_smoke/ckpts/model_step_1.pt
```

- [ ] **Step 6：运行冒烟并提交**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh bash scripts/smoke_rae_dino.sh"'
```

Expected：七个阶段均显示 `PASS`。

```bash
git add tests/integration/test_pretrain_to_online_projection.py tests/integration/test_training_stage_freeze.py scripts/smoke_rae_dino.sh
git commit -m "test: verify RAE DINOv2 training stages"
```

---

### Task 11：运行 CLIP 回归、完整测试和文档收尾

**Files:**
- Modify: `research.md`
- Modify: `docs/superpowers/specs/2026-07-10-rae-dinov2-visual-encoder-design.md`
- Create: `docs/rae-dinov2-eval-host-validation.md`

- [ ] **Step 1：运行所有自动测试**

```bash
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh pytest -q tests"'
```

Expected：0 failed、0 errors；真实 RAE 一致性测试未跳过。

- [ ] **Step 2：运行原 CLIP 配置回归**

使用原 `run_r2r/iter_train.yaml` 和现有 CLIP checkpoint 启动单 episode，断言：

```text
RGB encoder type=clip
RGB output dim=512
读取原 CLIP HDF5
没有 rgb_projection 参数 missing key
不加载 DINO 权重
```

命令：

```bash
ssh 4090 'mkdir -p /home/a6000/gwl/ETP-R1/data/logs/checkpoints/release_r2r_dagger/store'
rsync -a /home/gwl/project/etpr1/ETP-R1/data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth 4090:/home/a6000/gwl/ETP-R1/data/logs/checkpoints/release_r2r_dagger/store/
ssh 4090 'docker exec gwl-etpr1-rae bash -lc "source /home/a6000/gwl/miniconda3/etc/profile.d/conda.sh && conda activate etpr1_rae && cd /home/a6000/gwl/ETP-R1 && scripts/etpr1_rae_runtime_exec.sh python run.py --exp_name clip_regression --run-type eval --exp-config run_r2r/iter_train.yaml EVAL.CKPT_PATH_DIR data/logs/checkpoints/release_r2r_dagger/store/ckpt.iter25000.pth EVAL.EPISODE_COUNT 1 EVAL.SAVE_RESULTS False"'
```

Expected：完成一个 episode，退出码 0。

- [ ] **Step 3：记录性能基线**

在 RTX 4090、float32 下记录 12 个方向编码耗时、单环境显存、SFT iteration 时间和 GRPO rollout 时间。结果写入 `docs/rae-dinov2-eval-host-validation.md`，同时记录命令、commit、权重哈希、环境版本和日志路径。

- [ ] **Step 4：更新项目研究和设计状态**

`research.md` 增加实际创建的容器、环境、运行脚本、测试命令和产物位置。设计文档只有在全部完成标准满足后才把状态改为 `implemented`；如果某个完整训练尚未执行，明确写“冒烟已通过，长训练未启动”，不能写成完整实验已完成。

- [ ] **Step 5：最终验证并提交**

```bash
git diff --check
git status --short
git log --oneline --decorate -12
```

确认代码仓库不包含 HDF5、模型权重、checkpoint 或训练日志后提交：

```bash
git add research.md docs/rae-dinov2-eval-host-validation.md docs/superpowers/specs/2026-07-10-rae-dinov2-visual-encoder-design.md
git commit -m "docs: record RAE DINOv2 eval validation"
```

---

## 最终验收清单

- [ ] `gwl-etpr1-rae`、`etpr1_rae` 和 `.runtime/etpr1_habitat` 均独立存在。
- [ ] 原 `raenwm`、`gwl-etpnav`、ETPNav 和 RAE-NWM 未被修改。
- [ ] DINO encoder-only 输出和 RAE 原链路满足 `max_abs<=1e-5`、`cosine>=0.999999`。
- [ ] 全量 HDF5 恰好 10,567 个 `[36,768] float32` key，约 1.2GB 原始数组数据。
- [ ] 离线和在线都使用 `img_embeddings.rgb_projection`，结构为 `768→768→768→512`。
- [ ] 预训练和 SFT 的 MLP 有梯度，GRPO 的 MLP 冻结，DINO 所有阶段冻结。
- [ ] 在线 checkpoint 不含 DINO 主干，包含 MLP、配置、迭代、优化器/调度器状态和模型哈希。
- [ ] CLIP 原配置、HDF5、checkpoint 和单 episode 回归保持可用。
- [ ] MLM、SAP、SFT、GRPO、R2R 和 RxR 冒烟全部在测评机通过。
- [ ] 本机没有创建环境、运行测试、生成 HDF5 或启动实验。
