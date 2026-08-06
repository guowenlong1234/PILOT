import json
import inspect
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, PretrainedConfig

from pretrain_src.pretrain_src.data.dataset import R2RTextPathData
from pretrain_src.pretrain_src.data.tasks import (
    MlmDataset,
    SapDataset,
    mlm_collate,
    sap_collate,
)
from pretrain_src.pretrain_src.model.pretrain_cmt import (
    GlocalTextPathCMTPreTraining,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_DATA = PROJECT_ROOT / "pretrain_src" / "datasets" / "R2R"
IMAGE_FEATURES = Path(
    os.environ.get(
        "ETPR1_RAE_SMOKE_IMAGE_FEATURES",
        PROJECT_ROOT
        / "pretrain_src"
        / "img_features"
        / "RAE-DINOv2-B-14-RAW-CLS-views-habitat.hdf5",
    )
)
DEPTH_FEATURES = (
    PROJECT_ROOT
    / "pretrain_src"
    / "img_features"
    / "ddppo_resnet50_depth_features.hdf5"
)
REAL_JSONL = (
    PRETRAIN_DATA
    / "annotations"
    / "pretrain_R2R_RxR"
    / "R2R_train_enc_xlmr.jsonl"
)
SCANVP_CANDIDATES = (
    PRETRAIN_DATA / "annotations" / "scanvp_candview_relangles.json"
)
CONNECTIVITY = PRETRAIN_DATA / "connectivity"
MODEL_CONFIG = (
    PROJECT_ROOT
    / "pretrain_src"
    / "run_pt"
    / "mix_model_config_rae_dino.json"
)
TOKENIZER = PROJECT_ROOT / "bert_config" / "xlm-roberta-base"
IMG_LINEAR_PREFIX = "bert.img_embeddings.img_linear."
SMOKE_SEED = 20260710


@dataclass
class _PretrainContext:
    model: GlocalTextPathCMTPreTraining
    batches: dict
    device: torch.device


def _seed_pretrain_smoke(seed=SMOKE_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _assert_real_assets_exist():
    required_files = (
        IMAGE_FEATURES,
        DEPTH_FEATURES,
        REAL_JSONL,
        SCANVP_CANDIDATES,
        MODEL_CONFIG,
        TOKENIZER / "pytorch_model.bin",
        TOKENIZER / "sentencepiece.bpe.model",
    )
    for path in required_files:
        assert path.is_file(), f"missing required pretrain smoke asset: {path}"
    assert CONNECTIVITY.is_dir(), (
        f"missing required connectivity directory: {CONNECTIVITY}"
    )


def _write_one_real_record(source, destination):
    with source.open("r", encoding="utf-8") as reader:
        first_line = reader.readline()
    assert first_line, f"real pretrain JSONL is empty: {source}"
    record = json.loads(first_line)
    for key in (
        "scan",
        "path",
        "heading",
        "instr_encoding",
        "task_type_encoding",
    ):
        assert key in record, f"real pretrain record is missing {key!r}"
    destination.write_text(first_line, encoding="utf-8")
    return record


def _move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@pytest.fixture(scope="module")
def real_pretrain_context(tmp_path_factory):
    _seed_pretrain_smoke()
    _assert_real_assets_exist()
    assert torch.cuda.is_available(), "pretrain smoke requires a CUDA GPU"
    device = torch.device("cuda", 0)

    tmp_dir = tmp_path_factory.mktemp("real_pretrain_record")
    one_record = tmp_dir / "one_real_r2r_record.jsonl"
    _write_one_real_record(REAL_JSONL, one_record)

    nav_db = R2RTextPathData(
        [str(one_record)],
        str(IMAGE_FEATURES),
        str(DEPTH_FEATURES),
        str(SCANVP_CANDIDATES),
        str(CONNECTIVITY),
        image_feat_size=768,
        image_prob_size=0,
        depth_feat_size=128,
        angle_feat_size=4,
        max_txt_len=250,
        in_memory=False,
        raw_image_feat_size=768,
        rgb_encoder_type="rae_dinov2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER,
        local_files_only=True,
    )
    _seed_pretrain_smoke()
    mlm_batch = mlm_collate([MlmDataset(nav_db, tokenizer)[0]])
    _seed_pretrain_smoke()
    sap_batch = sap_collate(
        [SapDataset(nav_db, tokenizer, end_vp_pos_ratio=1.0)[0]]
    )

    _seed_pretrain_smoke()
    config = PretrainedConfig.from_json_file(MODEL_CONFIG)
    config.pretrain_tasks = {"mlm", "sap"}
    model = GlocalTextPathCMTPreTraining(config).to(device)
    model.train()

    batches = {
        "mlm": _move_batch(mlm_batch, device),
        "sap": _move_batch(sap_batch, device),
    }
    yield _PretrainContext(model=model, batches=batches, device=device)

    del model
    torch.cuda.empty_cache()


def _run_task_backward(context, task):
    model = context.model
    batch = context.batches[task]
    img_linear = model.bert.img_embeddings.img_linear
    img_linear_inputs = []
    img_linear_outputs = []
    def record_img_linear_dimensions(_module, inputs, output):
        img_linear_inputs.append(inputs[0].shape[-1])
        img_linear_outputs.append(output.shape[-1])

    hook = img_linear.register_forward_hook(record_img_linear_dimensions)
    try:
        _seed_pretrain_smoke(SMOKE_SEED + {"mlm": 1, "sap": 2}[task])
        model.zero_grad(set_to_none=True)
        losses = model(batch, task=task, compute_loss=True)
        loss = losses.mean()
        assert torch.isfinite(loss)
        loss.backward()
    finally:
        hook.remove()

    grad = img_linear.weight.grad
    assert img_linear_inputs == [768]
    assert img_linear_outputs == [768]
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
    assert not any(
        "dino" in name.lower()
        or "rgb_encoder" in name
        or "backbone" in name
        for name, _module in model.named_modules()
    )
    return loss.detach()


def _run_seeded_mlm_step(batch, device):
    _seed_pretrain_smoke()
    config = PretrainedConfig.from_json_file(MODEL_CONFIG)
    config.pretrain_tasks = {"mlm", "sap"}
    model = GlocalTextPathCMTPreTraining(config).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    _seed_pretrain_smoke(SMOKE_SEED + 1)
    losses = model(batch, task="mlm", compute_loss=True)
    losses.mean().backward()
    optimizer.step()
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.bert.img_embeddings.img_linear.state_dict().items()
    }

    del optimizer, model
    torch.cuda.empty_cache()
    return state


def test_real_mlm_batch_forward_backward_updates_img_linear(
    real_pretrain_context,
):
    loss = _run_task_backward(real_pretrain_context, "mlm")
    assert loss.item() > 0


def test_real_sap_batch_forward_backward_updates_img_linear(
    real_pretrain_context,
):
    loss = _run_task_backward(real_pretrain_context, "sap")
    assert loss.item() >= 0


def test_real_pretrain_one_step_writes_minimal_img_linear_checkpoint(
    real_pretrain_context, tmp_path
):
    model = real_pretrain_context.model
    img_linear = model.bert.img_embeddings.img_linear
    before = img_linear.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    _run_task_backward(real_pretrain_context, "mlm")
    optimizer.step()
    assert not torch.equal(img_linear.weight.detach(), before)

    output_root = Path(
        os.environ.get("ETPR1_RAE_SMOKE_PRETRAIN_DIR", str(tmp_path))
    )
    checkpoint_dir = output_root / "ckpts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_step_1.pt"
    checkpoint = {
        IMG_LINEAR_PREFIX + name: value.detach().cpu()
        for name, value in img_linear.state_dict().items()
    }
    torch.save(checkpoint, checkpoint_path)

    reloaded = torch.load(checkpoint_path, map_location="cpu")
    assert list(reloaded) == [
        IMG_LINEAR_PREFIX + name for name in img_linear.state_dict()
    ]
    for name, value in img_linear.state_dict().items():
        torch.testing.assert_close(
            reloaded[IMG_LINEAR_PREFIX + name],
            value.detach().cpu(),
            rtol=0,
            atol=0,
        )
    print(f"SMOKE_PRETRAIN_CHECKPOINT={checkpoint_path}")


def test_real_pretrain_fixture_explicitly_seeds_all_random_sources():
    source = inspect.getsource(real_pretrain_context.__wrapped__)

    assert "_seed_pretrain_smoke" in source
    assert source.count("_seed_pretrain_smoke") >= 3


def test_same_seed_reproduces_both_img_linear_parameters_after_one_step(
    real_pretrain_context,
):
    first = _run_seeded_mlm_step(
        real_pretrain_context.batches["mlm"],
        real_pretrain_context.device,
    )
    second = _run_seeded_mlm_step(
        real_pretrain_context.batches["mlm"],
        real_pretrain_context.device,
    )

    assert list(first) == list(second)
    assert len(first) == 2
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
