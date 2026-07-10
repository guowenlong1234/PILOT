import ast
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

import h5py
import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from pretrain_src.pretrain_src.data.dataset import (
    R2RTextPathData,
    ReverieTextPathData,
    _metadata_value_matches,
)
from pretrain_src.pretrain_src.model.vilmodel import ImageEmbeddings


ROOT = Path(__file__).resolve().parents[1]
RUN_PT = ROOT / "pretrain_src" / "run_pt"
RAE_FEATURE_FILE = (
    "pretrain_src/img_features/"
    "RAE-DINOv2-B-14-CLS-views-habitat.hdf5"
)
EXPECTED_RAE_METADATA = {
    "feature_extractor": "rae_dinov2_with_registers_base_cls",
    "feature_dim": 768,
    "dtype": "float32",
    "num_views": 36,
    "image_size": 224,
    "vfov": 60,
    "latent_normalized": True,
}


def _image_config(**overrides):
    values = {
        "rgb_encoder_type": "rae_dinov2",
        "raw_image_feat_size": 768,
        "image_feat_size": 512,
        "projection_hidden_size": 768,
        "hidden_size": 768,
        "depth_feat_size": 128,
        "angle_feat_size": 4,
        "obj_feat_size": 0,
        "hidden_dropout_prob": 0.0,
        "num_pano_layers": 0,
        "layer_norm_eps": 1e-5,
    }
    values.update(overrides)
    return PretrainedConfig(**values)


def _write_feature_hdf5(path, shape=(36, 768), metadata=None):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("scan_viewpoint", data=np.zeros(shape, np.float32))
        if metadata:
            for key, value in metadata.items():
                handle.attrs[key] = value


def _write_minimal_dataset_files(tmp_path, metadata=None):
    img_path = tmp_path / "rgb.hdf5"
    dep_path = tmp_path / "depth.hdf5"
    cands_path = tmp_path / "cands.json"
    connectivity_dir = tmp_path / "connectivity"
    anno_path = tmp_path / "annotations.jsonl"

    _write_feature_hdf5(img_path, metadata=metadata)
    _write_feature_hdf5(dep_path, shape=(36, 128))
    cands_path.write_text("{}", encoding="utf-8")
    connectivity_dir.mkdir()
    (connectivity_dir / "scans.txt").write_text("", encoding="utf-8")
    anno_path.write_text("", encoding="utf-8")
    return img_path, dep_path, cands_path, connectivity_dir, anno_path


def _make_uninitialized_dataset(dataset_cls, img_path, dep_path=None):
    dataset = object.__new__(dataset_cls)
    dataset.img_ft_file = str(img_path)
    dataset.dep_ft_file = None if dep_path is None else str(dep_path)
    dataset.raw_image_feat_size = 768
    dataset.image_feat_size = 512
    dataset.in_memory = False
    return dataset


def test_pretrain_projection_has_expected_layers_shape_and_gradients():
    module = ImageEmbeddings(_image_config())
    names = dict(module.named_parameters())
    linear_layers = [
        layer
        for layer in module.rgb_projection
        if isinstance(layer, torch.nn.Linear)
    ]

    assert "rgb_projection.0.weight" in names
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (768, 768),
        (768, 768),
        (768, 512),
    ]

    output = module.project_rgb(torch.randn(2, 36, 768))
    assert output.shape == (2, 36, 512)
    output.sum().backward()

    for name in ("rgb_projection.0.weight", "rgb_projection.4.weight"):
        gradient = names[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_forward_projects_rgb_once_before_img_linear():
    module = ImageEmbeddings(_image_config())
    projection_calls = []
    img_linear_inputs = []
    projection_hook = module.rgb_projection.register_forward_hook(
        lambda _module, inputs, output: projection_calls.append((inputs, output))
    )
    linear_hook = module.img_linear.register_forward_pre_hook(
        lambda _module, inputs: img_linear_inputs.append(inputs[0])
    )

    try:
        split_embeds, split_lens = module(
            traj_view_img_fts=torch.randn(2, 36, 768),
            traj_view_dep_fts=torch.randn(2, 36, 128),
            traj_obj_img_fts=None,
            traj_loc_fts=torch.randn(2, 36, 4),
            traj_nav_types=torch.zeros(2, 36, dtype=torch.long),
            traj_step_lens=[2],
            traj_vp_view_lens=torch.tensor([36, 36]),
            traj_vp_obj_lens=None,
            type_embed_layer=torch.nn.Embedding(2, 768),
        )
    finally:
        projection_hook.remove()
        linear_hook.remove()

    assert len(projection_calls) == 1
    assert len(img_linear_inputs) == 1
    assert img_linear_inputs[0].shape == (2, 36, 512)
    assert len(split_embeds) == len(split_lens) == 1


@pytest.mark.parametrize(
    ("features", "message"),
    (
        (torch.randn(2, 36, 767), "last dimension.*768.*767"),
        (
            torch.full((2, 36, 768), float("nan")),
            "NaN or infinity",
        ),
    ),
)
def test_project_rgb_rejects_invalid_features(features, message):
    with pytest.raises((ValueError, FloatingPointError), match=message):
        ImageEmbeddings(_image_config()).project_rgb(features)


def test_old_clip_config_uses_identity_and_preserves_tensor():
    config = _image_config()
    del config.rgb_encoder_type
    del config.raw_image_feat_size
    del config.projection_hidden_size
    module = ImageEmbeddings(config)
    features = torch.randn(2, 36, 512)

    assert isinstance(module.rgb_projection, torch.nn.Identity)
    assert module.project_rgb(features) is features
    assert module.img_linear(module.project_rgb(features)).shape == (2, 36, 768)


@pytest.mark.parametrize("dataset_cls", (ReverieTextPathData, R2RTextPathData))
@pytest.mark.parametrize(
    ("shape", "expected"),
    (
        ((36,), "two-dimensional"),
        ((35, 768), "36 views"),
        ((36, 767), "at least 768"),
    ),
)
def test_hdf5_feature_shape_is_validated_per_key(
    tmp_path,
    dataset_cls,
    shape,
    expected,
):
    img_path = tmp_path / "rgb.hdf5"
    dep_path = tmp_path / "depth.hdf5"
    _write_feature_hdf5(img_path, shape=shape)
    _write_feature_hdf5(dep_path, shape=(36, 128))
    dataset = _make_uninitialized_dataset(dataset_cls, img_path, dep_path)

    with pytest.raises(
        ValueError,
        match=rf"scan_viewpoint.*actual shape.*{expected}",
    ):
        dataset.get_scanvp_feature("scan", "viewpoint")


def test_r2r_outputs_raw_features_and_probabilities_start_after_raw_dimension():
    dataset = object.__new__(R2RTextPathData)
    dataset.data = [
        {
            "scan": "scan",
            "path": ["start"],
            "heading": 0.0,
            "instr_id": "instruction",
            "instr_encoding": [1, 2],
            "task_type_encoding": 1,
        }
    ]
    dataset.max_txt_len = 100
    dataset.raw_image_feat_size = 768
    dataset.image_feat_size = 512
    dataset.depth_feat_size = 128
    view_features = np.zeros((36, 770), dtype=np.float32)
    view_features[:, :768] = 3.0
    view_features[:, 768:] = np.array([1.0, 2.0], dtype=np.float32)

    dataset.get_cur_angle = MethodType(lambda self, *args: (0.0, 0.0), dataset)
    dataset.get_traj_pano_fts = MethodType(
        lambda self, *args: (
            [view_features],
            [np.zeros((36, 128), dtype=np.float32)],
            [np.zeros((36, 4), dtype=np.float32)],
            [[0] * 36],
            [[]],
            np.zeros((36, 2), dtype=np.float32),
        ),
        dataset,
    )
    dataset.get_gmap_inputs = MethodType(
        lambda self, *args: (
            [None],
            [0],
            [0],
            np.zeros((1, 7), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        ),
        dataset,
    )
    dataset.get_vp_pos_fts = MethodType(
        lambda self, *args: np.zeros((37, 14), dtype=np.float32),
        dataset,
    )

    output = dataset.get_input(0, "pos", return_img_probs=True)

    assert output["traj_view_img_fts"][0].shape == (36, 768)
    assert np.all(output["traj_view_img_fts"][0] == 3.0)
    expected_probs = np.tile(
        np.exp([1.0, 2.0]) / np.exp([1.0, 2.0]).sum(),
        (36, 1),
    )
    np.testing.assert_allclose(output["vp_view_probs"], expected_probs)


def test_raw_image_feature_size_defaults_to_legacy_image_size(tmp_path):
    img_path, dep_path, cands_path, connectivity_dir, anno_path = (
        _write_minimal_dataset_files(tmp_path)
    )

    dataset = R2RTextPathData(
        [str(anno_path)],
        str(img_path),
        str(dep_path),
        str(cands_path),
        str(connectivity_dir),
        image_feat_size=512,
    )

    assert dataset.rgb_encoder_type == "clip"
    assert dataset.raw_image_feat_size == dataset.image_feat_size == 512


@pytest.mark.parametrize(
    ("key", "bad_value"),
    (
        ("feature_extractor", "clip"),
        ("feature_dim", 512),
        ("dtype", "float16"),
        ("num_views", 35),
        ("image_size", 256),
        ("vfov", 90),
        ("latent_normalized", np.bool_(False)),
        ("latent_normalized", 1),
        ("feature_dim", 768.0),
        ("num_views", 36.0),
        ("image_size", 224.0),
        ("vfov", 60.0),
    ),
)
def test_rae_metadata_mismatch_fails_before_other_dataset_files_are_opened(
    tmp_path,
    key,
    bad_value,
):
    img_path = tmp_path / "rgb.hdf5"
    metadata = dict(EXPECTED_RAE_METADATA)
    metadata[key] = bad_value
    _write_feature_hdf5(img_path, metadata=metadata)

    with pytest.raises(
        ValueError,
        match=rf"metadata.*{key}.*expected.*got",
    ):
        R2RTextPathData(
            [str(tmp_path / "missing_annotations.jsonl")],
            str(img_path),
            str(tmp_path / "missing_depth.hdf5"),
            str(tmp_path / "missing_cands.json"),
            str(tmp_path / "missing_connectivity"),
            rgb_encoder_type="rae_dinov2",
            raw_image_feat_size=768,
            image_feat_size=512,
        )


def test_rae_metadata_accepts_normalized_structured_scalar_types(tmp_path):
    metadata = dict(EXPECTED_RAE_METADATA)
    metadata["feature_extractor"] = np.bytes_(
        "rae_dinov2_with_registers_base_cls"
    )
    metadata["feature_dim"] = np.int64(768)
    metadata["dtype"] = np.bytes_("float32")
    metadata["num_views"] = np.int64(36)
    metadata["image_size"] = np.int64(224)
    metadata["vfov"] = np.int64(60)
    metadata["latent_normalized"] = np.bool_(True)
    img_path, dep_path, cands_path, connectivity_dir, anno_path = (
        _write_minimal_dataset_files(tmp_path, metadata=metadata)
    )

    dataset = R2RTextPathData(
        [str(anno_path)],
        str(img_path),
        str(dep_path),
        str(cands_path),
        str(connectivity_dir),
        rgb_encoder_type="rae_dinov2",
        raw_image_feat_size=768,
        image_feat_size=512,
    )

    assert dataset.raw_image_feat_size == 768


@pytest.mark.parametrize(
    ("got", "expected"),
    (
        (True, 1),
        (1, True),
    ),
)
def test_metadata_comparison_does_not_mix_boolean_and_integer_types(
    got,
    expected,
):
    assert not _metadata_value_matches(got, expected)


def test_train_r2r_passes_encoder_type_and_raw_size_to_all_datasets():
    source_path = ROOT / "pretrain_src" / "pretrain_src" / "train_r2r.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "R2RTextPathData"
    ]

    assert len(calls) == 3
    for call in calls:
        keyword_names = {keyword.arg for keyword in call.keywords}
        assert "rgb_encoder_type" in keyword_names
        assert "raw_image_feat_size" in keyword_names


def test_pretrain_entrypoint_imports_shared_projection_from_clean_python_path():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "pretrain_src" / "pretrain_src" / "train_r2r.py"),
            "--help",
        ],
        cwd=ROOT / "pretrain_src" / "pretrain_src",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    assert "--model_config" in result.stdout


def test_rae_model_config_has_explicit_projection_contract():
    config = json.loads(
        (RUN_PT / "mix_model_config_rae_dino.json").read_text(encoding="utf-8")
    )

    assert config["rgb_encoder_type"] == "rae_dinov2"
    assert config["raw_image_feat_size"] == 768
    assert config["image_feat_size"] == 512
    assert config["projection_hidden_size"] == 768
    assert config["image_prob_size"] == 0


def test_rae_pretrain_config_only_replaces_rgb_feature_file():
    clip_config = json.loads(
        (RUN_PT / "mix_pretrain_server.json").read_text(encoding="utf-8")
    )
    rae_config = json.loads(
        (RUN_PT / "mix_pretrain_rae_dino.json").read_text(encoding="utf-8")
    )
    expected = copy.deepcopy(clip_config)
    expected["train_datasets"]["R2R"]["img_ft_file"] = RAE_FEATURE_FILE

    assert rae_config == expected
    dataset = rae_config["train_datasets"]["R2R"]
    assert len(dataset["train_traj_files"]) == 5
    assert len(dataset["val_unseen_r2r_traj_files"]) == 1
    assert len(dataset["val_unseen_rxr_traj_files"]) == 1
    assert dataset["tasks"] == ["mlm", "sap"]


def test_rae_launch_script_is_single_gpu_and_uses_isolated_output():
    script = (RUN_PT / "run_mix_rae_dino.bash").read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "NUM_GPUS=1" in script
    assert "torchrun" in script
    assert "--nproc_per_node=1" in script
    assert "--world_size 1" in script
    assert "mix_model_config_rae_dino.json" in script
    assert "mix_pretrain_rae_dino.json" in script
    assert "pretrained/r2r_rxr_ce/rae_dinov2_cls_mlp" in script
    assert "master_port" in script.lower()
    assert "${1:" in script


def test_rae_launch_script_uses_runtime_wrapper_from_nested_workdir(tmp_path):
    capture_path = tmp_path / "torchrun-call.txt"
    fake_torchrun = tmp_path / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "{\n"
        "  printf 'pwd=%s\\n' \"$PWD\"\n"
        "  printf 'runtime=%s\\n' \"${ETPR1_RUNTIME_ACTIVE-}\"\n"
        "  printf 'pythonpath=%s\\n' \"${PYTHONPATH-}\"\n"
        "  printf 'arg=%s\\n' \"$@\"\n"
        "} > \"$CAPTURE_PATH\"\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("ETPR1_RUNTIME_ACTIVE", None)
    environment["LD_LIBRARY_PATH"] = ":".join(
        (
            str(ROOT / ".runtime" / "etpr1_habitat" / "prefix" / "lib"),
            "/usr/local/nvidia/lib",
            "/usr/local/nvidia/lib64",
        )
    )
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"
    environment["CAPTURE_PATH"] = str(capture_path)

    result = subprocess.run(
        ["bash", str(RUN_PT / "run_mix_rae_dino.bash"), "23456"],
        cwd=ROOT / "pretrain_src" / "pretrain_src",
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    assert result.returncode == 0, result.stdout
    captured = capture_path.read_text(encoding="utf-8")
    assert f"pwd={ROOT}" in captured
    assert "runtime=1" in captured
    assert f"pythonpath={ROOT}:" in captured
    assert "arg=pretrain_src/pretrain_src/train_r2r.py" in captured
    assert "arg=--nproc_per_node=1" in captured


def test_base_and_r2r_dataset_expose_raw_image_feature_size():
    for dataset_cls in (ReverieTextPathData, R2RTextPathData):
        parameters = inspect.signature(dataset_cls.__init__).parameters
        assert "raw_image_feat_size" in parameters
        assert "rgb_encoder_type" in parameters
