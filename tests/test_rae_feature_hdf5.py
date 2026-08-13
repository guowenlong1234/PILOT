import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from precompute_img_features import extract_rae_dinov2_features as extractor_module
from precompute_img_features.extract_rae_dinov2_features import (
    DEFAULT_CONNECTIVITY_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_FILE,
    ViewpointRecord,
    build_metadata,
    build_parser as build_extract_parser,
    build_simulator,
    encode_views,
    load_connectivity_viewpoints,
    render_36_views,
    view_index_to_rotation_quat,
    write_feature_file,
)
from precompute_img_features.validate_rae_dinov2_features import (
    DEFAULT_CLIP_FEATURES,
    build_parser as build_validate_parser,
    validate_feature_file,
)


EXPECTED_METADATA = {
    "feature_extractor": "rae_dinov2_with_registers_base_raw_cls",
    "feature_dim": 768,
    "dtype": "float32",
    "num_views": 36,
    "image_size": 224,
    "vfov": 60,
    "sensor_height": 0.0,
    "camera_geometry": "mp3d_viewpoint_center_zero_sensor_offset",
    "cls_normalization": "none",
    "rae_stat_applied_to_cls": False,
    "dino_weights_sha256": "a" * 64,
    "preprocess_version": "etpnav_rae_navigation_cls_v2_fixed_camera_center",
}


def _quat_coeffs(quat):
    return np.asarray([quat.x, quat.y, quat.z, quat.w], dtype=np.float64)


def test_view_index_covers_three_elevations_and_twelve_headings_in_clip_order():
    from habitat_sim.utils.common import quat_from_angle_axis

    rotations = [view_index_to_rotation_quat(index) for index in range(36)]
    rounded = {tuple(np.round(_quat_coeffs(quat), 8)) for quat in rotations}

    assert len(rounded) == 36
    for index in (0, 11, 12, 23, 24, 35):
        heading = -np.deg2rad((index % 12) * 30.0)
        elevation = np.deg2rad((index // 12 - 1) * 30.0)
        expected = quat_from_angle_axis(heading, np.array([0.0, 1.0, 0.0]))
        expected *= quat_from_angle_axis(elevation, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(
            _quat_coeffs(rotations[index]),
            _quat_coeffs(expected),
            atol=1e-7,
        )


@pytest.mark.parametrize("index", (-1, 36, 1.5, True, "0"))
def test_view_index_rejects_invalid_values(index):
    with pytest.raises((TypeError, ValueError), match="view index"):
        view_index_to_rotation_quat(index)


class FakeAgent:
    def __init__(self):
        self.state = SimpleNamespace(position=None, rotation=None)
        self.set_states = []

    def get_state(self):
        return self.state

    def set_state(self, state, reset_sensors=True):
        self.set_states.append(
            (
                np.asarray(state.position).copy(),
                _quat_coeffs(state.rotation),
                reset_sensors,
            )
        )


class FakeSimulator:
    def __init__(self, pixel=(255, 0, 0, 17), image_size=2):
        self.agent = FakeAgent()
        self.pixel = np.asarray(pixel, dtype=np.uint8)
        self.image_size = image_size
        self.observation_count = 0
        self.closed = False

    def get_agent(self, index):
        assert index == 0
        return self.agent

    def get_sensor_observations(self):
        self.observation_count += 1
        rgb = np.empty(
            (self.image_size, self.image_size, len(self.pixel)), dtype=np.uint8
        )
        rgb[...] = self.pixel
        return {"rgb": rgb}

    def close(self):
        self.closed = True


def test_render_36_views_sets_absolute_states_and_keeps_rgb_channel_order():
    simulator = FakeSimulator(pixel=(255, 0, 0, 99))
    position = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    rendered = render_36_views(simulator, position, image_size=2)

    assert rendered.shape == (36, 2, 2, 3)
    assert rendered.dtype == np.uint8
    assert rendered[0, 0, 0].tolist() == [255, 0, 0]
    assert simulator.observation_count == 36
    assert len(simulator.agent.set_states) == 36
    assert all(state[2] is True for state in simulator.agent.set_states)
    assert all(np.array_equal(state[0], position) for state in simulator.agent.set_states)
    assert len({tuple(np.round(state[1], 8)) for state in simulator.agent.set_states}) == 36


def test_simulator_places_sensor_at_agent_camera_center(monkeypatch):
    captured = {}

    class FakeSimulatorConfiguration:
        def __init__(self):
            self.scene_id = None
            self.gpu_device_id = None

    class FakeCameraSensorSpec:
        def __init__(self):
            self.uuid = None
            self.sensor_type = None
            self.resolution = None
            self.hfov = None
            self.position = None

    class FakeAgentConfiguration:
        def __init__(self):
            self.sensor_specifications = None

    def fake_configuration(simulator_config, agent_configs):
        captured["simulator_config"] = simulator_config
        captured["agent_configs"] = agent_configs
        return "configuration"

    fake_habitat_sim = SimpleNamespace(
        SimulatorConfiguration=FakeSimulatorConfiguration,
        CameraSensorSpec=FakeCameraSensorSpec,
        SensorType=SimpleNamespace(COLOR="color"),
        agent=SimpleNamespace(AgentConfiguration=FakeAgentConfiguration),
        Configuration=fake_configuration,
        Simulator=lambda configuration: configuration,
    )
    monkeypatch.setitem(sys.modules, "habitat_sim", fake_habitat_sim)

    simulator = build_simulator(
        "scene.glb",
        image_size=224,
        hfov=60,
        sim_gpu_id=3,
    )

    assert simulator == "configuration"
    assert captured["simulator_config"].scene_id == "scene.glb"
    assert captured["simulator_config"].gpu_device_id == 3
    sensor_spec = captured["agent_configs"][0].sensor_specifications[0]
    assert sensor_spec.position == [0.0, 0.0, 0.0]


def test_connectivity_parser_filters_transforms_deduplicates_and_sorts(tmp_path):
    connectivity = tmp_path / "connectivity"
    connectivity.mkdir()
    (connectivity / "scans.txt").write_text("scan_b\nscan_a\nscan_b\n", encoding="utf-8")
    (connectivity / "scan_a_connectivity.json").write_text(
        json.dumps(
            [
                {"image_id": "vp_2", "included": False, "pose": list(range(12))},
                {
                    "image_id": "vp_1",
                    "included": True,
                    "pose": [0, 0, 0, 1.0, 0, 0, 0, 2.0, 0, 0, 0, 3.0],
                },
            ]
        ),
        encoding="utf-8",
    )
    (connectivity / "scan_b_connectivity.json").write_text(
        json.dumps(
            [
                {
                    "image_id": "vp_2",
                    "included": True,
                    "pose": [0, 0, 0, 4.0, 0, 0, 0, 5.0, 0, 0, 0, 6.0],
                },
                {
                    "image_id": "vp_2",
                    "included": True,
                    "pose": [0, 0, 0, 4.0, 0, 0, 0, 5.0, 0, 0, 0, 6.0],
                },
            ]
        ),
        encoding="utf-8",
    )

    records = load_connectivity_viewpoints(connectivity)

    assert [record.key for record in records] == ["scan_a_vp_1", "scan_b_vp_2"]
    np.testing.assert_allclose(records[0].position, [1.0, 3.0, -2.0])
    np.testing.assert_allclose(records[1].position, [4.0, 6.0, -5.0])


class FakeEncoder(torch.nn.Module):
    output_size = 768

    def __init__(self, fail_on_call=None, output_factory=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []
        self.fail_on_call = fail_on_call
        self.output_factory = output_factory

    def forward(self, observations):
        rgb = observations["rgb"]
        self.calls.append(rgb.detach().cpu().clone())
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("planned interruption")
        if self.output_factory is not None:
            return self.output_factory(rgb)
        values = rgb[:, 0, 0, 0].float().unsqueeze(1) + 1.0
        return values.repeat(1, self.output_size)


def test_encode_views_batches_and_returns_frozen_float32_features():
    encoder = FakeEncoder()
    encoder.train(True)
    views = np.zeros((36, 2, 2, 3), dtype=np.uint8)
    views[:, 0, 0, 0] = np.arange(36, dtype=np.uint8)

    features = encode_views(encoder, views, torch.device("cpu"), batch_size=10)

    assert features.shape == (36, 768)
    assert features.dtype == np.float32
    assert [len(batch) for batch in encoder.calls] == [10, 10, 10, 6]
    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    np.testing.assert_array_equal(features[:, 0], np.arange(1, 37, dtype=np.float32))


@pytest.mark.parametrize(
    ("output_factory", "message"),
    (
        (lambda rgb: torch.ones(len(rgb), 767), "shape"),
        (lambda rgb: torch.full((len(rgb), 768), float("nan")), "finite"),
        (lambda rgb: torch.zeros(len(rgb), 768), "all zero"),
    ),
)
def test_encode_views_rejects_invalid_features(output_factory, message):
    with pytest.raises(ValueError, match=message):
        encode_views(
            FakeEncoder(output_factory=output_factory),
            np.zeros((36, 2, 2, 3), dtype=np.uint8),
            torch.device("cpu"),
            batch_size=36,
        )


def _records(count=5):
    return [
        ViewpointRecord(
            scan_id="scan_a",
            viewpoint_id=f"vp_{index}",
            position=np.asarray([index, 0.0, 0.0], dtype=np.float32),
        )
        for index in range(count)
    ]


def _simulator_factory(created, pixel=(3, 2, 1, 255)):
    def factory(scan_id):
        simulator = FakeSimulator(pixel=pixel)
        created.append((scan_id, simulator))
        return simulator

    return factory


def _write_metadata(handle, metadata=EXPECTED_METADATA):
    for key, value in metadata.items():
        handle.attrs[key] = value


def test_writer_resume_skips_complete_key_and_recomputes_all_corrupt_keys(tmp_path):
    output = tmp_path / "features.hdf5"
    records = _records()
    with h5py.File(output, "w") as handle:
        _write_metadata(handle)
        handle.create_dataset(
            records[0].key,
            data=np.full((36, 768), 7, np.float32),
            compression="gzip",
        )
        handle.create_dataset(records[1].key, data=np.ones((35, 768), np.float32))
        handle.create_dataset(records[2].key, data=np.ones((36, 768), np.float64))
        nan_data = np.ones((36, 768), np.float32)
        nan_data[0, 0] = np.nan
        handle.create_dataset(records[3].key, data=nan_data)
        handle.create_dataset(records[4].key, data=np.zeros((36, 768), np.float32))

    created = []
    summary = write_feature_file(
        output,
        records,
        FakeEncoder(),
        _simulator_factory(created),
        metadata=EXPECTED_METADATA,
        device=torch.device("cpu"),
        batch_size=12,
        image_size=2,
    )

    assert summary == {"total": 5, "skipped": 1, "recomputed": 4, "completed": 4}
    assert len(created) == 1
    assert created[0][1].closed is True
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle[records[0].key][...], 7.0)
        for record in records:
            assert handle[record.key].shape == (36, 768)
            assert handle[record.key].dtype == np.dtype(np.float32)
            assert handle[record.key].compression == "gzip"
            assert np.isfinite(handle[record.key][...]).all()
            assert np.any(handle[record.key][...] != 0)


def test_writer_closes_resources_after_interruption_and_can_resume(tmp_path):
    output = tmp_path / "features.hdf5"
    records = _records(count=2)
    first_created = []
    with pytest.raises(RuntimeError, match="planned interruption"):
        write_feature_file(
            output,
            records,
            FakeEncoder(fail_on_call=2),
            _simulator_factory(first_created),
            metadata=EXPECTED_METADATA,
            device=torch.device("cpu"),
            batch_size=36,
            image_size=2,
        )
    assert first_created[0][1].closed is True
    with h5py.File(output, "r") as handle:
        first_value = handle[records[0].key][...].copy()
        assert records[1].key not in handle

    second_created = []
    summary = write_feature_file(
        output,
        records,
        FakeEncoder(),
        _simulator_factory(second_created),
        metadata=EXPECTED_METADATA,
        device=torch.device("cpu"),
        batch_size=36,
        image_size=2,
    )

    assert summary["skipped"] == 1
    assert summary["completed"] == 1
    with h5py.File(output, "r") as handle:
        np.testing.assert_array_equal(handle[records[0].key][...], first_value)
        assert records[1].key in handle


def test_writer_rejects_existing_metadata_mismatch_before_rendering(tmp_path):
    output = tmp_path / "features.hdf5"
    with h5py.File(output, "w") as handle:
        _write_metadata(handle, replace_metadata := dict(EXPECTED_METADATA))
    replace_metadata["vfov"] = 90
    created = []

    with pytest.raises(ValueError, match=r"metadata mismatch.*vfov"):
        write_feature_file(
            output,
            _records(1),
            FakeEncoder(),
            _simulator_factory(created),
            metadata=replace_metadata,
            device=torch.device("cpu"),
            image_size=2,
        )

    assert created == []


def test_writer_rejects_legacy_offset_camera_hdf5_before_rendering(tmp_path):
    output = tmp_path / "legacy_features.hdf5"
    legacy_metadata = dict(EXPECTED_METADATA)
    legacy_metadata.pop("camera_geometry")
    legacy_metadata["sensor_height"] = 1.25
    legacy_metadata["preprocess_version"] = "etpnav_rae_navigation_cls_v1"
    with h5py.File(output, "w") as handle:
        _write_metadata(handle, legacy_metadata)
        handle.create_dataset(
            "scan_a_vp_0",
            data=np.ones((36, 768), dtype=np.float32),
        )
    created = []

    legacy_geometry_keys = r"(?:sensor_height|camera_geometry|preprocess_version)"
    with pytest.raises(
        ValueError,
        match=rf"metadata mismatch.*{legacy_geometry_keys}",
    ):
        write_feature_file(
            output,
            _records(1),
            FakeEncoder(),
            _simulator_factory(created),
            metadata=EXPECTED_METADATA,
            device=torch.device("cpu"),
            image_size=2,
        )

    assert created == []


@pytest.mark.parametrize(
    "empty_file_kind",
    ("zero_bytes", "empty_hdf5", "matching_metadata_subset"),
)
def test_writer_recovers_empty_file_with_missing_metadata(tmp_path, empty_file_kind):
    output = tmp_path / "features.hdf5"
    if empty_file_kind == "zero_bytes":
        output.write_bytes(b"")
    else:
        with h5py.File(output, "w") as handle:
            if empty_file_kind == "matching_metadata_subset":
                handle.attrs["feature_extractor"] = EXPECTED_METADATA[
                    "feature_extractor"
                ]
                handle.attrs["feature_dim"] = EXPECTED_METADATA["feature_dim"]

    record = _records(1)[0]
    summary = write_feature_file(
        output,
        [record],
        FakeEncoder(),
        _simulator_factory([]),
        metadata=EXPECTED_METADATA,
        device=torch.device("cpu"),
        image_size=2,
    )

    assert summary["completed"] == 1
    with h5py.File(output, "r") as handle:
        assert set(handle.attrs) == set(EXPECTED_METADATA)
        for key, expected in EXPECTED_METADATA.items():
            actual = handle.attrs[key]
            if isinstance(actual, bytes):
                actual = actual.decode("utf-8")
            if isinstance(actual, np.generic):
                actual = actual.item()
            assert type(actual) is type(expected)
            assert actual == expected
        assert isinstance(handle[record.key], h5py.Dataset)


def test_writer_rejects_conflicting_partial_metadata_without_mutation(tmp_path):
    output = tmp_path / "features.hdf5"
    with h5py.File(output, "w") as handle:
        handle.attrs["feature_extractor"] = EXPECTED_METADATA["feature_extractor"]
        handle.attrs["vfov"] = 90

    with pytest.raises(ValueError, match=r"metadata mismatch.*vfov"):
        write_feature_file(
            output,
            _records(1),
            FakeEncoder(),
            _simulator_factory([]),
            metadata=EXPECTED_METADATA,
            device=torch.device("cpu"),
            image_size=2,
        )

    with h5py.File(output, "r") as handle:
        assert dict(handle.attrs)["vfov"] == 90
        assert len(handle) == 0


def test_writer_rejects_unknown_partial_metadata_without_mutation(tmp_path):
    output = tmp_path / "features.hdf5"
    with h5py.File(output, "w") as handle:
        handle.attrs["feature_dim"] = EXPECTED_METADATA["feature_dim"]
        handle.attrs["unknown_semantics"] = "keep-me"

    with pytest.raises(ValueError, match=r"unexpected metadata.*unknown_semantics"):
        write_feature_file(
            output,
            _records(1),
            FakeEncoder(),
            _simulator_factory([]),
            metadata=EXPECTED_METADATA,
            device=torch.device("cpu"),
            image_size=2,
        )

    with h5py.File(output, "r") as handle:
        assert handle.attrs["unknown_semantics"] == "keep-me"
        assert len(handle) == 0


@pytest.mark.parametrize("root_entry_kind", ("dataset", "group", "soft_link"))
def test_writer_rejects_incomplete_metadata_when_root_contains_entries(
    tmp_path,
    root_entry_kind,
):
    output = tmp_path / "features.hdf5"
    sentinel = np.asarray([3.0, 4.0], dtype=np.float32)
    with h5py.File(output, "w") as handle:
        handle.attrs["feature_dim"] = EXPECTED_METADATA["feature_dim"]
        if root_entry_kind == "dataset":
            handle.create_dataset("sentinel", data=sentinel)
        elif root_entry_kind == "group":
            handle.create_group("sentinel")
        else:
            handle["sentinel"] = h5py.SoftLink("/missing_target")

    with pytest.raises(ValueError, match=r"incomplete metadata.*root entries"):
        write_feature_file(
            output,
            _records(1),
            FakeEncoder(),
            _simulator_factory([]),
            metadata=EXPECTED_METADATA,
            device=torch.device("cpu"),
            image_size=2,
        )

    with h5py.File(output, "r") as handle:
        assert handle.attrs["feature_dim"] == 768
        if root_entry_kind == "dataset":
            np.testing.assert_array_equal(handle["sentinel"][...], sentinel)
        elif root_entry_kind == "group":
            assert isinstance(handle["sentinel"], h5py.Group)
        else:
            assert isinstance(handle.get("sentinel", getlink=True), h5py.SoftLink)


def test_writer_reuses_one_simulator_per_scan_and_closes_on_scan_change(tmp_path):
    records = _records(2) + [
        replace(_records(1)[0], scan_id="scan_b", viewpoint_id="vp_b")
    ]
    created = []

    write_feature_file(
        tmp_path / "features.hdf5",
        records,
        FakeEncoder(),
        _simulator_factory(created),
        metadata=EXPECTED_METADATA,
        device=torch.device("cpu"),
        image_size=2,
    )

    assert [scan for scan, _ in created] == ["scan_a", "scan_b"]
    assert all(simulator.closed for _, simulator in created)


@pytest.mark.parametrize(
    "entry_kind",
    ("group", "soft_link", "external_link", "dangling_soft_link"),
)
def test_writer_recomputes_non_hardlink_and_non_dataset_entries(
    tmp_path,
    entry_kind,
):
    output = tmp_path / "features.hdf5"
    external = tmp_path / "external.hdf5"
    record = _records(1)[0]
    valid = np.full((36, 768), 9, dtype=np.float32)
    with h5py.File(external, "w") as handle:
        handle.create_dataset("valid", data=valid, compression="gzip")
    with h5py.File(output, "w") as handle:
        _write_metadata(handle)
        if entry_kind == "group":
            handle.create_group(record.key)
        elif entry_kind == "soft_link":
            handle.create_dataset("soft_target", data=valid, compression="gzip")
            handle[record.key] = h5py.SoftLink("/soft_target")
        elif entry_kind == "external_link":
            handle[record.key] = h5py.ExternalLink(str(external), "/valid")
        else:
            handle[record.key] = h5py.SoftLink("/missing_target")

    summary = write_feature_file(
        output,
        [record],
        FakeEncoder(),
        _simulator_factory([]),
        metadata=EXPECTED_METADATA,
        device=torch.device("cpu"),
        image_size=2,
    )

    assert summary == {"total": 1, "skipped": 0, "recomputed": 1, "completed": 1}
    with h5py.File(output, "r") as handle:
        assert isinstance(handle.get(record.key, getlink=True), h5py.HardLink)
        dataset = handle.get(record.key)
        assert isinstance(dataset, h5py.Dataset)
        assert dataset.shape == (36, 768)
        assert dataset.dtype == np.dtype(np.float32)
        assert dataset.compression == "gzip"
        assert np.isfinite(dataset[...]).all()
        assert np.any(dataset[...] != 0)


def test_writer_removes_soft_link_target_outside_default_allowed_keys(tmp_path):
    output = tmp_path / "features.hdf5"
    record = _records(1)[0]
    valid = np.full((36, 768), 9, dtype=np.float32)
    with h5py.File(output, "w") as handle:
        _write_metadata(handle)
        handle.create_dataset("soft_target", data=valid, compression="gzip")
        handle[record.key] = h5py.SoftLink("/soft_target")

    write_feature_file(
        output,
        [record],
        FakeEncoder(),
        _simulator_factory([]),
        metadata=EXPECTED_METADATA,
        device=torch.device("cpu"),
        image_size=2,
    )
    summary = validate_feature_file(
        output,
        connectivity_keys=[record.key],
        expected_count=1,
        expected_metadata=EXPECTED_METADATA,
    )

    assert summary["valid"] is True
    assert summary["extra_key_count"] == 0


def test_writer_preserves_allowed_key_outside_selected_viewpoints(tmp_path):
    output = tmp_path / "features.hdf5"
    selected, preserved = _records(2)
    preserved_values = np.full((36, 768), 13, dtype=np.float32)
    with h5py.File(output, "w") as handle:
        _write_metadata(handle)
        handle.create_group(selected.key)
        handle.create_dataset(
            preserved.key,
            data=preserved_values,
            compression="gzip",
        )
        handle.create_dataset("unrelated", data=np.ones(1, dtype=np.float32))

    summary = write_feature_file(
        output,
        [selected],
        FakeEncoder(),
        _simulator_factory([]),
        metadata=EXPECTED_METADATA,
        device=torch.device("cpu"),
        image_size=2,
        allowed_keys={selected.key, preserved.key},
    )

    assert summary == {"total": 1, "skipped": 0, "recomputed": 1, "completed": 1}
    with h5py.File(output, "r") as handle:
        assert set(handle.keys()) == {selected.key, preserved.key}
        np.testing.assert_array_equal(handle[preserved.key][...], preserved_values)
        assert isinstance(handle[selected.key], h5py.Dataset)


def test_main_keeps_all_allowed_keys_when_max_viewpoints_limits_work(monkeypatch):
    records = _records(2)
    captured = {}
    monkeypatch.setattr(
        extractor_module,
        "load_connectivity_viewpoints",
        lambda *args, **kwargs: records,
    )
    monkeypatch.setattr(
        extractor_module,
        "build_metadata",
        lambda *args, **kwargs: EXPECTED_METADATA,
    )
    monkeypatch.setattr(
        "vlnce_baselines.models.encoders.rae_dinov2_encoder.RaeDinov2ClsEncoder",
        lambda *args, **kwargs: FakeEncoder(),
    )

    def fake_write(output_file, viewpoints, *args, allowed_keys, **kwargs):
        captured["viewpoints"] = list(viewpoints)
        captured["allowed_keys"] = set(allowed_keys)
        return {"total": 1, "skipped": 0, "recomputed": 0, "completed": 1}

    monkeypatch.setattr(extractor_module, "write_feature_file", fake_write)

    assert extractor_module.main(["--device", "cpu", "--max_viewpoints", "1"]) == 0
    assert [record.key for record in captured["viewpoints"]] == [records[0].key]
    assert captured["allowed_keys"] == {record.key for record in records}


def test_build_metadata_hashes_model_and_declares_raw_cls(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"model")
    metadata = build_metadata(model_dir)

    assert metadata["dino_weights_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert metadata["cls_normalization"] == "none"
    assert metadata["rae_stat_applied_to_cls"] is False
    assert "rae_stat_sha256" not in metadata


def _write_valid_fixture(path, keys, metadata=EXPECTED_METADATA):
    with h5py.File(path, "w") as handle:
        _write_metadata(handle, metadata)
        for index, key in enumerate(keys, start=1):
            handle.create_dataset(
                key,
                data=np.full((36, 768), index, dtype=np.float32),
                compression="gzip",
            )


def test_validator_accepts_small_fixture_and_returns_json_summary(tmp_path):
    keys = ["scan_a_vp_1", "scan_b_vp_2"]
    features = tmp_path / "rae.hdf5"
    clip = tmp_path / "clip.hdf5"
    _write_valid_fixture(features, keys)
    with h5py.File(clip, "w") as handle:
        for key in keys:
            handle.create_dataset(key, data=np.ones((36, 512), np.float32))

    summary = validate_feature_file(
        features,
        connectivity_keys=keys,
        clip_features=clip,
        expected_count=2,
        expected_metadata=EXPECTED_METADATA,
    )

    assert summary["valid"] is True
    assert summary["error_count"] == 0
    assert summary["checked_datasets"] == 2
    json.dumps(summary)


def test_validator_counts_all_errors_but_limits_error_examples(tmp_path):
    keys = ["scan_a_vp_1", "scan_a_vp_2", "scan_a_vp_3", "scan_a_vp_4"]
    features = tmp_path / "bad.hdf5"
    bad_metadata = dict(EXPECTED_METADATA)
    bad_metadata["feature_dim"] = np.float64(768)
    with h5py.File(features, "w") as handle:
        _write_metadata(handle, bad_metadata)
        handle.create_dataset(keys[0], data=np.ones((35, 768), np.float32))
        handle.create_dataset(keys[1], data=np.ones((36, 768), np.float64))
        nan_data = np.ones((36, 768), np.float32)
        nan_data[0, 0] = np.nan
        handle.create_dataset(keys[2], data=nan_data)
        handle.create_group(keys[3])

    summary = validate_feature_file(
        features,
        connectivity_keys=keys,
        expected_count=4,
        expected_metadata=EXPECTED_METADATA,
        max_error_examples=2,
    )

    assert summary["valid"] is False
    assert summary["error_count"] >= 5
    assert len(summary["errors"]) == 2
    assert any("actual shape" in error for error in summary["errors"])
    json.dumps(summary)


def test_validator_rejects_key_mismatches_nonfinite_zero_and_bad_hash(tmp_path):
    features = tmp_path / "bad.hdf5"
    metadata = dict(EXPECTED_METADATA)
    metadata["dino_weights_sha256"] = "not-a-hash"
    with h5py.File(features, "w") as handle:
        _write_metadata(handle, metadata)
        handle.create_dataset("extra", data=np.zeros((36, 768), np.float32))

    summary = validate_feature_file(
        features,
        connectivity_keys=["missing"],
        expected_count=1,
        expected_metadata=EXPECTED_METADATA,
    )

    assert summary["valid"] is False
    assert summary["missing_key_count"] == 1
    assert summary["extra_key_count"] == 1
    assert any("sha256" in error for error in summary["errors"])
    assert any("all zero" in error for error in summary["errors"])


def test_validator_isolates_string_dtype_and_continues_to_later_bad_key(tmp_path):
    features = tmp_path / "bad_types.hdf5"
    string_key = "scan_a_string"
    zero_key = "scan_b_zero"
    with h5py.File(features, "w") as handle:
        _write_metadata(handle)
        handle.create_dataset(
            string_key,
            data=np.full((36, 768), b"x", dtype="S1"),
        )
        handle.create_dataset(
            zero_key,
            data=np.zeros((36, 768), dtype=np.float32),
        )

    summary = validate_feature_file(
        features,
        connectivity_keys=[string_key, zero_key],
        expected_count=2,
        expected_metadata=EXPECTED_METADATA,
    )

    assert summary["valid"] is False
    assert summary["checked_datasets"] == 2
    assert summary["error_count"] >= 3
    assert any(string_key in error and "dtype" in error for error in summary["errors"])
    assert any(
        string_key in error and "non-numeric" in error for error in summary["errors"]
    )
    assert any(zero_key in error and "all zero" in error for error in summary["errors"])


def test_cli_defaults_and_key_parameters():
    extract = build_extract_parser().parse_args([])
    validate = build_validate_parser().parse_args([])

    assert extract.model_dir == DEFAULT_MODEL_DIR
    assert not hasattr(extract, "stat_path")
    assert extract.connectivity_dir == DEFAULT_CONNECTIVITY_DIR
    assert extract.output_file == DEFAULT_OUTPUT_FILE
    assert extract.max_viewpoints == -1
    assert extract.image_size == 224
    assert extract.vfov == 60
    assert not hasattr(extract, "sensor_height")
    assert validate.features == DEFAULT_OUTPUT_FILE
    assert validate.connectivity == DEFAULT_CONNECTIVITY_DIR
    assert validate.clip_features == DEFAULT_CLIP_FEATURES
    assert validate.expected_count == 10567


def test_validator_cli_invalid_argument_is_single_json_error():
    script = (
        Path(__file__).parents[1]
        / "precompute_img_features"
        / "validate_rae_dinov2_features.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--expected_count", "nope"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["valid"] is False
    assert summary["error_count"] == 1
    assert len(summary["errors"]) == 1
    assert "expected_count" in summary["errors"][0]


def test_validator_cli_help_remains_normal_text():
    script = (
        Path(__file__).parents[1]
        / "precompute_img_features"
        / "validate_rae_dinov2_features.py"
    )

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert result.stdout.startswith("usage:")
    assert "--expected_count" in result.stdout
