import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from precompute_img_features.extract_rae_dinov2_features import (
    DEFAULT_CONNECTIVITY_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_FILE,
    ViewpointRecord,
    build_metadata,
    build_parser as build_extract_parser,
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
    "feature_extractor": "rae_dinov2_with_registers_base_cls",
    "feature_dim": 768,
    "dtype": "float32",
    "num_views": 36,
    "image_size": 224,
    "vfov": 60,
    "sensor_height": 1.25,
    "latent_normalized": True,
    "dino_weights_sha256": "a" * 64,
    "rae_stat_sha256": "b" * 64,
    "preprocess_version": "rae_native_224_rgb_v1",
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

    records = load_connectivity_viewpoints(connectivity, sensor_height=1.25)

    assert [record.key for record in records] == ["scan_a_vp_1", "scan_b_vp_2"]
    np.testing.assert_allclose(records[0].position, [1.0, 1.75, -2.0])
    np.testing.assert_allclose(records[1].position, [4.0, 4.75, -5.0])


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


def test_build_metadata_hashes_local_model_and_stat(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"model")
    stat = model_dir / "stat.pt"
    stat.write_bytes(b"stat")

    metadata = build_metadata(model_dir, stat)

    assert metadata["dino_weights_sha256"] == hashlib.sha256(b"model").hexdigest()
    assert metadata["rae_stat_sha256"] == hashlib.sha256(b"stat").hexdigest()


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


def test_cli_defaults_and_key_parameters():
    extract = build_extract_parser().parse_args([])
    validate = build_validate_parser().parse_args([])

    assert extract.model_dir == DEFAULT_MODEL_DIR
    assert extract.stat_path == f"{DEFAULT_MODEL_DIR}/stat.pt"
    assert extract.connectivity_dir == DEFAULT_CONNECTIVITY_DIR
    assert extract.output_file == DEFAULT_OUTPUT_FILE
    assert extract.max_viewpoints == -1
    assert extract.image_size == 224
    assert extract.vfov == 60
    assert extract.sensor_height == 1.25
    assert validate.features == DEFAULT_OUTPUT_FILE
    assert validate.connectivity == DEFAULT_CONNECTIVITY_DIR
    assert validate.clip_features == DEFAULT_CLIP_FEATURES
    assert validate.expected_count == 10567
