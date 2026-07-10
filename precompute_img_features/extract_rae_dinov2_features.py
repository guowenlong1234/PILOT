#!/usr/bin/env python3
"""Render MP3D panoramas and write normalized RAE/DINOv2 CLS features."""

import argparse
import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


NUM_VIEWS = 36
FEATURE_DIM = 768
DEFAULT_MODEL_DIR = "pretrained/rae_dinov2_with_registers_base"
DEFAULT_CONNECTIVITY_DIR = "precompute_img_features/connectivity"
DEFAULT_SCENES_DIR = "data/scene_datasets/mp3d"
DEFAULT_OUTPUT_FILE = (
    "pretrain_src/img_features/RAE-DINOv2-B-14-CLS-views-habitat.hdf5"
)

LOGGER = logging.getLogger("rae_dinov2_feature_extractor")


@dataclass(frozen=True)
class ViewpointRecord:
    scan_id: str
    viewpoint_id: str
    position: np.ndarray

    @property
    def key(self):
        return f"{self.scan_id}_{self.viewpoint_id}"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(model_dir, stat_path):
    model_path = Path(model_dir) / "model.safetensors"
    stat_path = Path(stat_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"DINO weights not found: {model_path}")
    if not stat_path.is_file():
        raise FileNotFoundError(f"RAE stat not found: {stat_path}")
    return {
        "feature_extractor": "rae_dinov2_with_registers_base_cls",
        "feature_dim": 768,
        "dtype": "float32",
        "num_views": 36,
        "image_size": 224,
        "vfov": 60,
        "sensor_height": 1.25,
        "latent_normalized": True,
        "dino_weights_sha256": sha256_file(model_path),
        "rae_stat_sha256": sha256_file(stat_path),
        "preprocess_version": "rae_native_224_rgb_v1",
    }


def _mp3d_to_habitat_agent_position(mp3d_position, sensor_height):
    position = np.asarray(mp3d_position, dtype=np.float32)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"MP3D position must contain three finite values, got {position}")
    x, y, z = position
    return np.asarray([x, z - sensor_height, -y], dtype=np.float32)


def load_connectivity_viewpoints(connectivity_dir, sensor_height=1.25):
    connectivity_dir = Path(connectivity_dir)
    scans_file = connectivity_dir / "scans.txt"
    with scans_file.open("r", encoding="utf-8") as handle:
        scans = sorted({line.strip() for line in handle if line.strip()})

    records = {}
    for scan_id in scans:
        path = connectivity_dir / f"{scan_id}_connectivity.json"
        with path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in entries:
            if not entry.get("included", False):
                continue
            viewpoint_id = str(entry["image_id"])
            pose = entry.get("pose")
            if not isinstance(pose, list) or len(pose) < 12:
                raise ValueError(f"connectivity pose for {scan_id}_{viewpoint_id} is invalid")
            mp3d_position = [pose[3], pose[7], pose[11]]
            record = ViewpointRecord(
                scan_id=scan_id,
                viewpoint_id=viewpoint_id,
                position=_mp3d_to_habitat_agent_position(
                    mp3d_position, float(sensor_height)
                ),
            )
            old_record = records.get(record.key)
            if old_record is not None and not np.array_equal(
                old_record.position, record.position
            ):
                raise ValueError(f"conflicting duplicate connectivity key: {record.key}")
            records[record.key] = record
    return [records[key] for key in sorted(records)]


def view_index_to_rotation_quat(view_index):
    if isinstance(view_index, bool) or not isinstance(view_index, (int, np.integer)):
        raise TypeError(f"view index must be an integer in [0, 35], got {view_index!r}")
    view_index = int(view_index)
    if not 0 <= view_index < NUM_VIEWS:
        raise ValueError(f"view index must be in [0, 35], got {view_index}")

    from habitat_sim.utils.common import quat_from_angle_axis

    heading = -math.radians((view_index % 12) * 30.0)
    elevation = math.radians((view_index // 12 - 1) * 30.0)
    heading_rotation = quat_from_angle_axis(
        heading, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    elevation_rotation = quat_from_angle_axis(
        elevation, np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    )
    return heading_rotation * elevation_rotation


def build_simulator(
    scene_path,
    image_size=224,
    hfov=60.0,
    sensor_height=1.25,
    sim_gpu_id=0,
):
    import habitat_sim

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene_path)
    if hasattr(simulator_config, "gpu_device_id"):
        simulator_config.gpu_device_id = int(sim_gpu_id)

    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "rgb"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [int(image_size), int(image_size)]
    sensor_spec.hfov = float(hfov)
    sensor_spec.position = [0.0, float(sensor_height), 0.0]

    agent_config = habitat_sim.agent.AgentConfiguration()
    agent_config.sensor_specifications = [sensor_spec]
    configuration = habitat_sim.Configuration(simulator_config, [agent_config])
    return habitat_sim.Simulator(configuration)


def render_36_views(simulator, position, image_size=224):
    position = np.asarray(position, dtype=np.float32)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"Habitat position must contain three finite values, got {position}")

    agent = simulator.get_agent(0)
    state = agent.get_state()
    state.position = position
    views = []
    for view_index in range(NUM_VIEWS):
        state.rotation = view_index_to_rotation_quat(view_index)
        try:
            agent.set_state(state, reset_sensors=True)
        except TypeError:
            agent.set_state(state)
        observations = simulator.get_sensor_observations()
        if "rgb" not in observations:
            raise KeyError("Habitat-Sim observations do not contain 'rgb'")
        rgb = np.asarray(observations["rgb"])
        if rgb.shape[:2] != (image_size, image_size) or rgb.ndim != 3:
            raise ValueError(
                f"Habitat-Sim RGB has shape {rgb.shape}, expected "
                f"({image_size}, {image_size}, 3 or 4)"
            )
        if rgb.shape[2] not in (3, 4):
            raise ValueError(f"Habitat-Sim RGB must have 3 or 4 channels, got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"Habitat-Sim RGB must be uint8, got {rgb.dtype}")
        views.append(rgb[..., :3].copy())
    return np.stack(views, axis=0).astype(np.uint8, copy=False)


def _validate_feature_array(features):
    if not isinstance(features, np.ndarray):
        raise ValueError(f"RAE/DINOv2 features must be a numpy array, got {type(features)}")
    if features.shape != (NUM_VIEWS, FEATURE_DIM):
        raise ValueError(
            f"RAE/DINOv2 feature shape must be {(NUM_VIEWS, FEATURE_DIM)}, "
            f"got {features.shape}"
        )
    if features.dtype != np.dtype(np.float32):
        raise ValueError(f"RAE/DINOv2 feature dtype must be float32, got {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("RAE/DINOv2 features must contain only finite values")
    if not np.any(features != 0):
        raise ValueError("RAE/DINOv2 features must not be all zero")


def encode_views(encoder, rgb_views, device, batch_size=12):
    rgb_views = np.asarray(rgb_views)
    if (
        rgb_views.ndim != 4
        or rgb_views.shape[0] != NUM_VIEWS
        or rgb_views.shape[-1] != 3
        or rgb_views.dtype != np.uint8
    ):
        raise ValueError(
            "RAE/DINOv2 views must have shape [36, H, W, 3] and dtype uint8, "
            f"got shape {rgb_views.shape} dtype {rgb_views.dtype}"
        )
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")

    encoder.train(False)
    encoder.requires_grad_(False)
    outputs = []
    with torch.inference_mode():
        for start in range(0, NUM_VIEWS, batch_size):
            batch = torch.from_numpy(rgb_views[start : start + batch_size]).to(device)
            output = encoder({"rgb": batch})
            if not torch.is_tensor(output) or output.ndim != 2:
                shape = tuple(output.shape) if torch.is_tensor(output) else None
                raise ValueError(f"RAE/DINOv2 encoder returned invalid shape {shape}")
            expected_shape = (len(batch), FEATURE_DIM)
            if tuple(output.shape) != expected_shape:
                raise ValueError(
                    f"RAE/DINOv2 encoder returned shape {tuple(output.shape)}, "
                    f"expected {expected_shape}"
                )
            outputs.append(output.float().cpu().numpy())
    features = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    _validate_feature_array(features)
    return features


def _normalize_attribute(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_existing_metadata(handle, expected_metadata):
    for key, expected in expected_metadata.items():
        got = _normalize_attribute(handle.attrs.get(key))
        if type(got) is not type(expected) or got != expected:
            raise ValueError(
                f"HDF5 metadata mismatch for {key}: expected {expected!r} "
                f"({type(expected).__name__}), got {got!r} ({type(got).__name__})"
            )


def _dataset_is_complete(obj):
    if not isinstance(obj, h5py.Dataset):
        return False
    if obj.shape != (NUM_VIEWS, FEATURE_DIM) or obj.dtype != np.dtype(np.float32):
        return False
    values = obj[...]
    return bool(np.isfinite(values).all() and np.any(values != 0))


def write_feature_file(
    output_file,
    viewpoints,
    encoder,
    simulator_factory,
    metadata,
    device,
    batch_size=12,
    image_size=224,
):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existed = output_file.exists()
    viewpoints = list(viewpoints)
    summary = {"total": len(viewpoints), "skipped": 0, "recomputed": 0, "completed": 0}

    current_scan = None
    simulator = None
    try:
        with h5py.File(output_file, "a") as handle:
            if existed:
                _validate_existing_metadata(handle, metadata)
            else:
                for key, value in metadata.items():
                    handle.attrs[key] = value
                handle.flush()

            for index, record in enumerate(viewpoints, start=1):
                key = record.key
                had_corrupt_entry = False
                if key in handle:
                    if _dataset_is_complete(handle[key]):
                        summary["skipped"] += 1
                        LOGGER.info(
                            "[%d/%d] skip complete key=%s skipped=%d",
                            index,
                            len(viewpoints),
                            key,
                            summary["skipped"],
                        )
                        continue
                    del handle[key]
                    handle.flush()
                    had_corrupt_entry = True
                    summary["recomputed"] += 1

                if record.scan_id != current_scan:
                    if simulator is not None:
                        simulator.close()
                    simulator = simulator_factory(record.scan_id)
                    current_scan = record.scan_id

                LOGGER.info(
                    "[%d/%d] encode scan=%s key=%s skipped=%d recomputed=%d completed=%d",
                    index,
                    len(viewpoints),
                    record.scan_id,
                    key,
                    summary["skipped"],
                    summary["recomputed"],
                    summary["completed"],
                )
                views = render_36_views(simulator, record.position, image_size=image_size)
                features = encode_views(
                    encoder, views, device=device, batch_size=batch_size
                )
                dataset = handle.create_dataset(
                    key,
                    data=features,
                    dtype=np.float32,
                    compression="gzip",
                )
                dataset.attrs["scanId"] = record.scan_id
                dataset.attrs["viewpointId"] = record.viewpoint_id
                handle.flush()
                summary["completed"] += 1
                if had_corrupt_entry:
                    LOGGER.info("recomputed corrupt key=%s", key)
    finally:
        if simulator is not None:
            simulator.close()
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate 36-view normalized RAE/DINOv2 CLS HDF5 features."
    )
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--stat_path", default=f"{DEFAULT_MODEL_DIR}/stat.pt")
    parser.add_argument("--connectivity_dir", default=DEFAULT_CONNECTIVITY_DIR)
    parser.add_argument("--scenes_dir", default=DEFAULT_SCENES_DIR)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim_gpu_id", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--hfov", type=float, default=60.0)
    parser.add_argument("--vfov", type=float, default=60.0)
    parser.add_argument("--sensor_height", type=float, default=1.25)
    parser.add_argument(
        "--max_viewpoints",
        type=int,
        default=-1,
        help="Process at most N sorted viewpoints; -1 means all.",
    )
    return parser


def _validate_fixed_geometry(args):
    if args.image_size != 224:
        raise ValueError("RAE native preprocessing requires --image_size=224")
    if args.hfov != 60.0 or args.vfov != 60.0:
        raise ValueError("feature semantics require --hfov=60 and --vfov=60")
    if args.sensor_height != 1.25:
        raise ValueError("feature semantics require --sensor_height=1.25")
    if args.max_viewpoints == 0 or args.max_viewpoints < -1:
        raise ValueError("--max_viewpoints must be -1 or a positive integer")


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _validate_fixed_geometry(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable but --device requests CUDA")

    from vlnce_baselines.models.encoders.rae_dinov2_encoder import (
        RaeDinov2ClsEncoder,
    )

    viewpoints = load_connectivity_viewpoints(
        args.connectivity_dir, sensor_height=args.sensor_height
    )
    if args.max_viewpoints > 0:
        viewpoints = viewpoints[: args.max_viewpoints]
    metadata = build_metadata(args.model_dir, args.stat_path)
    encoder = RaeDinov2ClsEncoder(args.model_dir, args.stat_path, device)

    scenes_dir = Path(args.scenes_dir)

    def simulator_factory(scan_id):
        scene_path = scenes_dir / scan_id / f"{scan_id}.glb"
        if not scene_path.is_file():
            raise FileNotFoundError(f"MP3D scene not found: {scene_path}")
        return build_simulator(
            scene_path,
            image_size=args.image_size,
            hfov=args.hfov,
            sensor_height=args.sensor_height,
            sim_gpu_id=args.sim_gpu_id,
        )

    LOGGER.info(
        "start total=%d output=%s connectivity=%s",
        len(viewpoints),
        os.path.abspath(args.output_file),
        os.path.abspath(args.connectivity_dir),
    )
    summary = write_feature_file(
        args.output_file,
        viewpoints,
        encoder,
        simulator_factory,
        metadata=metadata,
        device=device,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    LOGGER.info(
        "finished total=%d skipped=%d recomputed=%d completed=%d",
        summary["total"],
        summary["skipped"],
        summary["recomputed"],
        summary["completed"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
