#!/usr/bin/env python3
"""Validate a complete RAE/DINOv2 CLS feature HDF5 file."""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

try:
    from precompute_img_features.extract_rae_dinov2_features import (
        DEFAULT_CONNECTIVITY_DIR,
        DEFAULT_MODEL_DIR,
        DEFAULT_OUTPUT_FILE,
        build_metadata,
        load_connectivity_viewpoints,
    )
except ModuleNotFoundError:
    from extract_rae_dinov2_features import (
        DEFAULT_CONNECTIVITY_DIR,
        DEFAULT_MODEL_DIR,
        DEFAULT_OUTPUT_FILE,
        build_metadata,
        load_connectivity_viewpoints,
    )


DEFAULT_CLIP_FEATURES = "pretrain_src/img_features/CLIP-ViT-B-32-views-habitat.hdf5"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _normalize_attribute(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def validate_feature_file(
    features,
    connectivity_keys,
    clip_features=None,
    expected_count=10567,
    expected_metadata=None,
    max_error_examples=20,
):
    expected_keys = set(connectivity_keys)
    summary = {
        "features": str(features),
        "valid": True,
        "expected_count": int(expected_count),
        "actual_count": 0,
        "connectivity_count": len(expected_keys),
        "checked_datasets": 0,
        "missing_key_count": 0,
        "extra_key_count": 0,
        "clip_key_count": None,
        "error_count": 0,
        "errors": [],
    }

    def add_error(message):
        summary["error_count"] += 1
        if len(summary["errors"]) < max_error_examples:
            summary["errors"].append(str(message))

    with h5py.File(features, "r") as handle:
        actual_keys = set(handle.keys())
        summary["actual_count"] = len(actual_keys)
        if len(actual_keys) != expected_count:
            add_error(
                f"feature key count is {len(actual_keys)}, expected {expected_count}"
            )
        if len(expected_keys) != expected_count:
            add_error(
                f"connectivity key count is {len(expected_keys)}, expected {expected_count}"
            )

        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        summary["missing_key_count"] = len(missing)
        summary["extra_key_count"] = len(extra)
        for key in missing:
            add_error(f"missing feature key: {key}")
        for key in extra:
            add_error(f"extra feature key: {key}")

        if expected_metadata is not None:
            for key, expected in expected_metadata.items():
                got = _normalize_attribute(handle.attrs.get(key))
                if type(got) is not type(expected) or got != expected:
                    add_error(
                        f"metadata {key} expected {expected!r} "
                        f"({type(expected).__name__}), got {got!r} "
                        f"({type(got).__name__})"
                    )
        for hash_key in ("dino_weights_sha256", "rae_stat_sha256"):
            hash_value = _normalize_attribute(handle.attrs.get(hash_key))
            if not isinstance(hash_value, str) or not SHA256_PATTERN.fullmatch(hash_value):
                add_error(
                    f"metadata {hash_key} must be a 64-character lowercase sha256 hash, "
                    f"got {hash_value!r}"
                )

        for key in sorted(actual_keys):
            link = handle.get(key, getlink=True)
            if not isinstance(link, h5py.HardLink):
                add_error(f"key {key} uses unsupported HDF5 link {type(link).__name__}")
            obj = handle.get(key)
            if not isinstance(obj, h5py.Dataset):
                add_error(f"key {key} is not an HDF5 dataset: {type(obj).__name__}")
                continue
            summary["checked_datasets"] += 1
            if obj.shape != (36, 768):
                add_error(
                    f"key {key} has actual shape {obj.shape}, expected (36, 768)"
                )
            if obj.dtype != np.dtype(np.float32):
                add_error(
                    f"key {key} has actual dtype {obj.dtype}, expected float32"
                )
            values = obj[...]
            if not np.isfinite(values).all():
                add_error(f"key {key} contains non-finite values")
            if not np.any(values != 0):
                add_error(f"key {key} is all zero")

    if clip_features is not None:
        with h5py.File(clip_features, "r") as clip_handle:
            clip_keys = set(clip_handle.keys())
        summary["clip_key_count"] = len(clip_keys)
        for key in sorted(expected_keys - clip_keys):
            add_error(f"CLIP reference is missing key: {key}")
        for key in sorted(clip_keys - expected_keys):
            add_error(f"CLIP reference has extra key: {key}")

    summary["valid"] = summary["error_count"] == 0
    return summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate RAE/DINOv2 feature keys, arrays, and metadata."
    )
    parser.add_argument("--features", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--connectivity", default=DEFAULT_CONNECTIVITY_DIR)
    parser.add_argument("--clip_features", default=DEFAULT_CLIP_FEATURES)
    parser.add_argument("--expected_count", type=int, default=10567)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--stat_path", default=f"{DEFAULT_MODEL_DIR}/stat.pt")
    parser.add_argument("--max_error_examples", type=int, default=20)
    parser.add_argument(
        "--skip_asset_hash_check",
        action="store_true",
        help="Validate hash syntax without recomputing local model/stat hashes.",
    )
    return parser


def _metadata_without_asset_hashes():
    return {
        "feature_extractor": "rae_dinov2_with_registers_base_cls",
        "feature_dim": 768,
        "dtype": "float32",
        "num_views": 36,
        "image_size": 224,
        "vfov": 60,
        "sensor_height": 1.25,
        "latent_normalized": True,
        "preprocess_version": "rae_native_224_rgb_v1",
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        records = load_connectivity_viewpoints(args.connectivity)
        expected_metadata = (
            _metadata_without_asset_hashes()
            if args.skip_asset_hash_check
            else build_metadata(args.model_dir, args.stat_path)
        )
        summary = validate_feature_file(
            Path(args.features),
            connectivity_keys=[record.key for record in records],
            clip_features=Path(args.clip_features),
            expected_count=args.expected_count,
            expected_metadata=expected_metadata,
            max_error_examples=args.max_error_examples,
        )
    except Exception as error:
        summary = {
            "features": str(args.features),
            "valid": False,
            "error_count": 1,
            "errors": [f"{type(error).__name__}: {error}"],
        }
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
