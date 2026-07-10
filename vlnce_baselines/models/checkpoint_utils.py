import hashlib
from collections.abc import Mapping
from pathlib import Path


_RAE_TYPE = "rae_dinov2"
_CLIP_TYPE = "clip"
_RAE_RAW_OUTPUT_SIZE = 768
_NAVIGATION_OUTPUT_SIZE = 512


def sha256_file(path):
    """Return the SHA256 digest of a regular file without loading it at once."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SHA256 input does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"SHA256 input is not a file: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_config(config):
    try:
        return config.MODEL.RGB_ENCODER
    except AttributeError as error:
        raise ValueError("Config is missing MODEL.RGB_ENCODER") from error


def _encoder_type(config):
    rgb_config = _rgb_config(config)
    try:
        encoder_type = str(rgb_config.type).lower()
    except AttributeError as error:
        raise ValueError("Config is missing MODEL.RGB_ENCODER.type") from error
    if encoder_type not in {_CLIP_TYPE, _RAE_TYPE}:
        raise ValueError(f"Unsupported RGB encoder type: {encoder_type}")
    return encoder_type


def _key_has_adjacent_parts(key, *expected):
    parts = str(key).split(".")
    width = len(expected)
    return any(
        tuple(parts[index:index + width]) == expected
        for index in range(len(parts) - width + 1)
    )


def _is_rgb_backbone_key(key):
    return _key_has_adjacent_parts(key, "rgb_encoder", "backbone")


def _is_rgb_projection_key(key):
    return _key_has_adjacent_parts(key, "img_embeddings", "rgb_projection")


def _is_first_rgb_projection_weight(key):
    return _key_has_adjacent_parts(
        key,
        "img_embeddings",
        "rgb_projection",
        "0",
        "weight",
    )


def _rae_dimensions(config):
    rgb_config = _rgb_config(config)
    try:
        raw_output_size = rgb_config.raw_output_size
        output_size = rgb_config.output_size
    except AttributeError as error:
        raise ValueError(
            "RAE/DINOv2 config must define integer raw_output_size and output_size"
        ) from error
    for field, value in (
        ("raw_output_size", raw_output_size),
        ("output_size", output_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"RAE/DINOv2 config {field} must be an integer, got {value!r}"
            )
    if raw_output_size != _RAE_RAW_OUTPUT_SIZE:
        raise ValueError(
            "RAE/DINOv2 config raw_output_size must be "
            f"{_RAE_RAW_OUTPUT_SIZE}, got {raw_output_size}"
        )
    if output_size != _NAVIGATION_OUTPUT_SIZE:
        raise ValueError(
            "RAE/DINOv2 config output_size must be "
            f"{_NAVIGATION_OUTPUT_SIZE}, got {output_size}"
        )
    return raw_output_size, output_size


def _rae_asset_metadata(config):
    rgb_config = _rgb_config(config)
    raw_output_size, output_size = _rae_dimensions(config)
    try:
        model_dir = Path(rgb_config.model_dir)
        stat_path = Path(rgb_config.stat_path)
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "RAE/DINOv2 config must define model_dir and stat_path"
        ) from error
    return {
        "type": _RAE_TYPE,
        "model_dir": str(rgb_config.model_dir),
        "model_sha256": sha256_file(model_dir / "model.safetensors"),
        "stat_sha256": sha256_file(stat_path),
        "raw_output_size": raw_output_size,
        "output_size": output_size,
    }


def navigation_state_dict(policy, config):
    """Build the online navigation checkpoint state and RGB metadata."""
    state_dict = policy.state_dict()
    encoder_type = _encoder_type(config)
    if encoder_type == _CLIP_TYPE:
        return state_dict, {"type": _CLIP_TYPE}

    filtered_state = {
        key: value
        for key, value in state_dict.items()
        if not _is_rgb_backbone_key(key)
    }
    return filtered_state, _rae_asset_metadata(config)


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Navigation checkpoint must be a mapping")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Navigation checkpoint state_dict is missing or invalid")
    return state_dict


def validate_rgb_checkpoint_metadata(checkpoint, config):
    """Validate RGB checkpoint compatibility before loading policy weights."""
    state_dict = _checkpoint_state_dict(checkpoint)
    encoder_type = _encoder_type(config)
    metadata = checkpoint.get("rgb_encoder")

    if encoder_type == _CLIP_TYPE:
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValueError("CLIP checkpoint rgb_encoder metadata is invalid")
            checkpoint_type = str(metadata.get("type", "")).lower()
            if checkpoint_type != _CLIP_TYPE:
                raise ValueError(
                    "CLIP config cannot load RAE/DINOv2 checkpoint metadata "
                    f"(got type {checkpoint_type or '<missing>'})"
                )
        if any(_is_rgb_projection_key(key) for key in state_dict):
            raise ValueError(
                "CLIP config cannot load a RAE/DINOv2 rgb_projection MLP"
            )
        return None

    if metadata is None:
        raise ValueError("RAE/DINOv2 checkpoint metadata is missing")
    if not isinstance(metadata, Mapping):
        raise ValueError("RAE/DINOv2 checkpoint metadata must be a mapping")

    checkpoint_type = str(metadata.get("type", "")).lower()
    if checkpoint_type != _RAE_TYPE:
        raise ValueError(
            "RGB encoder type mismatch: expected rae_dinov2, "
            f"got {checkpoint_type or '<missing>'}"
        )

    expected = _rae_asset_metadata(config)
    for field, label in (
        ("model_sha256", "model SHA256"),
        ("stat_sha256", "stat SHA256"),
    ):
        actual_value = metadata.get(field)
        if actual_value != expected[field]:
            raise ValueError(
                f"RAE/DINOv2 {label} mismatch: expected {expected[field]}, "
                f"got {actual_value}"
            )

    for field in ("raw_output_size", "output_size"):
        actual_value = metadata.get(field)
        if actual_value != expected[field]:
            raise ValueError(
                f"RAE/DINOv2 {field} mismatch: expected {expected[field]}, "
                f"got {actual_value}"
            )

    if not any(_is_first_rgb_projection_weight(key) for key in state_dict):
        raise ValueError(
            "RAE/DINOv2 checkpoint is missing "
            "img_embeddings.rgb_projection.0.weight"
        )
    return None


def report_navigation_incompatible_keys(
    incompatible_keys,
    config,
    print_fn=print,
):
    """Report all load mismatches while separating expected RAE backbone gaps."""
    encoder_type = _encoder_type(config)
    missing_keys = sorted(incompatible_keys.missing_keys)
    unexpected_keys = sorted(incompatible_keys.unexpected_keys)
    if encoder_type == _RAE_TYPE:
        ignored_missing_keys = [
            key for key in missing_keys if _is_rgb_backbone_key(key)
        ]
    else:
        ignored_missing_keys = []
    ignored_missing_set = set(ignored_missing_keys)
    remaining_missing_keys = [
        key for key in missing_keys if key not in ignored_missing_set
    ]

    print_fn("\n" + "=" * 25 + " Weight loading mismatch report " + "=" * 25)
    if ignored_missing_keys:
        print_fn(
            "The following missing keys are the ignored frozen "
            "RAE/DINOv2 backbone:"
        )
        for key in ignored_missing_keys:
            print_fn(f"  - {key}")
    if remaining_missing_keys:
        print_fn(
            "The following network layers exist in the model but are missing "
            "in the weight file (initial values will be used):"
        )
        for key in remaining_missing_keys:
            print_fn(f"  - {key}")
    else:
        print_fn("There are no unhandled missing network layers.")
    if unexpected_keys:
        print_fn(
            "The following network layers exist in the weight file but are "
            "missing in the model (will be ignored):"
        )
        for key in unexpected_keys:
            print_fn(f"  - {key}")
    else:
        print_fn("There are no extra network layers in the weight file.")
    print_fn("=" * 75 + "\n")

    return {
        "ignored_missing_keys": ignored_missing_keys,
        "missing_keys": remaining_missing_keys,
        "unexpected_keys": unexpected_keys,
    }
