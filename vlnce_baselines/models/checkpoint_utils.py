import hashlib
from collections.abc import Mapping
from pathlib import Path


_RAE_TYPE = "rae_dinov2"
_CLIP_TYPE = "clip"
_RAE_RAW_OUTPUT_SIZE = 768
_NAVIGATION_OUTPUT_SIZE = 512
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECTION_PARAMETER_SUFFIXES = (
    "0.weight",
    "0.bias",
    "2.weight",
    "2.bias",
    "4.weight",
    "4.bias",
)


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


def is_rgb_backbone_key(key):
    return _key_has_adjacent_parts(key, "rgb_encoder", "backbone")


_is_rgb_backbone_key = is_rgb_backbone_key


def _is_rgb_projection_key(key):
    return _key_has_adjacent_parts(key, "img_embeddings", "rgb_projection")


def _rgb_projection_parameter_groups(state_dict):
    groups = {}
    marker = ("img_embeddings", "rgb_projection")
    expected_suffixes = set(_PROJECTION_PARAMETER_SUFFIXES)
    for key in state_dict:
        parts = str(key).split(".")
        for index in range(len(parts) - 1):
            if tuple(parts[index:index + 2]) != marker:
                continue
            suffix = ".".join(parts[index + 2:])
            if suffix in expected_suffixes:
                prefix = ".".join(parts[:index + 2])
                groups.setdefault(prefix, set()).add(suffix)
            break
    return groups


def _validate_complete_rgb_projection(state_dict):
    groups = _rgb_projection_parameter_groups(state_dict)
    expected_suffixes = set(_PROJECTION_PARAMETER_SUFFIXES)
    if any(expected_suffixes.issubset(suffixes) for suffixes in groups.values()):
        return

    missing_keys = []
    if groups:
        for prefix, suffixes in sorted(groups.items()):
            for suffix in _PROJECTION_PARAMETER_SUFFIXES:
                if suffix not in suffixes:
                    missing_keys.append(f"{prefix}.{suffix}")
    else:
        missing_keys = [
            f"img_embeddings.rgb_projection.{suffix}"
            for suffix in _PROJECTION_PARAMETER_SUFFIXES
        ]
    raise ValueError(
        "RAE/DINOv2 checkpoint requires a complete rgb_projection parameter "
        "set; missing expected semantic keys: "
        + ", ".join(missing_keys)
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
        configured_model_dir = Path(rgb_config.model_dir)
        configured_stat_path = Path(rgb_config.stat_path)
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "RAE/DINOv2 config must define model_dir and stat_path"
        ) from error
    project_root = _PROJECT_ROOT.resolve()
    model_dir = (
        configured_model_dir
        if configured_model_dir.is_absolute()
        else project_root / configured_model_dir
    ).resolve()
    stat_path = (
        configured_stat_path
        if configured_stat_path.is_absolute()
        else project_root / configured_stat_path
    ).resolve()
    try:
        relative_model_dir = model_dir.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            "RAE/DINOv2 model_dir is outside the ETP-R1 project root: "
            f"{model_dir}"
        ) from error
    return {
        "type": _RAE_TYPE,
        "model_dir": relative_model_dir.as_posix(),
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
    if not state_dict:
        raise ValueError("Navigation checkpoint state_dict is empty")
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
    actual_model_dir = metadata.get("model_dir")
    if actual_model_dir != expected["model_dir"]:
        raise ValueError(
            "RAE/DINOv2 model_dir mismatch: expected "
            f"{expected['model_dir']}, got {actual_model_dir}"
        )
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
        if isinstance(actual_value, bool) or not isinstance(actual_value, int):
            raise ValueError(
                f"RAE/DINOv2 metadata {field} must be a non-boolean "
                f"integer, got {actual_value!r}"
            )
        if actual_value != expected[field]:
            raise ValueError(
                f"RAE/DINOv2 {field} mismatch: expected {expected[field]}, "
                f"got {actual_value}"
            )

    _validate_complete_rgb_projection(state_dict)
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
