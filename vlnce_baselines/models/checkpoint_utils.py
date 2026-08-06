import hashlib
from collections.abc import Mapping
from pathlib import Path

import torch


_RAE_TYPE = "rae_dinov2"
_CLIP_TYPE = "clip"
_RAE_OUTPUT_SIZE = 768
_RAE_PIPELINE = "etpnav_raw_cls_residual_mlp_v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESIDUAL_MLP_PARAMETER_SUFFIXES = (
    "0.weight",
    "0.bias",
    "2.weight",
    "2.bias",
    "4.weight",
    "4.bias",
)
_RESIDUAL_MLP_PARAMETER_SHAPES = {
    "0.weight": (768, 768),
    "0.bias": (768,),
    "2.weight": (768, 768),
    "2.bias": (768,),
    "4.weight": (768, 768),
    "4.bias": (768,),
}


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


def _is_legacy_rgb_projection_key(key):
    return _key_has_adjacent_parts(key, "img_embeddings", "rgb_projection")


def _is_cls_residual_mlp_key(key):
    return _key_has_adjacent_parts(key, "rgb_encoder", "cls_residual_mlp")


def _exact_config_bool(rgb_config, field):
    try:
        value = getattr(rgb_config, field)
    except AttributeError as error:
        raise ValueError(
            f"RAE/DINOv2 config must define {field}"
        ) from error
    if type(value) is not bool:
        raise ValueError(
            f"RAE/DINOv2 config {field} must be a boolean, got {value!r}"
        )
    return value


def _rae_config_contract(config):
    rgb_config = _rgb_config(config)
    try:
        output_size = rgb_config.output_size
        hidden_dim = rgb_config.cls_residual_mlp_hidden_dim
    except AttributeError as error:
        raise ValueError(
            "RAE/DINOv2 config must define output_size and the CLS residual "
            "MLP settings"
        ) from error
    if isinstance(output_size, bool) or not isinstance(output_size, int):
        raise ValueError(
            "RAE/DINOv2 config output_size must be an integer, "
            f"got {output_size!r}"
        )
    if output_size != _RAE_OUTPUT_SIZE:
        raise ValueError(
            "RAE/DINOv2 config output_size must be "
            f"{_RAE_OUTPUT_SIZE}, got {output_size}"
        )
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int):
        raise ValueError(
            "RAE/DINOv2 config cls_residual_mlp_hidden_dim must be an "
            f"integer, got {hidden_dim!r}"
        )
    if hidden_dim != _RAE_OUTPUT_SIZE:
        raise ValueError(
            "ETPNav-compatible CLS residual MLP hidden dimension must be "
            f"{_RAE_OUTPUT_SIZE}, got {hidden_dim}"
        )
    enabled = _exact_config_bool(rgb_config, "cls_residual_mlp_enabled")
    zero_init = _exact_config_bool(rgb_config, "cls_residual_mlp_zero_init")
    if not enabled:
        raise ValueError(
            "ETPNav rgb17400 compatibility requires "
            "cls_residual_mlp_enabled=True"
        )
    if not zero_init:
        raise ValueError(
            "ETPNav rgb17400 compatibility requires "
            "cls_residual_mlp_zero_init=True"
        )
    return {
        "output_size": output_size,
        "cls_residual_mlp_enabled": enabled,
        "cls_residual_mlp_hidden_dim": hidden_dim,
        "cls_residual_mlp_zero_init": zero_init,
    }


def _rae_asset_metadata(config):
    rgb_config = _rgb_config(config)
    contract = _rae_config_contract(config)
    try:
        configured_model_dir = Path(rgb_config.model_dir)
    except (AttributeError, TypeError) as error:
        raise ValueError(
            "RAE/DINOv2 config must define model_dir"
        ) from error
    project_root = _PROJECT_ROOT.resolve()
    model_dir = (
        configured_model_dir
        if configured_model_dir.is_absolute()
        else project_root / configured_model_dir
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
        "pipeline": _RAE_PIPELINE,
        "model_dir": relative_model_dir.as_posix(),
        "model_sha256": sha256_file(model_dir / "model.safetensors"),
        "cls_normalization": "none",
        **contract,
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


def _residual_mlp_groups(state_dict):
    marker = ("rgb_encoder", "cls_residual_mlp", "layers")
    groups = {}
    for key, value in state_dict.items():
        parts = str(key).split(".")
        for index in range(len(parts) - len(marker) + 1):
            if tuple(parts[index:index + len(marker)]) != marker:
                continue
            prefix_parts = parts[:index + len(marker)]
            prefix = ".".join(prefix_parts)
            suffix = ".".join(parts[index + len(marker):])
            groups.setdefault(prefix, {})[suffix] = (key, value)
            break
    return groups


def _validate_complete_cls_residual_mlp(state_dict):
    groups = _residual_mlp_groups(state_dict)
    expected = set(_RESIDUAL_MLP_PARAMETER_SUFFIXES)
    if not groups:
        raise ValueError(
            "RAE/DINOv2 checkpoint is missing the ETPNav CLS residual MLP"
        )

    errors = []
    complete_groups = []
    for prefix, values in sorted(groups.items()):
        suffixes = set(values)
        missing = sorted(expected - suffixes)
        extra = sorted(suffixes - expected)
        if missing or extra:
            errors.append(
                f"{prefix}: missing={missing}, extra={extra}"
            )
            continue
        complete_groups.append((prefix, values))
    if errors or not complete_groups:
        detail = "; ".join(errors) if errors else "no complete parameter group"
        raise ValueError(
            "RAE/DINOv2 checkpoint requires one complete ETPNav CLS residual "
            f"MLP parameter set; {detail}"
        )
    if len(complete_groups) != 1:
        raise ValueError(
            "RAE/DINOv2 checkpoint contains multiple CLS residual MLP "
            f"parameter groups: {[prefix for prefix, _ in complete_groups]}"
        )

    _prefix, values = complete_groups[0]
    for suffix, expected_shape in _RESIDUAL_MLP_PARAMETER_SHAPES.items():
        key, value = values[suffix]
        actual_shape = tuple(value.shape) if torch.is_tensor(value) else None
        if actual_shape != expected_shape:
            raise ValueError(
                f"RAE/DINOv2 checkpoint {key} must have shape "
                f"{expected_shape}, got {actual_shape}"
            )


def validate_rgb_checkpoint_metadata(checkpoint, config):
    """Validate RGB checkpoint compatibility before loading policy weights."""
    state_dict = _checkpoint_state_dict(checkpoint)
    encoder_type = _encoder_type(config)
    metadata = checkpoint.get("rgb_encoder")

    legacy_projection_keys = sorted(
        key for key in state_dict if _is_legacy_rgb_projection_key(key)
    )
    if legacy_projection_keys:
        raise ValueError(
            "Checkpoint uses the retired RAE/DINOv2 rgb_projection pipeline: "
            + ", ".join(legacy_projection_keys)
        )

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
        if any(_is_cls_residual_mlp_key(key) for key in state_dict):
            raise ValueError(
                "CLIP config cannot load a RAE/DINOv2 CLS residual MLP"
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
    for field in (
        "pipeline",
        "model_dir",
        "model_sha256",
        "cls_normalization",
        "output_size",
        "cls_residual_mlp_enabled",
        "cls_residual_mlp_hidden_dim",
        "cls_residual_mlp_zero_init",
    ):
        actual_value = metadata.get(field)
        expected_value = expected[field]
        if type(actual_value) is not type(expected_value):
            raise ValueError(
                f"RAE/DINOv2 metadata {field} has invalid type: expected "
                f"{type(expected_value).__name__}, got "
                f"{type(actual_value).__name__}"
            )
        if actual_value != expected_value:
            raise ValueError(
                f"RAE/DINOv2 {field} mismatch: expected "
                f"{expected_value}, got {actual_value}"
            )

    _validate_complete_cls_residual_mlp(state_dict)
    return None


def report_navigation_incompatible_keys(
    incompatible_keys,
    config,
    print_fn=print,
):
    """Report load mismatches while separating expected frozen-backbone gaps."""
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
