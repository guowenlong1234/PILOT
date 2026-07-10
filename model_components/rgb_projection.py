import torch.nn as nn


def _exact_int(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact integer, got {value!r}")

    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"{name} must be an exact integer, got {value!r}"
        ) from None

    if not isinstance(value, str):
        try:
            is_exact = value == converted
            if not is_exact:
                raise ValueError(
                    f"{name} must be an exact integer, got {value!r}"
                )
        except (TypeError, ValueError):
            raise ValueError(
                f"{name} must be an exact integer, got {value!r}"
            ) from None

    return converted


def build_rgb_projection(encoder_type, raw_size, output_size, hidden_size):
    encoder_type = str(encoder_type).lower()
    raw_size = _exact_int("raw_size", raw_size)
    output_size = _exact_int("output_size", output_size)
    hidden_size = _exact_int("hidden_size", hidden_size)

    if encoder_type == "clip":
        if raw_size != 512 or output_size != 512:
            raise ValueError(
                "CLIP requires raw_size=512 and output_size=512; "
                f"got raw_size={raw_size}, output_size={output_size}, "
                f"hidden_size={hidden_size}"
            )
        return nn.Identity()

    if encoder_type != "rae_dinov2":
        raise ValueError(f"Unsupported RGB encoder type: {encoder_type}")

    if raw_size != 768 or output_size != 512 or hidden_size != 768:
        raise ValueError(
            "RAE/DINOv2 projection requires raw_size=768, hidden_size=768, "
            f"and output_size=512; got raw_size={raw_size}, "
            f"output_size={output_size}, hidden_size={hidden_size}"
        )

    return nn.Sequential(
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 512),
    )
