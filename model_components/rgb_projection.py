import torch.nn as nn


def build_rgb_projection(encoder_type, raw_size, output_size, hidden_size):
    encoder_type = str(encoder_type).lower()
    raw_size = int(raw_size)
    output_size = int(output_size)
    hidden_size = int(hidden_size)

    if encoder_type == "clip":
        if raw_size != 512 or output_size != 512:
            raise ValueError("CLIP requires raw_size=output_size=512")
        return nn.Identity()

    if encoder_type != "rae_dinov2":
        raise ValueError(f"Unsupported RGB encoder type: {encoder_type}")

    if raw_size != 768 or output_size != 512 or hidden_size != 768:
        raise ValueError("RAE/DINOv2 projection requires 768->768->768->512")

    return nn.Sequential(
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 512),
    )
