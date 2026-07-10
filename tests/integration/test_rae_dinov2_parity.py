import os
from pathlib import Path
import subprocess
import sys
import textwrap

import torch
import torch.nn.functional as F

ETPNAV_ROOT = Path("/home/a6000/gwl/ETPNav")
ETPR1_ROOT = Path("/home/a6000/gwl/ETP-R1")
MODEL_DIR = Path(
    "/home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base"
)
STAT_PATH = MODEL_DIR / "stat.pt"


REFERENCE_SCRIPT = textwrap.dedent(
    r"""
    import importlib.util
    import sys
    from pathlib import Path

    import torch
    from transformers import AutoImageProcessor

    etpnav_root = Path(sys.argv[1]).resolve()
    etpr1_root = Path(sys.argv[2]).resolve()
    model_dir = Path(sys.argv[3]).resolve()
    stat_path = Path(sys.argv[4]).resolve()
    image_path = Path(sys.argv[5]).resolve()
    output_path = Path(sys.argv[6]).resolve()

    assert sys.dont_write_bytecode, "reference import must not write ETPNav bytecode"
    sys.path.insert(0, str(etpr1_root))
    from vlnce_baselines.models.encoders.rae_dinov2_encoder import (
        RaeDinov2ClsEncoder,
    )

    reference_file = (
        etpnav_root / "vlnce_baselines/common/rae_visual_encoder.py"
    )
    spec = importlib.util.spec_from_file_location(
        "etpnav_rae_visual_encoder_reference",
        reference_file,
    )
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    Dinov2withNorm = reference_module.Dinov2withNorm

    rgb = torch.load(image_path, map_location="cpu")
    device = torch.device("cuda:0")
    pixels = rgb.permute(0, 3, 1, 2).contiguous()
    pixels = pixels.to(device=device, dtype=torch.float32).div(255.0)
    processor = AutoImageProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    image_mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    image_std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    pixels = (pixels - image_mean) / image_std

    encoder = Dinov2withNorm(str(model_dir), normalize=True).to(device).eval()
    with torch.no_grad():
        _patch_tokens, cls = encoder.encode_with_cls(pixels)
        cls = cls.float()

    stats = torch.load(stat_path, map_location="cpu")
    var = stats["var"].float().to(device).mean(dim=(1, 2)).unsqueeze(0)
    mean = stats.get("mean")
    if mean is not None:
        mean = mean.float().to(device).mean(dim=(1, 2)).unsqueeze(0)
    else:
        mean = 0.0
    cls = (cls - mean) / torch.sqrt(var + 1e-5)
    if not torch.isfinite(cls).all():
        raise FloatingPointError("reference RAE CLS contains NaN or infinity")

    production_encoder = RaeDinov2ClsEncoder(
        model_dir=model_dir,
        stat_path=stat_path,
        device=device,
    )
    actual = production_encoder({"rgb": rgb})
    torch.save(
        {
            "reference": cls.cpu(),
            "actual": actual.cpu(),
            "encoder_training": production_encoder.training,
            "backbone_training": production_encoder.backbone.training,
            "all_backbone_parameters_frozen": all(
                not parameter.requires_grad
                for parameter in production_encoder.backbone.parameters()
            ),
            "final_layernorm_affine": (
                production_encoder.backbone.layernorm.elementwise_affine
            ),
            "final_layernorm_has_weight": (
                production_encoder.backbone.layernorm.weight is not None
            ),
            "final_layernorm_has_bias": (
                production_encoder.backbone.layernorm.bias is not None
            ),
        },
        output_path,
    )
    print(f"reference_file={reference_file}")
    """
)


def test_production_encoder_matches_etpnav_rae_reference(tmp_path):
    assert ETPNAV_ROOT.is_dir(), f"missing read-only ETPNav reference: {ETPNAV_ROOT}"
    for asset_name in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "stat.pt",
    ):
        assert (MODEL_DIR / asset_name).is_file(), (
            f"missing ETP-R1 RAE/DINOv2 asset: {MODEL_DIR / asset_name}"
        )
    assert torch.cuda.is_available(), "real RAE/DINOv2 parity requires CUDA"

    generator = torch.Generator(device="cpu").manual_seed(20260710)
    rgb = torch.randint(
        0,
        256,
        (1, 224, 224, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    image_path = tmp_path / "rgb.pt"
    reference_path = tmp_path / "reference.pt"
    torch.save(rgb, image_path)

    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            REFERENCE_SCRIPT,
            str(ETPNAV_ROOT),
            str(ETPR1_ROOT),
            str(MODEL_DIR),
            str(STAT_PATH),
            str(image_path),
            str(reference_path),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert str(ETPNAV_ROOT / "vlnce_baselines") in result.stdout

    outputs = torch.load(reference_path, map_location="cpu")
    reference = outputs["reference"].float()
    actual = outputs["actual"].float()

    assert outputs["encoder_training"] is False
    assert outputs["backbone_training"] is False
    assert outputs["all_backbone_parameters_frozen"] is True
    assert outputs["final_layernorm_affine"] is False
    assert outputs["final_layernorm_has_weight"] is False
    assert outputs["final_layernorm_has_bias"] is False
    assert reference.shape == actual.shape == (1, 768)
    max_abs = (reference - actual).abs().max().item()
    cosine = F.cosine_similarity(reference, actual).item()
    print(f"RAE/DINOv2 parity max_abs={max_abs:.10g} cosine={cosine:.10g}")
    assert max_abs <= 1e-5
    assert cosine >= 0.999999
