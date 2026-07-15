import hashlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import torch
import torch.nn.functional as F

DINO_CWP_ROOT = Path("/home/a6000/gwl/dino_cwp")
ETPR1_ROOT = Path("/home/a6000/gwl/ETP-R1")
MODEL_DIR = Path(
    "/home/a6000/gwl/ETP-R1/pretrained/rae_dinov2_with_registers_base"
)
STAT_PATH = MODEL_DIR / "stat.pt"
REFERENCE_CONFIG = DINO_CWP_ROOT / (
    "dino_cwp/nwm/raenwm_core/RAE/configs/stage1/pretrained/DINOv2-B.yaml"
)
REFERENCE_MODEL_DIR = Path(
    "/home/a6000/gwl/RAE-NWM/raenwm/models/encoders/"
    "dinov2-with-registers-base"
)
REFERENCE_STAT_PATH = Path(
    "/home/a6000/gwl/RAE-NWM/raenwm/models/stats/"
    "dinov2/wReg_base/imagenet1k/stat.pt"
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


REFERENCE_SCRIPT = textwrap.dedent(
    r"""
    import inspect
    import sys
    from pathlib import Path

    import torch

    dino_cwp_root = Path(sys.argv[1]).resolve()
    reference_config = Path(sys.argv[2]).resolve()
    etpr1_root = Path(sys.argv[3]).resolve()
    model_dir = Path(sys.argv[4]).resolve()
    stat_path = Path(sys.argv[5]).resolve()
    image_path = Path(sys.argv[6]).resolve()
    output_path = Path(sys.argv[7]).resolve()

    assert sys.dont_write_bytecode, "reference import must not write bytecode"
    sys.path.insert(0, str(etpr1_root))
    from vlnce_baselines.models.encoders.rae_dinov2_encoder import (
        RaeDinov2ClsEncoder,
    )

    sys.path.insert(0, str(dino_cwp_root))
    from dino_cwp.nwm import rae_visual_encoder as reference_module
    from dino_cwp.nwm.raenwm_core.RAE.src.stage1.rae import RAE

    rgb = torch.load(image_path, map_location="cpu")
    device = torch.device("cuda:0")
    reference_params = reference_module.load_rae_stage1_params(
        reference_config
    )
    reference_rae = reference_module._build_rae_stage1(
        reference_params
    ).to(device).eval()
    assert isinstance(reference_rae, RAE)
    _patch_latents, reference = reference_module.encode_rae_rgb_with_cls(
        reference_rae,
        rgb,
        device=device,
        size=224,
    )

    protected_package = (dino_cwp_root / "dino_cwp").resolve()
    source_paths = {
        "reference_wrapper": Path(
            inspect.getsourcefile(reference_module.encode_rae_rgb_with_cls)
        ).resolve(),
        "reference_rae": Path(inspect.getsourcefile(RAE)).resolve(),
        "reference_stat_for_shape": Path(
            inspect.getsourcefile(RAE._stat_for_shape)
        ).resolve(),
        "reference_normalize_latent": Path(
            inspect.getsourcefile(RAE._normalize_latent)
        ).resolve(),
    }
    for name, source_path in source_paths.items():
        assert source_path.is_relative_to(protected_package), (
            f"{name} was imported from unexpected path {source_path}"
        )

    production_encoder = RaeDinov2ClsEncoder(
        model_dir=model_dir,
        stat_path=stat_path,
        device=device,
    )
    actual = production_encoder({"rgb": rgb})
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        autocast_actual = production_encoder({"rgb": rgb})
    torch.save(
        {
            "reference": reference.cpu(),
            "actual": actual.cpu(),
            "autocast_actual": autocast_actual.cpu(),
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
    for name, source_path in source_paths.items():
        print(f"{name}={source_path}")
    """
)


def test_production_encoder_matches_full_rae_reference(tmp_path):
    assert DINO_CWP_ROOT.is_dir(), (
        f"missing read-only full RAE reference: {DINO_CWP_ROOT}"
    )
    assert REFERENCE_CONFIG.is_file(), (
        f"missing full RAE reference config: {REFERENCE_CONFIG}"
    )
    reference_assets = {
        "config.json": REFERENCE_MODEL_DIR / "config.json",
        "preprocessor_config.json": (
            REFERENCE_MODEL_DIR / "preprocessor_config.json"
        ),
        "model.safetensors": REFERENCE_MODEL_DIR / "model.safetensors",
        "stat.pt": REFERENCE_STAT_PATH,
    }
    for asset_name, reference_asset in reference_assets.items():
        production_asset = MODEL_DIR / asset_name
        assert reference_asset.is_file(), (
            f"missing read-only RAE-NWM asset: {reference_asset}"
        )
        assert production_asset.is_file(), (
            f"missing ETP-R1 RAE/DINOv2 asset: {production_asset}"
        )
        assert _sha256(reference_asset) == _sha256(production_asset), (
            f"reference and production {asset_name} hashes differ"
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
    portable_reference_config = tmp_path / "DINOv2-B.yaml"
    reference_root = "/home/gwl/project/RAE-NWM/raenwm"
    eval_root = "/home/a6000/gwl/RAE-NWM/raenwm"
    reference_config_text = REFERENCE_CONFIG.read_text(encoding="utf-8")
    assert reference_root in reference_config_text, (
        "reference config no longer contains the expected source paths"
    )
    portable_reference_config.write_text(
        reference_config_text.replace(reference_root, eval_root),
        encoding="utf-8",
    )
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
            str(DINO_CWP_ROOT),
            str(portable_reference_config),
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
    assert str(DINO_CWP_ROOT / "dino_cwp/nwm/rae_visual_encoder.py") in result.stdout
    assert str(
        DINO_CWP_ROOT / "dino_cwp/nwm/raenwm_core/RAE/src/stage1/rae.py"
    ) in result.stdout

    outputs = torch.load(reference_path, map_location="cpu")
    reference = outputs["reference"].float()
    actual = outputs["actual"].float()
    autocast_actual = outputs["autocast_actual"].float()

    assert outputs["encoder_training"] is False
    assert outputs["backbone_training"] is False
    assert outputs["all_backbone_parameters_frozen"] is True
    assert outputs["final_layernorm_affine"] is False
    assert outputs["final_layernorm_has_weight"] is False
    assert outputs["final_layernorm_has_bias"] is False
    assert reference.shape == actual.shape == (1, 768)
    assert autocast_actual.shape == actual.shape
    assert outputs["autocast_actual"].dtype == torch.float32
    torch.testing.assert_close(autocast_actual, actual, rtol=0.0, atol=0.0)
    max_abs = (reference - actual).abs().max().item()
    cosine = F.cosine_similarity(reference, actual).item()
    autocast_max_abs = (autocast_actual - actual).abs().max().item()
    print(f"RAE/DINOv2 parity max_abs={max_abs:.10g} cosine={cosine:.10g}")
    print(f"RAE/DINOv2 autocast max_abs={autocast_max_abs:.10g}")
    assert max_abs <= 1e-5
    assert cosine >= 0.999999
