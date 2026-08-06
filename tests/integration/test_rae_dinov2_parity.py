import hashlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ETPNAV_ROOT = Path(
    os.environ.get("ETPNAV_ROOT", "/home/gwl/project/ETPNav/ETPNav")
)
MODEL_DIR = PROJECT_ROOT / "pretrained" / "rae_dinov2_with_registers_base"
REFERENCE_CONFIG = ETPNAV_ROOT / (
    "vlnce_baselines/nwm/raenwm_core/RAE/configs/stage1/pretrained/"
    "DINOv2-B.yaml"
)
REFERENCE_MODEL_DIR = Path(
    os.environ.get(
        "RAE_DINOV2_MODEL_DIR",
        "/home/gwl/project/RAE-NWM/raenwm/models/encoders/"
        "dinov2-with-registers-base",
    )
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ETPNAV_REFERENCE_SCRIPT = textwrap.dedent(
    r"""
    import inspect
    import sys
    from pathlib import Path

    import torch

    etpnav_root = Path(sys.argv[1]).resolve()
    reference_config = Path(sys.argv[2]).resolve()
    image_path = Path(sys.argv[3]).resolve()
    output_path = Path(sys.argv[4]).resolve()

    assert sys.dont_write_bytecode
    sys.path.insert(0, str(etpnav_root))
    from vlnce_baselines.common import rae_visual_encoder

    rgb = torch.load(image_path, map_location="cpu")
    encoder = rae_visual_encoder.RaeDinov2RgbEncoder(
        device=torch.device("cuda:0"),
        output_size=768,
        rae_config_path=str(reference_config),
        cls_residual_mlp_enabled=True,
        cls_residual_mlp_hidden_dim=768,
        cls_residual_mlp_zero_init=True,
    )
    encoder.eval()
    actual = encoder({"rgb": rgb})
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        autocast_actual = encoder({"rgb": rgb})
    backbone = encoder.rae.encoder.encoder
    torch.save(
        {
            "features": actual.cpu(),
            "autocast_features": autocast_actual.cpu(),
            "source": str(
                Path(inspect.getsourcefile(rae_visual_encoder)).resolve()
            ),
            "backbone_training": backbone.training,
            "all_backbone_parameters_frozen": all(
                not parameter.requires_grad
                for parameter in backbone.parameters()
            ),
            "final_layernorm_affine": backbone.layernorm.elementwise_affine,
            "final_layernorm_has_weight": backbone.layernorm.weight is not None,
            "final_layernorm_has_bias": backbone.layernorm.bias is not None,
            "residual_last_weight_nonzero": int(
                torch.count_nonzero(
                    encoder.cls_residual_mlp.layers[-1].weight
                )
            ),
            "residual_last_bias_nonzero": int(
                torch.count_nonzero(
                    encoder.cls_residual_mlp.layers[-1].bias
                )
            ),
        },
        output_path,
    )
    """
)


ETPR1_PRODUCTION_SCRIPT = textwrap.dedent(
    r"""
    import inspect
    import sys
    from pathlib import Path

    import torch

    project_root = Path(sys.argv[1]).resolve()
    model_dir = Path(sys.argv[2]).resolve()
    image_path = Path(sys.argv[3]).resolve()
    output_path = Path(sys.argv[4]).resolve()

    assert sys.dont_write_bytecode
    sys.path.insert(0, str(project_root))
    from vlnce_baselines.models.encoders import rae_dinov2_encoder

    rgb = torch.load(image_path, map_location="cpu")
    encoder = rae_dinov2_encoder.RaeDinov2RgbEncoder(
        model_dir=model_dir,
        device=torch.device("cuda:0"),
        cls_residual_mlp_enabled=True,
        cls_residual_mlp_hidden_dim=768,
        cls_residual_mlp_zero_init=True,
    )
    encoder.eval()
    actual = encoder({"rgb": rgb})
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        autocast_actual = encoder({"rgb": rgb})
    backbone = encoder.backbone
    torch.save(
        {
            "features": actual.cpu(),
            "autocast_features": autocast_actual.cpu(),
            "source": str(
                Path(inspect.getsourcefile(rae_dinov2_encoder)).resolve()
            ),
            "backbone_training": backbone.training,
            "all_backbone_parameters_frozen": all(
                not parameter.requires_grad
                for parameter in backbone.parameters()
            ),
            "final_layernorm_affine": backbone.layernorm.elementwise_affine,
            "final_layernorm_has_weight": backbone.layernorm.weight is not None,
            "final_layernorm_has_bias": backbone.layernorm.bias is not None,
            "residual_last_weight_nonzero": int(
                torch.count_nonzero(
                    encoder.cls_residual_mlp.layers[-1].weight
                )
            ),
            "residual_last_bias_nonzero": int(
                torch.count_nonzero(
                    encoder.cls_residual_mlp.layers[-1].bias
                )
            ),
            "has_latent_mean": hasattr(encoder, "latent_mean"),
            "has_latent_var": hasattr(encoder, "latent_var"),
        },
        output_path,
    )
    """
)


def _run_isolated(script, arguments, cwd):
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script, *map(str, arguments)],
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_production_encoder_matches_etpnav_navigation_encoder(tmp_path):
    assert ETPNAV_ROOT.is_dir(), f"missing read-only ETPNav: {ETPNAV_ROOT}"
    assert REFERENCE_CONFIG.is_file(), (
        f"missing ETPNav RAE stage-1 config: {REFERENCE_CONFIG}"
    )
    for asset_name in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ):
        reference_asset = REFERENCE_MODEL_DIR / asset_name
        production_asset = MODEL_DIR / asset_name
        assert reference_asset.is_file(), (
            f"missing ETPNav reference asset: {reference_asset}"
        )
        assert production_asset.is_file(), (
            f"missing ETP-R1 DINO asset: {production_asset}"
        )
        assert _sha256(reference_asset) == _sha256(production_asset), (
            f"ETPNav and ETP-R1 {asset_name} hashes differ"
        )
    assert torch.cuda.is_available(), "real ETPNav parity requires CUDA"

    generator = torch.Generator(device="cpu").manual_seed(20260710)
    rgb = torch.randint(
        0,
        256,
        (1, 224, 224, 3),
        dtype=torch.uint8,
        generator=generator,
    )
    image_path = tmp_path / "rgb.pt"
    reference_path = tmp_path / "etpnav.pt"
    production_path = tmp_path / "etpr1.pt"
    torch.save(rgb, image_path)

    _run_isolated(
        ETPNAV_REFERENCE_SCRIPT,
        (ETPNAV_ROOT, REFERENCE_CONFIG, image_path, reference_path),
        ETPNAV_ROOT,
    )
    _run_isolated(
        ETPR1_PRODUCTION_SCRIPT,
        (PROJECT_ROOT, MODEL_DIR, image_path, production_path),
        PROJECT_ROOT,
    )

    reference = torch.load(reference_path, map_location="cpu")
    production = torch.load(production_path, map_location="cpu")
    assert reference["source"] == str(
        ETPNAV_ROOT / "vlnce_baselines/common/rae_visual_encoder.py"
    )
    assert production["source"] == str(
        PROJECT_ROOT
        / "vlnce_baselines/models/encoders/rae_dinov2_encoder.py"
    )
    for output in (reference, production):
        assert output["backbone_training"] is False
        assert output["all_backbone_parameters_frozen"] is True
        assert output["final_layernorm_affine"] is False
        assert output["final_layernorm_has_weight"] is False
        assert output["final_layernorm_has_bias"] is False
        assert output["residual_last_weight_nonzero"] == 0
        assert output["residual_last_bias_nonzero"] == 0

    expected = reference["features"].float()
    actual = production["features"].float()
    autocast_expected = reference["autocast_features"].float()
    autocast_actual = production["autocast_features"].float()
    assert production["has_latent_mean"] is False
    assert production["has_latent_var"] is False
    assert expected.shape == actual.shape == (1, 768)
    assert production["features"].dtype == torch.float32
    assert reference["autocast_features"].dtype == torch.float32
    assert production["autocast_features"].dtype == torch.float32
    torch.testing.assert_close(expected, actual, rtol=0.0, atol=1.0e-5)
    torch.testing.assert_close(
        autocast_expected,
        autocast_actual,
        rtol=0.0,
        atol=1.0e-5,
    )
    cosine = F.cosine_similarity(expected, actual).item()
    max_abs = (expected - actual).abs().max().item()
    print(f"ETPNav parity max_abs={max_abs:.10g} cosine={cosine:.10g}")
    assert cosine >= 0.999999
