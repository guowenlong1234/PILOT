"""Real frozen DINO output parity when unused hidden states are omitted."""
from pathlib import Path

import pytest
import torch

from vlnce_baselines.models.encoders.rae_dinov2_encoder import RaeDinov2RgbEncoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires actual training GPU')
def test_omitting_intermediate_hidden_states_preserves_all_final_tokens():
    model_path = Path('pretrained/rae_dinov2_with_registers_base')
    if not (model_path/'model.safetensors').exists():
        pytest.skip('project-owned DINO assets unavailable')
    encoder = RaeDinov2RgbEncoder(model_path, torch.device('cuda')).cuda().eval()
    generator = torch.Generator().manual_seed(20260920)
    pixels = torch.rand((3,3,224,224), generator=generator).cuda()
    with torch.no_grad(), torch.autocast('cuda'):
        full = encoder.backbone(pixels, output_hidden_states=True).last_hidden_state
        lean = encoder.backbone(pixels, output_hidden_states=False).last_hidden_state
    torch.testing.assert_close(full, lean, rtol=0, atol=0)
    assert all(not p.requires_grad for p in encoder.backbone.parameters())
