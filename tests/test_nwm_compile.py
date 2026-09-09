"""Compiler contracts: shape changes, retained outputs, and bounded graph memory."""
import pytest
import torch

from vlnce_baselines.nwm.compile_runtime import compile_frozen_world_model


class TinyWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.register_buffer('offset', torch.linspace(-1, 1, 16))

    def forward(self, x, t):
        return self.linear(torch.nn.functional.silu(x)) + t[:, None] * self.offset


def test_compiler_rejects_trainable_and_cpu_models():
    model = TinyWorld()
    with pytest.raises(ValueError, match='frozen eval'):
        compile_frozen_world_model(model)
    model.eval().requires_grad_(False)
    with pytest.raises(ValueError, match='requires CUDA'):
        compile_frozen_world_model(model)
    with pytest.raises(ValueError, match='Unknown'):
        compile_frozen_world_model(model, backend_name='unsupported')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires target-machine GPU')
def test_native_compiler_variable_shapes_values_and_owned_outputs():
    model = TinyWorld().cuda().eval().requires_grad_(False)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    compiled, backend = compile_frozen_world_model(model, max_graphs=2)
    retained = []
    tf32_before = torch.backends.cuda.matmul.allow_tf32
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in [8, 8, 3, 1, 5, 8, 3]:
            x = torch.randn(batch, 16, device='cuda')
            for t_value in [1., .5, .01]:
                t = torch.full((batch,), t_value, device='cuda')
                expected = model(x, t)
                actual = compiled(x, t)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                retained.append((actual, actual.clone()))
                x = actual
            assert len(backend.graphs) <= 2
        for actual, snapshot in retained:
            torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
    assert backend.stats['captures'] > 2
    assert backend.stats['evictions'] > 0
    assert backend.stats['replays'] == 21
    assert backend.stats['compilations'] <= 3
    assert torch.backends.cuda.matmul.allow_tf32 == tf32_before
    for key,value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
