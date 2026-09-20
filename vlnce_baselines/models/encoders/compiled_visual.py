"""Optional frozen-visual compilation with a dynamic batch and fixed image size."""
import torch


def compile_visual_backbone(module, input_key=None):
    """Compile forward without adding a wrapper module or changing state keys.

    PyTorch 2.2 antialiased interpolation requires concrete spatial sizes.
    Only dimension zero varies as navigation episodes finish.
    """
    if hasattr(module, '_visual_eager_forward'):
        return
    original = module.forward
    compiled = torch.compile(original, dynamic=True)

    def forward(*args, **kwargs):
        value = args[0] if args else kwargs['observations' if input_key else 'pixel_values']
        tensor = value[input_key] if input_key else value
        torch._dynamo.mark_static(tensor, list(range(1, tensor.ndim)))
        return compiled(*args, **kwargs)

    module._visual_eager_forward = original
    module.forward = forward
