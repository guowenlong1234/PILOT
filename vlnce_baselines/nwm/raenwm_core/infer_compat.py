from contextlib import contextmanager

import torch

from vlnce_baselines.nwm.raenwm_core.models import (
    latent_to_patch_map,
    make_latent_noise,
    pack_cls_patch,
)


@contextmanager
def _scoped_tf32(device):
    if torch.device(device).type != "cuda":
        yield
        return
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _sample_time_latent(
    *,
    bundle,
    x_latent,
    curr_delta,
    rel_t,
    return_rgb,
    generator=None,
    initial_noise=None,
):
    x_latent = x_latent.to(next(bundle.model.parameters()).device)
    y = curr_delta.to(x_latent.device)
    with _scoped_tf32(x_latent.device), torch.cuda.amp.autocast(
        enabled=x_latent.device.type == "cuda",
        dtype=torch.bfloat16,
    ):
        batch_size = x_latent.shape[0]
        if int(x_latent.shape[1]) < int(bundle.num_cond):
            raise ValueError(
                f"NWM context has {x_latent.shape[1]} frames, "
                f"requires {bundle.num_cond}"
            )
        native_cls = bool(bundle.config.get("predict_cls_token", False))
        expected_rank = 4 if native_cls else 5
        if x_latent.ndim != expected_rank:
            contract = "[B,T,257,768]" if native_cls else "[B,T,768,16,16]"
            raise ValueError(
                f"NWM context must have shape {contract}, got {tuple(x_latent.shape)}"
            )
        x_cond = x_latent[:, : bundle.num_cond]
        expected_noise_shape = (
            (batch_size, bundle.latent_size * bundle.latent_size + 1, x_latent.shape[-1])
            if native_cls
            else (
                batch_size,
                x_latent.shape[2],
                bundle.latent_size,
                bundle.latent_size,
            )
        )
        if initial_noise is None:
            z = make_latent_noise(
                batch_size,
                x_latent.shape[-1] if native_cls else x_latent.shape[2],
                bundle.latent_size,
                x_latent.device,
                dtype=x_latent.dtype if native_cls else None,
                predict_cls_token=native_cls,
                generator=generator,
            )
        else:
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    "initial_noise has shape "
                    f"{tuple(initial_noise.shape)}, expected {expected_noise_shape}"
                )
            if not torch.isfinite(initial_noise).all():
                raise FloatingPointError("initial_noise contains NaN or infinity")
            z = initial_noise.to(
                device=x_latent.device,
                dtype=x_latent.dtype if native_cls else torch.float32,
            )
        sample_fn = bundle.sampler.sample_ode(
            sampling_method=str(
                bundle.config.get("transport", {}).get("sampling_method", "euler")
            ),
            num_steps=int(bundle.config.get("transport", {}).get("num_steps", 50)),
            atol=1e-6,
            rtol=1e-3,
            reverse=False,
            final_only=bool(
                bundle.config.get("transport", {}).get(
                    "final_only_euler", False
                )
            ),
        )
        sample = sample_fn(
            z, bundle.model, y=y.flatten(0, 1), x_cond=x_cond, rel_t=rel_t
        )
        if bool(
            bundle.config.get("transport", {}).get("final_only_euler", False)
        ):
            pred_latent = torch.nan_to_num(sample)
        else:
            pred_latent = torch.nan_to_num(sample[-1])
        if not return_rgb:
            return None, pred_latent
        decode_latent = latent_to_patch_map(pred_latent, bundle.latent_size)
        pred_rgb = torch.nan_to_num(bundle.rae.decode(decode_latent).float()).clamp(
            0.0, 1.0
        )
        return pred_rgb, pred_latent


@torch.no_grad()
def model_forward_time(
    *,
    bundle,
    obs_image,
    curr_delta,
    rel_t,
    return_rgb,
    generator=None,
    initial_noise=None,
):
    x = obs_image.to(next(bundle.model.parameters()).device)
    with torch.cuda.amp.autocast(enabled=x.device.type == "cuda", dtype=torch.bfloat16):
        batch_size, context_len = x.shape[:2]
        x_flat = x.flatten(0, 1)
        x_pix = x_flat * 0.5 + 0.5
        if bool(bundle.config.get("predict_cls_token", False)):
            cls_latent, patch_latent = bundle.rae.encode(
                x_pix, return_cls_token=True
            )
            x_latent = pack_cls_patch(cls_latent, patch_latent).unflatten(
                0, (batch_size, context_len)
            )
        else:
            x_latent = bundle.rae.encode(x_pix).unflatten(
                0, (batch_size, context_len)
            )
    return _sample_time_latent(
        bundle=bundle,
        x_latent=x_latent,
        curr_delta=curr_delta,
        rel_t=rel_t,
        return_rgb=return_rgb,
        generator=generator,
        initial_noise=initial_noise,
    )


@torch.no_grad()
def model_forward_time_from_latents(
    *,
    bundle,
    context_latent,
    curr_delta,
    rel_t,
    return_rgb,
    generator=None,
    initial_noise=None,
):
    return _sample_time_latent(
        bundle=bundle,
        x_latent=context_latent,
        curr_delta=curr_delta,
        rel_t=rel_t,
        return_rgb=return_rgb,
        generator=generator,
        initial_noise=initial_noise,
    )
