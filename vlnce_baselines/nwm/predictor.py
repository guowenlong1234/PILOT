import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import numpy as np

from vlnce_baselines.nwm.paths import NWM_CONFIG_DIR, resolve_repo_path
from vlnce_baselines.nwm.types import NwmPrediction


# eval_config.yaml is the NWM eval base config in the mainline config tree.
DEFAULT_EVAL_CONFIG = NWM_CONFIG_DIR / "eval_config.yaml"


def _freeze_for_inference(model):
    """Put a migrated prediction module in eval mode and freeze its parameters."""
    model.requires_grad_(False)
    return model.eval()


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_merged_config(config_path) -> Dict[str, Any]:
    base = _load_yaml(DEFAULT_EVAL_CONFIG)
    user = _load_yaml(resolve_repo_path(config_path))
    return _deep_update(base, user)


def default_head_checkpoint_path(config: Dict[str, Any]):
    return config.get("head_checkpoint_path") or config.get("heads", {}).get(
        "checkpoint_path"
    )


def resolve_existing_local_or_keep(path_value, *base_dirs):
    if path_value is None:
        return None
    raw = str(path_value)
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    for base_dir in base_dirs:
        local = Path(base_dir) / candidate
        if local.exists():
            return str(local)
    repo_local = resolve_repo_path(candidate)
    if repo_local.exists():
        return str(repo_local)
    return raw


def _extract_ema_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "NWM checkpoint must be a dict containing a non-empty 'ema' state."
        )
    state = checkpoint.get("ema")
    if not isinstance(state, dict) or not state:
        raise ValueError(
            "NWM checkpoint is missing a non-empty 'ema' state; "
            "refusing to run with random weights."
        )
    return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def _validate_loaded_state(model, state, incompatible):
    model_keys = set(model.state_dict())
    loaded_keys = set(state) & model_keys
    if (
        not loaded_keys
        or incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        missing_preview = list(incompatible.missing_keys[:5])
        unexpected_preview = list(incompatible.unexpected_keys[:5])
        raise ValueError(
            "NWM checkpoint EMA does not match CDiT model; "
            f"matched={len(loaded_keys)} missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)} "
            f"missing_preview={missing_preview} "
            f"unexpected_preview={unexpected_preview}"
        )
    return loaded_keys


def compose_delta_sequence(delta_seq):
    import torch

    if delta_seq.dim() != 3:
        raise ValueError(
            f"delta_seq must have shape [B, T, D], got {tuple(delta_seq.shape)}"
        )
    batch, steps, dims = delta_seq.shape
    if dims < 3:
        return delta_seq.sum(dim=1)

    x = torch.zeros(batch, device=delta_seq.device, dtype=delta_seq.dtype)
    y = torch.zeros_like(x)
    th = torch.zeros_like(x)
    for step in range(steps):
        dx = delta_seq[:, step, 0]
        dy = delta_seq[:, step, 1]
        dth = delta_seq[:, step, 2]
        c = torch.cos(th)
        s = torch.sin(th)
        x = x + c * dx - s * dy
        y = y + s * dx + c * dy
        th = th + dth
    two_pi = 2.0 * math.pi
    th = th - two_pi * torch.floor((th + math.pi) / two_pi)
    delta_se2 = torch.stack([x, y, th], dim=-1)
    if dims > 3:
        return torch.cat([delta_se2, delta_seq[:, :, 3:].sum(dim=1)], dim=-1)
    return delta_se2


def manual_condition_to_tensors(condition, device):
    import torch

    curr_delta = torch.tensor(
        [[[float(condition.dx), float(condition.dy), float(condition.dtheta)]]],
        dtype=torch.float32,
        device=device,
    )
    rel_t = torch.tensor([float(condition.rel_t)], dtype=torch.float32, device=device)
    return curr_delta, rel_t


def load_manual_context_images(context_images):
    import torch
    from PIL import Image

    from vlnce_baselines.nwm.raenwm_core.misc import transform

    frames = []
    for path in context_images:
        with Image.open(path) as image:
            frames.append(transform(image.convert("RGB")))
    return torch.stack(frames, dim=0).unsqueeze(0)


def build_prediction_meta(
    *, checkpoint_path, return_rgb, enable_decoder, num_steps, rel_t
):
    return {
        "checkpoint_path": str(checkpoint_path),
        "return_rgb": bool(return_rgb),
        "enable_decoder": bool(enable_decoder),
        "num_steps": int(num_steps),
        "rel_t": float(rel_t),
    }


def resize_prediction_to_context(pred_rgb, context_image):
    if pred_rgb is None:
        return None
    target_size = tuple(context_image.shape[-2:])
    if tuple(pred_rgb.shape[-2:]) == target_size:
        return pred_rgb
    import torch.nn.functional as F

    return F.interpolate(
        pred_rgb,
        size=target_size,
        mode="bicubic",
        align_corners=False,
    )


@dataclass
class ModelBundle:
    model: Any
    transport: Any
    sampler: Any
    rae: Any
    config: Dict[str, Any]
    latent_size: int
    num_cond: int


class ExternalContextLatentRae:
    def __init__(self, latent_dim: int):
        self.latent_dim = int(latent_dim)

    def encode(self, *_args, **_kwargs):
        raise RuntimeError(
            "This NWM predictor was initialized for navigation-provided context "
            "latents, so the RAE image encoder is not available."
        )

    def decode(self, *_args, **_kwargs):
        raise RuntimeError(
            "RAE decoder is not available when using navigation-provided context latents."
        )


class RaeNwmPredictor:
    def __init__(
        self,
        config_path,
        checkpoint_path,
        device="cuda",
        enable_decoder=False,
        torch_compile=False,
        lazy_init=False,
        num_steps=None,
        final_only_euler=False,
        use_external_context_latents=False,
    ):
        if enable_decoder:
            raise ValueError(
                "The migrated Stage-0 runtime is prediction-only and has no RGB decoder"
            )
        if not use_external_context_latents:
            raise ValueError(
                "The migrated Stage-0 runtime requires navigation-provided patch latents"
            )
        self.config_path = resolve_repo_path(config_path)
        self.checkpoint_path = resolve_repo_path(checkpoint_path)
        self.device = str(device)
        self.enable_decoder = bool(enable_decoder)
        self.torch_compile = bool(torch_compile)
        self.use_external_context_latents = bool(use_external_context_latents)
        self.config = load_merged_config(self.config_path)
        if num_steps is not None:
            self.config.setdefault("transport", {})
            self.config["transport"]["num_steps"] = int(num_steps)
        self.config.setdefault("transport", {})
        self.config["transport"]["final_only_euler"] = bool(final_only_euler)
        self.bundle: Optional[ModelBundle] = None
        if not lazy_init:
            self.bundle = self._load_bundle()

    def _validate_return_rgb(self, return_rgb: bool) -> None:
        if return_rgb and not self.enable_decoder:
            raise ValueError("return_rgb=True requires enable_decoder=True")

    def _load_bundle(self) -> ModelBundle:
        import torch

        from vlnce_baselines.nwm.raenwm_core.RAE.src.stage2.transport.transport import (
            ModelType,
            PathType,
            Sampler,
            Transport,
            WeightType,
        )
        from vlnce_baselines.nwm.raenwm_core.models import CDiT_models

        device = torch.device(self.device)
        config = self.config
        latent_size = int(config["image_size"]) // 14
        num_cond = int(config["context_size"])
        if self.use_external_context_latents and not self.enable_decoder:
            latent_dim = int(
                config.get("latent_dim")
                or config.get("heads", {}).get("input_dim", 768)
            )
            rae = ExternalContextLatentRae(latent_dim)
        else:
            from vlnce_baselines.nwm.raenwm_core.RAE.src.stage1.rae import RAE
            from vlnce_baselines.nwm.raenwm_core.RAE.src.utils.model_utils import (
                instantiate_from_config,
            )
            from vlnce_baselines.nwm.raenwm_core.RAE.src.utils.train_utils import (
                parse_configs,
            )

            rae_config_path = resolve_repo_path(
                config.get(
                    "config_path",
                    "vlnce_baselines/nwm/raenwm_core/RAE/configs/stage1/pretrained/DINOv2-B.yaml",
                )
            )
            rae_config, *_ = parse_configs(str(rae_config_path))
            if "params" not in rae_config:
                rae_config["params"] = {}
            rae_config["target"] = (
                "vlnce_baselines.nwm.raenwm_core.RAE.src.stage1.rae.RAE"
            )
            rae_config["params"]["enable_decoder"] = bool(self.enable_decoder)
            for path_key in (
                "pretrained_decoder_path",
                "normalization_stat_path",
                "decoder_config_path",
            ):
                if (
                    path_key in rae_config["params"]
                    and rae_config["params"][path_key] is not None
                ):
                    rae_config["params"][path_key] = resolve_existing_local_or_keep(
                        rae_config["params"][path_key],
                        rae_config_path.parent,
                    )
            encoder_params = rae_config["params"].get("encoder_params", {})
            if "dinov2_path" in encoder_params:
                encoder_params["dinov2_path"] = resolve_existing_local_or_keep(
                    encoder_params["dinov2_path"],
                    rae_config_path.parent,
                )
            if "encoder_config_path" in rae_config["params"]:
                rae_config["params"]["encoder_config_path"] = resolve_existing_local_or_keep(
                    rae_config["params"]["encoder_config_path"],
                    rae_config_path.parent,
                )

            rae: RAE = instantiate_from_config(rae_config).to(device).eval()
        model_kwargs = {
            "context_size": num_cond,
            "input_size": latent_size,
            "in_channels": rae.latent_dim,
            "learn_sigma": bool(config.get("learn_sigma", False)),
            "head_width": config.get("head_width", rae.latent_dim),
            "head_depth": int(config.get("head_depth", 2)),
            "head_num_heads": int(config.get("head_num_heads", 16)),
        }
        model = CDiT_models[config["model"]](**model_kwargs)
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        state = _extract_ema_state(checkpoint)
        incompatible = model.load_state_dict(state, strict=False)
        loaded_keys = _validate_loaded_state(model, state, incompatible)
        print(
            "[NWM] State load: "
            f"matched={len(loaded_keys)} "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
        model = _freeze_for_inference(model).to(device)
        if self.torch_compile:
            model = torch.compile(model)

        transport_config = config.get("transport", {})
        token_count = int(latent_size) * int(latent_size)
        if bool(config.get("predict_cls_token", False)):
            token_count += 1
        shift_dim = int(rae.latent_dim) * token_count
        shift_base = float(transport_config.get("time_dist_shift_base", 4096))
        time_dist_shift = math.sqrt(float(shift_dim) / float(shift_base))
        if transport_config.get("time_dist_shift") is not None:
            time_dist_shift = float(transport_config["time_dist_shift"])
        if bool(transport_config.get("time_dist_shift_disable", False)):
            time_dist_shift = 1.0

        transport = Transport(
            model_type=getattr(
                ModelType, str(transport_config.get("model_type", "velocity")).upper()
            ),
            path_type=getattr(
                PathType, str(transport_config.get("path_type", "linear")).upper()
            ),
            loss_type=getattr(
                WeightType, str(transport_config.get("loss_type", "velocity")).upper()
            ),
            time_dist_type=str(transport_config.get("time_dist_type", "uniform")),
            time_dist_shift=time_dist_shift,
            train_eps=1e-3,
            sample_eps=1e-3,
        )
        sampler = Sampler(transport)
        return ModelBundle(
            model=model,
            transport=transport,
            sampler=sampler,
            rae=rae,
            config=config,
            latent_size=latent_size,
            num_cond=num_cond,
        )

    def _require_bundle(self) -> ModelBundle:
        if self.bundle is None:
            self.bundle = self._load_bundle()
        return self.bundle

    def _predict_time_from_tensors(
        self,
        obs_image,
        curr_delta,
        rel_t,
        return_rgb=False,
        generator=None,
        initial_noise=None,
    ):
        from vlnce_baselines.nwm.raenwm_core.infer_compat import model_forward_time

        bundle = self._require_bundle()
        pred_rgb, pred_latent = model_forward_time(
            bundle=bundle,
            obs_image=obs_image[:, -bundle.num_cond :].to(self.device),
            curr_delta=curr_delta.to(self.device),
            rel_t=rel_t.to(self.device),
            return_rgb=return_rgb,
            generator=generator,
            initial_noise=initial_noise,
        )
        pred_rgb = resize_prediction_to_context(pred_rgb, obs_image)
        return pred_rgb, pred_latent

    def _predict_time_from_latents(
        self,
        context_latent,
        curr_delta,
        rel_t,
        return_rgb=False,
        generator=None,
        initial_noise=None,
    ):
        from vlnce_baselines.nwm.raenwm_core.infer_compat import (
            model_forward_time_from_latents,
        )

        bundle = self._require_bundle()
        pred_rgb, pred_latent = model_forward_time_from_latents(
            bundle=bundle,
            context_latent=context_latent[:, -bundle.num_cond :].to(self.device),
            curr_delta=curr_delta.to(self.device),
            rel_t=rel_t.to(self.device),
            return_rgb=return_rgb,
            generator=generator,
            initial_noise=initial_noise,
        )
        return pred_rgb, pred_latent

    def predict_time_from_dataset_batch(
        self, batch, horizons, return_rgb=False
    ) -> NwmPrediction:
        self._validate_return_rgb(return_rgb)

        _idxs, obs_image, _gt_image, delta = batch
        if len(horizons) != 1:
            raise ValueError("First implementation expects exactly one horizon.")
        horizon = int(horizons[0])
        if horizon < 1 or horizon > delta.shape[1]:
            raise ValueError(
                f"horizon must satisfy 1 <= horizon <= {delta.shape[1]}, got {horizon}"
            )
        import torch

        bundle = self._require_bundle()
        delta_comp = compose_delta_sequence(delta[:, :horizon].to(self.device))
        curr_delta = delta_comp.unsqueeze(1)
        rel_t = torch.full(
            (obs_image.shape[0],), float(horizon) / 128.0, device=self.device
        )
        pred_rgb, pred_latent = self._predict_time_from_tensors(
            obs_image=obs_image,
            curr_delta=curr_delta,
            rel_t=rel_t,
            return_rgb=return_rgb,
        )
        meta = build_prediction_meta(
            checkpoint_path=self.checkpoint_path,
            return_rgb=return_rgb,
            enable_decoder=self.enable_decoder,
            num_steps=int(bundle.config.get("transport", {}).get("num_steps", 50)),
            rel_t=float(rel_t[0].item()),
        )
        meta["horizon"] = horizon
        return NwmPrediction(pred_latent=pred_latent, pred_rgb=pred_rgb, meta=meta)

    def predict_time_from_manual_condition(
        self, context_images, condition, return_rgb=False
    ) -> NwmPrediction:
        self._validate_return_rgb(return_rgb)
        expected_context = int(self.config["context_size"])
        if len(context_images) != expected_context:
            raise ValueError(
                f"manual prediction requires exactly {expected_context} context images, "
                f"got {len(context_images)}"
            )

        bundle = self._require_bundle()
        obs_image = load_manual_context_images(context_images).to(self.device)
        curr_delta, rel_t = manual_condition_to_tensors(condition, self.device)
        pred_rgb, pred_latent = self._predict_time_from_tensors(
            obs_image=obs_image,
            curr_delta=curr_delta,
            rel_t=rel_t,
            return_rgb=return_rgb,
        )
        meta = build_prediction_meta(
            checkpoint_path=self.checkpoint_path,
            return_rgb=return_rgb,
            enable_decoder=self.enable_decoder,
            num_steps=int(bundle.config.get("transport", {}).get("num_steps", 50)),
            rel_t=float(rel_t[0].item()),
        )
        meta.update(condition.as_dict())
        return NwmPrediction(pred_latent=pred_latent, pred_rgb=pred_rgb, meta=meta)


class RaeNwmHeadPredictor(RaeNwmPredictor):
    def __init__(
        self,
        config_path,
        checkpoint_path,
        device="cuda",
        enable_decoder=False,
        torch_compile=False,
        lazy_init=False,
        num_steps=None,
        final_only_euler=False,
        use_external_context_latents=False,
        head_config=None,
        head_checkpoint_path=None,
        strict_heads=True,
        heads_trainable=False,
        token_head_trainable=True,
        confidence_head_trainable=True,
        head_state_dict_override=None,
    ):
        super().__init__(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            enable_decoder=enable_decoder,
            torch_compile=torch_compile,
            lazy_init=lazy_init,
            num_steps=num_steps,
            final_only_euler=final_only_euler,
            use_external_context_latents=use_external_context_latents,
        )
        resolved_head_checkpoint_path = head_checkpoint_path
        if resolved_head_checkpoint_path is None and head_config is None:
            resolved_head_checkpoint_path = default_head_checkpoint_path(self.config)
        if resolved_head_checkpoint_path:
            self.head_checkpoint_path = resolve_repo_path(resolved_head_checkpoint_path)
        else:
            self.head_checkpoint_path = None
        self.strict_heads = bool(strict_heads)
        self.heads_random_init = self.head_checkpoint_path is None
        self.heads_trainable = bool(heads_trainable)
        self.token_head_trainable = bool(token_head_trainable)
        self.confidence_head_trainable = bool(confidence_head_trainable)
        self.head_state_dict_override = head_state_dict_override
        self.heads = self._build_heads(head_config)

    def _build_heads(self, head_config):
        import torch

        from vlnce_baselines.nwm.heads import NwmHeadConfig, NwmOutputHeads

        checkpoint = None
        config_data = dict(self.config.get("heads", {}))
        if self.head_checkpoint_path is not None:
            checkpoint = torch.load(
                self.head_checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            checkpoint_config = checkpoint.get("head_config")
            if checkpoint_config is not None:
                config_data = dict(checkpoint_config)
        if head_config is not None:
            if isinstance(head_config, NwmHeadConfig):
                config_data.update(head_config.to_dict())
            else:
                config_data.update(dict(head_config))
        heads = NwmOutputHeads(NwmHeadConfig.from_mapping(config_data)).to(self.device)
        if checkpoint is not None:
            state_dict = checkpoint.get("state_dict", checkpoint)
            heads.load_state_dict(state_dict, strict=self.strict_heads)
        if self.head_state_dict_override is not None:
            heads.load_state_dict(
                self.head_state_dict_override,
                strict=self.strict_heads,
            )
        if self.heads_trainable:
            heads.set_trainable(
                self.token_head_trainable,
                self.confidence_head_trainable,
            )
            heads.train()
        else:
            heads.set_trainable(False, False)
            heads.eval()
        return heads

    def _load_manual_obs_image(self, context_images):
        return load_manual_context_images(context_images).to(self.device)

    def _condition_tensor_from_manual(self, condition):
        import torch

        return torch.tensor(
            [
                [
                    float(condition.dx),
                    float(condition.dy),
                    float(condition.dtheta),
                    float(condition.rel_t),
                ]
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _build_head_prediction(self, pred_latent, condition_tensor, pred_rgb, meta):
        from vlnce_baselines.nwm.raenwm_core.infer_compat import _scoped_tf32

        with _scoped_tf32(pred_latent.device):
            if self.heads_trainable:
                outputs = self.heads(pred_latent, condition_tensor)
            else:
                import torch

                with torch.no_grad():
                    outputs = self.heads(pred_latent, condition_tensor)
        meta = dict(meta)
        head_checkpoint_path = None
        if self.head_checkpoint_path is not None:
            head_checkpoint_path = str(self.head_checkpoint_path)
        meta["head_checkpoint_path"] = head_checkpoint_path
        meta["heads_random_init"] = bool(self.heads_random_init)
        meta["head_state_overridden"] = self.head_state_dict_override is not None
        return NwmPrediction(
            pred_latent=pred_latent,
            pred_rgb=pred_rgb,
            pred_cls=outputs.pred_cls,
            confidence=outputs.confidence,
            conf_logit=outputs.conf_logit,
            meta=meta,
        )

    def predict_time_with_heads_from_manual_condition(
        self,
        context_images,
        condition,
        return_rgb=False,
    ):
        self._validate_return_rgb(return_rgb)
        expected_context = int(self.config["context_size"])
        if len(context_images) != expected_context:
            raise ValueError(
                f"manual prediction requires exactly {expected_context} context images, "
                f"got {len(context_images)}"
            )
        obs_image = self._load_manual_obs_image(context_images)
        curr_delta, rel_t = manual_condition_to_tensors(condition, self.device)
        pred_rgb, pred_latent = self._predict_time_from_tensors(
            obs_image=obs_image,
            curr_delta=curr_delta,
            rel_t=rel_t,
            return_rgb=return_rgb,
        )
        condition_tensor = self._condition_tensor_from_manual(condition)
        meta = build_prediction_meta(
            checkpoint_path=self.checkpoint_path,
            return_rgb=return_rgb,
            enable_decoder=self.enable_decoder,
            num_steps=int(self.config.get("transport", {}).get("num_steps", 50)),
            rel_t=float(condition.rel_t),
        )
        meta.update(condition.as_dict())
        return self._build_head_prediction(
            pred_latent,
            condition_tensor,
            pred_rgb,
            meta,
        )

    def predict_time_with_heads_from_dataset_batch(
        self,
        batch,
        horizons,
        return_rgb=False,
    ):
        self._validate_return_rgb(return_rgb)
        _idxs, obs_image, _gt_image, delta = batch
        if len(horizons) != 1:
            raise ValueError("Head predictor expects exactly one horizon.")
        horizon = int(horizons[0])
        if horizon < 1 or horizon > delta.shape[1]:
            raise ValueError(
                f"horizon must satisfy 1 <= horizon <= {delta.shape[1]}, got {horizon}"
            )
        import torch

        delta_comp = compose_delta_sequence(delta[:, :horizon].to(self.device))
        curr_delta = delta_comp.unsqueeze(1)
        rel_t = torch.full(
            (obs_image.shape[0],),
            float(horizon) / 128.0,
            device=self.device,
        )
        pred_rgb, pred_latent = self._predict_time_from_tensors(
            obs_image=obs_image,
            curr_delta=curr_delta,
            rel_t=rel_t,
            return_rgb=return_rgb,
        )
        condition_tensor = torch.cat([delta_comp[:, :3], rel_t.unsqueeze(-1)], dim=-1)
        meta = build_prediction_meta(
            checkpoint_path=self.checkpoint_path,
            return_rgb=return_rgb,
            enable_decoder=self.enable_decoder,
            num_steps=int(self.config.get("transport", {}).get("num_steps", 50)),
            rel_t=float(rel_t[0].item()),
        )
        meta["horizon"] = horizon
        return self._build_head_prediction(
            pred_latent,
            condition_tensor,
            pred_rgb,
            meta,
        )

    def predict_time_with_heads_from_etp_batch(
        self,
        batch,
        return_rgb=False,
        generator=None,
        initial_noise=None,
    ):
        self._validate_return_rgb(return_rgb)
        if batch.is_empty:
            return NwmPrediction(
                pred_latent=None,
                pred_rgb=None,
                pred_cls=None,
                confidence=None,
                conf_logit=None,
                meta={
                    "records": [],
                    "skipped": dict(batch.skipped),
                    "empty": True,
                },
            )

        context_latent = getattr(batch, "context_latent", None)
        if context_latent is not None:
            pred_rgb, pred_latent = self._predict_time_from_latents(
                context_latent=context_latent.to(self.device),
                curr_delta=batch.curr_delta.to(self.device),
                rel_t=batch.rel_t.to(self.device),
                return_rgb=return_rgb,
                generator=generator,
                initial_noise=initial_noise,
            )
            context_source = "nav_latent"
        else:
            pred_rgb, pred_latent = self._predict_time_from_tensors(
                obs_image=batch.context.to(self.device),
                curr_delta=batch.curr_delta.to(self.device),
                rel_t=batch.rel_t.to(self.device),
                return_rgb=return_rgb,
                generator=generator,
                initial_noise=initial_noise,
            )
            context_source = "image"
        condition_tensor = batch.condition_tensor.to(self.device)
        # The adapter created batch.rel_t from these record values as float32.
        # Reconstruct the same metadata on CPU instead of synchronizing CUDA
        # after the world-model forward.
        rel_t_values = [
            float(np.float32(record.condition.rel_t)) for record in batch.records
        ]
        meta = build_prediction_meta(
            checkpoint_path=self.checkpoint_path,
            return_rgb=return_rgb,
            enable_decoder=self.enable_decoder,
            num_steps=int(self.config.get("transport", {}).get("num_steps", 50)),
            rel_t=float(rel_t_values[0]),
        )
        meta["rel_t_values"] = [float(value) for value in rel_t_values]
        meta["records"] = list(batch.records)
        meta["skipped"] = dict(batch.skipped)
        meta["empty"] = False
        meta["context_source"] = context_source
        return self._build_head_prediction(
            pred_latent,
            condition_tensor,
            pred_rgb,
            meta,
        )
