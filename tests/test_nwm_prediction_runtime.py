import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vlnce_baselines.nwm.assets import validate_external_asset
from vlnce_baselines.nwm.etp_adapter import (
    NwmEtpAdapter,
    RaeEtpAdapterConfig,
    RaeGhostInputRequest,
)
from vlnce_baselines.nwm.predictor import _extract_ema_state, _freeze_for_inference
from vlnce_baselines.nwm.raenwm_core.infer_compat import _sample_time_latent
from vlnce_baselines.nwm.runtime import RaeNwmLatentNormalizer


def _write_stats(path, mean=2.0, var=4.0):
    torch.save(
        {
            "mean": torch.full((768, 1, 1), mean),
            "var": torch.full((768, 1, 1), var),
        },
        path,
    )


def test_external_asset_requires_matching_sha256(tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"rae-nwm")
    expected = hashlib.sha256(b"rae-nwm").hexdigest()

    assert validate_external_asset(path, expected, "test asset") == path.resolve()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_external_asset(path, "0" * 64, "test asset")
    with pytest.raises(ValueError, match="requires a 64-character"):
        validate_external_asset(path, "", "test asset")


def test_latent_normalizer_is_nonpersistent_and_matches_rae_formula(tmp_path):
    stat_path = tmp_path / "stat.pt"
    _write_stats(stat_path)
    normalizer = RaeNwmLatentNormalizer(stat_path)

    latent = torch.full((2, 768, 16, 16), 6.0)
    output = normalizer(latent)

    expected = torch.full_like(output, 4.0 / (4.0 + 1.0e-5) ** 0.5)
    torch.testing.assert_close(output, expected)
    assert normalizer.state_dict() == {}


def test_latent_normalizer_accepts_official_none_mean(tmp_path):
    stat_path = tmp_path / "stat.pt"
    torch.save(
        {
            "mean": None,
            "var": torch.full((768, 16, 16), 4.0),
        },
        stat_path,
    )

    normalizer = RaeNwmLatentNormalizer(stat_path)
    output = normalizer(torch.full((1, 768, 16, 16), 6.0))

    expected = torch.full_like(output, 6.0 / (4.0 + 1.0e-5) ** 0.5)
    torch.testing.assert_close(output, expected)


def test_context_adapter_skips_until_four_frames_and_tracks_query_ids():
    adapter = NwmEtpAdapter(RaeEtpAdapterConfig(context_size=4))
    adapter.reset(1)
    request = RaeGhostInputRequest(
        env_index=0,
        ghost_vp="0_0",
        current_position=np.zeros(3, dtype=np.float32),
        current_yaw=0.0,
        ghost_position=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
    )

    for step in range(3):
        adapter.update_context(
            0,
            rgb=None,
            position=np.asarray([0.0, 0.0, -step], dtype=np.float32),
            yaw=0.0,
            latent=torch.full((768, 16, 16), float(step)),
        )
        batch = adapter.build_raenwm_batch([request])
        assert batch.is_empty
        assert batch.skipped == {"context_not_ready": 1}

    adapter.update_context(
        0,
        rgb=None,
        position=np.asarray([0.0, 0.0, -3.0], dtype=np.float32),
        yaw=0.0,
        latent=torch.full((768, 16, 16), 3.0),
    )
    batch = adapter.build_raenwm_batch([request])

    assert not batch.is_empty
    assert batch.context_latent.shape == (1, 4, 768, 16, 16)
    assert batch.records[0].query_id == "0_0"
    assert batch.records[0].horizon == pytest.approx(1.0 / 0.24975892673356762)
    assert batch.records[0].condition.dtheta == pytest.approx(0.0)
    assert batch.records[0].condition.rel_t == pytest.approx(
        batch.records[0].horizon / 128.0
    )

    adapter.pause_at(0)
    assert adapter.buffers == []


def test_static_run_of_three_is_dropped():
    adapter = NwmEtpAdapter(RaeEtpAdapterConfig(context_size=4, static_run_k=3))
    adapter.reset(1)
    latent = torch.zeros(768, 16, 16)
    position = np.zeros(3, dtype=np.float32)

    assert adapter.update_context(0, None, position, 0.0, latent)
    assert not adapter.update_context(0, None, position, 0.0, latent)
    assert not adapter.update_context(0, None, position, 0.0, latent)
    assert not adapter.update_context(0, None, position, 0.0, latent)

    buffer = adapter.buffers[0]
    assert buffer.num_frames == 1
    assert buffer.stats.dropped_static_runs == 1
    assert buffer.stats.dropped_static_frames == 3


def test_checkpoint_requires_nonempty_ema_and_strips_compile_prefix():
    with pytest.raises(ValueError, match="non-empty 'ema'"):
        _extract_ema_state({})
    tensor = torch.ones(1)
    state = _extract_ema_state({"ema": {"_orig_mod.weight": tensor}})
    assert state == {"weight": tensor}


def test_world_model_is_explicitly_frozen_for_inference():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Dropout(p=0.5),
    ).train()

    returned = _freeze_for_inference(model)

    assert returned is model
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


class _FakeSampler:
    @staticmethod
    def sample_ode(**_kwargs):
        def sample(z, _model, **_model_kwargs):
            return [z + 1.0]

        return sample


def test_explicit_initial_noise_controls_prediction_and_preserves_rng():
    model = torch.nn.Linear(1, 1)
    bundle = SimpleNamespace(
        model=model,
        sampler=_FakeSampler(),
        num_cond=4,
        latent_size=16,
        config={"transport": {"final_only_euler": False}},
        rae=SimpleNamespace(),
    )
    context = torch.zeros(1, 4, 768, 16, 16)
    noise = torch.full((1, 768, 16, 16), 2.0)
    rng_before = torch.random.get_rng_state()

    pred_rgb, pred_latent = _sample_time_latent(
        bundle=bundle,
        x_latent=context,
        curr_delta=torch.zeros(1, 1, 3),
        rel_t=torch.zeros(1),
        return_rgb=False,
        initial_noise=noise,
    )

    assert pred_rgb is None
    torch.testing.assert_close(pred_latent, torch.full_like(noise, 3.0))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
