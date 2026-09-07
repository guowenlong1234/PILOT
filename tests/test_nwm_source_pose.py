import math

import numpy as np
import pytest
import torch

from vlnce_baselines.nwm.etp_adapter import (
    NwmEtpAdapter, RaeEtpAdapterConfig, RaeGhostInputRequest,
)


def _adapter(source="context_last", last_yaw=0.0):
    adapter = NwmEtpAdapter(RaeEtpAdapterConfig(condition_source_pose=source))
    adapter.reset(1)
    for i in range(4):
        adapter.update_context(
            0, None, np.array([0.0, 0.0, -i * 0.25]), last_yaw,
            latent=torch.zeros(257, 768),
        )
    return adapter


def _request(position=(0.0, 0.0, -0.75), yaw=math.pi / 2):
    return RaeGhostInputRequest(
        env_index=0, ghost_vp="g", current_position=np.array(position),
        current_yaw=yaw, ghost_position=np.array([0.0, 0.0, -1.75]),
    )


def test_collision_turn_uses_last_observed_image_coordinate_frame():
    adapter = _adapter()
    batch = adapter.build_raenwm_batch([_request()])
    record = batch.records[0]
    assert record.local_dx_m == pytest.approx(1.0)
    assert record.local_dy_m == pytest.approx(0.0)
    assert record.condition.dtheta == pytest.approx(0.0)
    assert adapter.source_pose_totals["yaw_mismatch_queries"] == 1
    assert adapter.source_pose_totals["position_mismatch_queries"] == 0


def test_explicit_legacy_mode_reproduces_query_current_reference():
    adapter = _adapter("query_current")
    record = adapter.build_raenwm_batch([_request()]).records[0]
    assert record.local_dx_m == pytest.approx(0.0, abs=1e-6)
    assert record.local_dy_m == pytest.approx(-1.0)
    assert record.condition.dtheta == pytest.approx(-math.pi / 2)


def test_position_mismatch_preserves_absolute_target_camera_heading():
    adapter = _adapter()
    query = _request(position=(1.0, 0.0, -0.75), yaw=0.0)
    record = adapter.build_raenwm_batch([query]).records[0]
    # Target view faces southwest from the real query, even though the target
    # lies due south of the older image. Do not silently change the target view.
    assert record.local_dx_m == pytest.approx(1.0)
    assert record.local_dy_m == pytest.approx(0.0)
    assert record.condition.dtheta == pytest.approx(math.pi / 4)
    assert adapter.source_pose_totals["position_mismatch_queries"] == 1


def test_equal_source_pose_keeps_old_condition_and_context_exactly():
    fixed, legacy = _adapter(), _adapter("query_current")
    query = _request(yaw=0.0)
    new = fixed.build_raenwm_batch([query])
    old = legacy.build_raenwm_batch([query])
    torch.testing.assert_close(new.condition_tensor, old.condition_tensor, rtol=0, atol=0)
    torch.testing.assert_close(new.context_latent, old.context_latent, rtol=0, atol=0)
    assert fixed.source_pose_totals["mismatch_queries"] == 0


def test_invalid_source_contract_is_rejected():
    with pytest.raises(ValueError, match="condition_source_pose"):
        RaeEtpAdapterConfig(condition_source_pose="unknown")
