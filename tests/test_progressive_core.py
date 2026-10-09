import math
import pytest
import torch
from vlnce_baselines.nwm.active_lookahead.progressive_core import certify_decision, successor_geometry, validate_budget


def test_budget_and_geometry():
    assert validate_budget() == (.5, .5)
    for kwargs in [dict(deployment_gain=2), dict(residual_clip=1), dict(score_scale='probability'), dict(budget_fractions=(.6,.6)), dict(budget_fractions=(-.1,.5))]:
        with pytest.raises(ValueError): validate_budget(**kwargs)
    assert successor_geometry([0,0,0], math.pi, [0,0,2], math.pi, 2) == pytest.approx([0,2,0,1,2])
    assert successor_geometry([0,0,0], math.pi, [2,0,0], 1.5*math.pi, 2) == pytest.approx([2,0,1,0,2])
    angle=.7
    assert successor_geometry([3,8,2], math.pi+angle, [3+2*math.sin(angle),8,2+2*math.cos(angle)], math.pi+angle,2) == pytest.approx([0,2,0,1,2])


def test_certificate_and_nested_intervals():
    gen=torch.Generator().manual_seed(42)
    for _ in range(100):
        s=torch.randn(7,generator=gen)
        increments=torch.rand(2,7,generator=gen)-.5
        remaining=torch.ones(7)
        original=s.clone()
        for d in range(3):
            cert=certify_decision(s,remaining)
            if cert.certified:
                assert cert.winner_id == int((original+increments.sum(0)).argmax())
            if d==2: break
            next_s=s+increments[d]
            next_r=remaining-.5
            assert torch.all(next_s-next_r >= s-remaining-1e-6)
            assert torch.all(next_s+next_r <= s+remaining+1e-6)
            s,remaining=next_s,next_r
    assert not certify_decision([1.,1.], [0.,0.]).certified
    assert not certify_decision([1.+1e-7,1.], [0.,0.]).certified
    assert certify_decision([0.,3.], [1.,0.]).winner_id == 1
    assert certify_decision([0.,3.], [1.,0.]).certified
