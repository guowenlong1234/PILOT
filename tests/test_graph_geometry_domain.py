import numpy as np
import pytest
from vlnce_baselines.models.graph_utils import calculate_vp_rel_pos_fts


@pytest.mark.parametrize('b', [(2.,0.,0.),(-2.,0.,0.),(0.,0.,2.),(0.,0.,-2.)])
def test_axis_roundoff_cannot_make_angles_nan(monkeypatch,b):
    original=np.sqrt
    # Reproduce a one-ULP underestimate from the distance calculation.
    monkeypatch.setattr(np,'sqrt',lambda x:np.nextafter(original(x),0.))
    with np.errstate(invalid='raise'):
        result=calculate_vp_rel_pos_fts((0.,0.,0.),b)
    assert np.isfinite(result).all()


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_nonboundary_geometry_keeps_original_values(dtype):
    a=np.array([1.,2.,3.],dtype=dtype);b=np.array([4.,0.,1.],dtype=dtype)
    dx,dy,dz=b-a
    xz=max(np.sqrt(dx**2+dz**2),1e-8);xyz=max(np.sqrt(dx**2+dy**2+dz**2),1e-8)
    expected=(np.arcsin(-dx/xz),np.arcsin(dz/xyz),xyz)
    assert np.array_equal(calculate_vp_rel_pos_fts(a,b),expected)


def test_coincident_positions_are_finite():
    assert np.array_equal(calculate_vp_rel_pos_fts((0.,0.,0.),(0.,0.,0.)),(0.,0.,1e-8))
