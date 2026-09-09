import numpy as np

from vlnce_baselines.nwm.panorama_context import (
    _perspective_sampling_map, perspective_from_cube,
)


def test_projection_geometry_reused_without_reusing_image_pixels():
    _perspective_sampling_map.cache_clear()
    a = np.zeros((6, 32, 32, 3), np.uint8)
    b = np.full_like(a, 217)
    assert perspective_from_cube(a, .321, size=32).max() == 0
    assert perspective_from_cube(b, .321, size=32).min() == 217
    assert _perspective_sampling_map.cache_info().hits == 1
    for array in _perspective_sampling_map(32, .321, 32, 90.):
        assert not array.flags.writeable


def test_projection_cache_distinguishes_camera_geometry_and_is_bounded():
    _perspective_sampling_map.cache_clear()
    cube = np.zeros((6, 32, 32, 3), np.uint8)
    for yaw in np.linspace(-3, 3, 30):
        perspective_from_cube(cube, yaw, size=16)
    assert _perspective_sampling_map.cache_info().currsize == 16
    perspective_from_cube(cube, 3, size=32)
    perspective_from_cube(cube, 3, size=32, hfov=80)
    assert _perspective_sampling_map.cache_info().misses == 32
