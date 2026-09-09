import numpy as np
import pytest
import torch

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


def test_panorama_history_accepts_raw_events_without_unused_latent_encoding():
    from vlnce_baselines.nwm.runtime import NwmPredictionRuntime
    from vlnce_baselines.nwm.etp_adapter import NwmEtpAdapter
    from vlnce_baselines.nwm.low_level_context import LOW_LEVEL_CONTEXT_SOURCE
    runtime = object.__new__(NwmPredictionRuntime)
    runtime.panorama_mode = 'world_exact_select'
    runtime.context_source = LOW_LEVEL_CONTEXT_SOURCE
    runtime.adapter = NwmEtpAdapter()
    runtime._panorama_frame_counter = 0
    runtime.reset(1)
    events = [{'type': 'reset', 'env_index': 0}]
    for i in range(4):
        events.append(dict(type='frame', env_index=0, frame_index=i,
            position=[0., 0., -i*.25], yaw=0., rgb=np.full((224,224,3),i,np.uint8),
            panorama=dict(format='observed_panorama_v1',cube_rgb=np.zeros((6,8,8,3),np.uint8),
                native_world_rgb12=np.zeros((12,224,224,3),np.uint8))))
    # No normalizer/encoder exists on this object: invoking one would fail.
    runtime.apply_low_level_context_events(events,raw_patch_latents=None,raw_cls=None)
    assert runtime.adapter.buffers[0].is_ready()
    assert len(runtime.panorama_histories[0].frames) == 4
    for i,f in enumerate(runtime.adapter.buffers[0].get_context()):
        assert f.latent is None and np.all(f.rgb == i)
    runtime.apply_low_level_context_events([events[0]],raw_patch_latents=None,raw_cls=None)
    assert not runtime.adapter.buffers[0].is_ready()
    assert not runtime.panorama_histories[0].frames


@pytest.mark.skipif(not torch.cuda.is_available(),reason='GPU pixel parity requires CUDA')
def test_gpu_projection_is_pixel_exact_for_native_and_arbitrary_headings():
    from collections import OrderedDict
    from vlnce_baselines.nwm.panorama_runtime import PanoramaPredictionRuntime,ObservedPanoramaFrame
    from vlnce_baselines.nwm.panorama_context import observed_view
    runtime=object.__new__(PanoramaPredictionRuntime)
    runtime.device=torch.device('cuda:0');runtime.mode='world_exact_select'
    runtime._projection_cache=OrderedDict()
    rng=np.random.default_rng(73);views=[]
    for i,yaw in enumerate([0,np.pi/6,.327,-2.172,np.pi,2*np.pi-.01]):
        f=ObservedPanoramaFrame(str(i),'a',[0,0,0],0,
            rng.integers(0,256,(6,224,224,3),dtype=np.uint8),
            rng.integers(0,256,(12,224,224,3),dtype=np.uint8))
        views.append((f,yaw))
    expected=np.stack([observed_view(f.cube_rgb,yaw,f.native_world_rgb12) for f,yaw in views])
    for _ in range(2):
        np.testing.assert_array_equal(runtime._observed_rgb_batch(views).cpu().numpy(),expected)
