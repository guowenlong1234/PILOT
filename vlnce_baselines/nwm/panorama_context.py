"""Target-pose-preserving virtual contexts from observed spherical panoramas.

No simulator or oracle is used here. Six RGB cube faces cover all rays so
arbitrary 90-degree views do not require inventing uncovered polar pixels.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


SPACING = 0.24975892673356762
FORMAT = 'nwm_observed_panorama_virtual_context_v2'


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def yaw_matrix(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def cube_rotations():
    rotations = [yaw_matrix(i * np.pi / 2) for i in range(4)]
    for a in (np.pi / 2, -np.pi / 2):
        c, s = math.cos(a), math.sin(a)
        rotations.append(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))
    return np.stack(rotations)


def cube_quaternions():
    return [[0., math.sin(i*np.pi/4), 0., math.cos(i*np.pi/4)] for i in range(4)] + [
        [math.sin(a/2), 0., 0., math.cos(a/2)] for a in (np.pi/2, -np.pi/2)]


def perspective_from_cube(cube, yaw, size=224, hfov=90.):
    """Reproject real RGB rays; camera looks along -z, with +y up.

    Cube order: yaw 0/90/180/270, pitch +90/-90. Faces are square 90deg.
    Bilinear sampling uses the nearest cube face and clamps half-pixel edges.
    """
    cube = np.asarray(cube)
    if cube.ndim != 4 or cube.shape[0] != 6 or cube.shape[-1] != 3 or cube.shape[1] != cube.shape[2]:
        raise ValueError('cube must be [6,H,H,3] RGB')
    if cube.dtype != np.uint8:
        raise ValueError('cube RGB must be uint8')
    if not np.isfinite(yaw) or not 0 < hfov < 180 or size < 1:
        raise ValueError('invalid perspective geometry')
    face, y0, x0, y1, x1, wx, wy = _perspective_sampling_map(
        cube.shape[1], float(yaw), int(size), float(hfov))
    out = ((1-wx)*(1-wy)*cube[face,y0,x0] + wx*(1-wy)*cube[face,y0,x1]
           + (1-wx)*wy*cube[face,y1,x0] + wx*wy*cube[face,y1,x1])
    return np.rint(out).clip(0,255).astype(np.uint8)


@lru_cache(maxsize=16)
def _perspective_sampling_map(n, yaw, size, hfov):
    """Share exact pixel geometry across the four historical images.

    Keys retain the full heading, resolution and field of view. Only geometry
    is cached, never RGB or a feature; sixteen maps bound CPU memory usage.
    """
    axis = (2*(np.arange(size)+.5)/size-1) * math.tan(math.radians(hfov)/2)
    xx, yy = np.meshgrid(axis, -axis)
    rays = np.stack([xx, yy, -np.ones_like(xx)], -1) @ yaw_matrix(yaw).T
    local = np.einsum('hwc,vcd->vhwd', rays, cube_rotations())
    forward = -local[..., 2]
    face = forward.argmax(0)
    iy, ix = np.indices((size, size))
    chosen = local[face, iy, ix]
    depth = -chosen[..., 2]
    uv = chosen[..., :2] / depth[..., None]
    if np.max(np.abs(uv)) > 1+1e-6:
        raise ValueError('cube did not cover requested rays')
    sx = np.clip((uv[..., 0]+1)*n/2-.5, 0, n-1)
    sy = np.clip((1-uv[..., 1])*n/2-.5, 0, n-1)
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    x1, y1 = np.minimum(x0+1, n-1), np.minimum(y0+1, n-1)
    wx, wy = (sx-x0)[..., None], (sy-y0)[..., None]
    arrays = (face, y0, x0, y1, x1, wx, wy)
    for array in arrays:
        array.flags.writeable = False
    return arrays


def local_xy(source, target, yaw):
    delta = np.asarray(target, dtype=float)-np.asarray(source, dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([-c*delta[2]-s*delta[0], s*delta[2]-c*delta[0]])


def observed_view(cube,yaw,native_world_rgb12=None):
    """Use an actual recorded 30deg view when present, otherwise cube rays."""
    if native_world_rgb12 is not None:
        bank=np.asarray(native_world_rgb12)
        if bank.dtype!=np.uint8 or bank.shape!=(12,224,224,3):
            raise ValueError('native world views must be [12,224,224,3] uint8')
        sector=int(np.floor(float(yaw)/(np.pi/6)+.5))%12
        if abs(float(wrap(yaw-sector*np.pi/6)))<1e-6:
            return bank[sector].copy()
    return perspective_from_cube(cube,yaw)


@dataclass(frozen=True)
class ContextPlan:
    mode: str
    order: tuple
    view_yaws: tuple  # indexed in original chronological frame order
    source_index: int
    source_position: tuple
    source_yaw: float
    target_position: tuple
    target_yaw: float
    delta: tuple
    rel_t: float
    fallback: bool = False


MODES = ('front', 'relative12', 'world30', 'world_exact',
         'world30_reverse', 'world30_select', 'world_exact_select')


def make_context_plan(positions, yaws, target_position, target_yaw, mode,
                      segment_ids=None):
    """Choose observation views and anchor without inspecting target appearance.

    Selection: prefer forward-aligned history AND a target ahead of its final
    camera; break ties by target distance. If neither order is eligible, keep
    chronological target-aligned context. This fallback keeps query coverage.
    """
    pos, ys = np.asarray(positions, float), np.asarray(yaws, float)
    target = np.asarray(target_position, float)
    if pos.shape != (4,3) or ys.shape != (4,) or target.shape != (3,):
        raise ValueError('expected four positions/yaws and one 3D target')
    if not (np.isfinite(pos).all() and np.isfinite(ys).all() and np.isfinite(target).all() and np.isfinite(target_yaw)):
        raise ValueError('non-finite geometry')
    if mode not in MODES:
        raise ValueError('unknown context mode')
    if segment_ids is not None and (len(segment_ids) != 4 or len(set(segment_ids)) != 1):
        raise ValueError('cannot join different movement segments')
    step_lengths = np.linalg.norm(np.diff(pos[:,[0,2]],axis=0),axis=1)
    if np.any(step_lengths > .51):
        raise ValueError('context contains a discontinuous movement step')
    if mode == 'front':
        view_yaws = ys.copy()
    elif mode == 'relative12':
        phi = np.floor(float(wrap(target_yaw-ys[-1]))/(np.pi/6)+.5)*(np.pi/6)
        view_yaws = ys+phi
    else:
        beta = float(target_yaw) if 'exact' in mode else float(np.floor(target_yaw/(np.pi/6)+.5)*(np.pi/6))
        view_yaws = np.full(4,beta)
    source, fallback = 3, False
    if mode.endswith('_reverse'):
        h = np.linalg.norm((target-pos[0])[[0,2]])/SPACING
        if 1 <= h <= 64 and local_xy(pos[0],target,view_yaws[0])[0] > 0:
            source = 0
        else:
            fallback = True
    elif mode.endswith('_select'):
        candidates = []
        for endpoint, order in [(3,[0,1,2,3]),(0,[3,2,1,0])]:
            motion = [local_xy(pos[a],pos[b],view_yaws[a])[0] for a,b in zip(order[:-1],order[1:])]
            displacement = local_xy(pos[endpoint],target,view_yaws[endpoint])
            h = np.linalg.norm(displacement)/SPACING
            if min(motion) >= -1e-4 and np.mean(motion) > .025 and displacement[0] > 0 and 1 <= h <= 64:
                candidates.append((float(np.linalg.norm(displacement)), -endpoint, endpoint))
        if candidates:
            source = min(candidates)[2]
        else:
            fallback = True
    order = (0,1,2,3) if source == 3 else (3,2,1,0)
    xy = local_xy(pos[source],target,view_yaws[source])
    distance_steps = np.linalg.norm(xy)/SPACING
    # Match the original runtime for all modes; report unclipped geometry in
    # diagnostics. Alternate anchors are accepted only inside [1,64] above.
    rel_t = float(np.clip(distance_steps,1,64)/128)
    delta = (float(xy[0]/SPACING/64),float(xy[1]/SPACING/64),float(wrap(target_yaw-view_yaws[source])))
    return ContextPlan(mode,order,tuple(view_yaws),source,tuple(pos[source]),
        float(view_yaws[source]),tuple(target),float(target_yaw),delta,rel_t,fallback)


def materialize_context(cubes, plan, native_world_rgb12=None):
    if len(cubes) != 4:
        raise ValueError('four distinct observed time points are required')
    return np.stack([observed_view(cubes[i],plan.view_yaws[i],
        None if native_world_rgb12 is None else native_world_rgb12[i]) for i in plan.order])
