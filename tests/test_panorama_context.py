import math

import numpy as np
import pytest

from vlnce_baselines.nwm.panorama_context import (
    cube_rotations, local_xy, make_context_plan, materialize_context,
    perspective_from_cube, wrap)


def colored_rays_cube(n=64):
    axis=2*(np.arange(n)+.5)/n-1
    x,y=np.meshgrid(axis,-axis)
    rays=np.stack([x,y,-np.ones_like(x)],-1)
    cubes=[]
    for rot in cube_rotations():
        world=rays@rot.T;world/=np.linalg.norm(world,axis=-1,keepdims=True)
        cubes.append(np.rint((world+1)*127.5).astype(np.uint8))
    return np.stack(cubes)


@pytest.mark.parametrize('yaw',[0,.25,math.pi/2,math.pi,1.5*math.pi,2*math.pi-.01])
def test_reprojection_preserves_world_rays(yaw):
    from vlnce_baselines.nwm.panorama_context import yaw_matrix
    n=64;cube=colored_rays_cube(n)
    out=perspective_from_cube(cube,yaw,size=n)
    axis=2*(np.arange(n)+.5)/n-1;x,y=np.meshgrid(axis,-axis)
    rays=np.stack([x,y,-np.ones_like(x)],-1)@yaw_matrix(yaw).T
    rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    expected=(rays+1)*127.5
    assert np.abs(out-expected).max()<2.1


def test_cube_identity_is_exact():
    rng=np.random.default_rng(4);cube=rng.integers(0,256,(6,32,32,3),dtype=np.uint8)
    np.testing.assert_array_equal(perspective_from_cube(cube,0,size=32),cube[0])
    np.testing.assert_array_equal(perspective_from_cube(cube,np.pi/2,size=32),cube[1])


def test_reversed_anchor_preserves_target_and_recomputes_all_conditions():
    pos=np.array([[0,0,-i*.25] for i in range(4)])
    target=np.array([0.,0.,.5]);yaw=np.pi
    plan=make_context_plan(pos,np.zeros(4),target,yaw,'world_exact_select')
    assert plan.order==(3,2,1,0) and plan.source_index==0
    assert plan.target_position==tuple(target) and plan.target_yaw==yaw
    assert plan.delta[0]>0 and plan.delta[2]==pytest.approx(0)
    assert plan.rel_t==pytest.approx(.5/.24975892673356762/128)
    np.testing.assert_array_equal(pos[-1],[0,0,-.75])


def test_world_alignment_removes_body_anchor_rotation_without_relabeling_rgb():
    pos=np.array([[0,0,-i*.25] for i in range(4)])
    plan=make_context_plan(pos,[0,2,2,2],[1,0,-1],2.5,'world30')
    assert len(set(plan.view_yaws))==1
    assert abs(plan.delta[2])<=math.pi/12+1e-8
    cubes=np.stack([colored_rays_cube() for _ in range(4)])
    views=materialize_context(cubes,plan)
    for view in views[1:]:np.testing.assert_array_equal(view,views[0])


def test_target_between_history_endpoints_falls_back_and_keeps_coverage():
    pos=np.array([[0,0,-i*.25] for i in range(4)])
    plan=make_context_plan(pos,np.zeros(4),[0,0,-.4],0,'world_exact_select')
    assert plan.fallback and plan.source_index==3


def test_cross_segment_and_teleport_rejected():
    pos=np.array([[0,0,-i*.25] for i in range(4)])
    with pytest.raises(ValueError,match='segments'):
        make_context_plan(pos,np.zeros(4),[0,0,-2],0,'front',[1,1,2,2])
    pos[0,0]=3
    with pytest.raises(ValueError,match='discontinuous'):
        make_context_plan(pos,np.zeros(4),[0,0,-2],0,'front')


@pytest.mark.parametrize('yaw',[-3,-1,0,.7,3])
def test_condition_round_trip_world_target(yaw):
    pos=np.array([[1,0,-i*.25] for i in range(4)])
    target=np.array([1.4,0,-1.8]);plan=make_context_plan(pos,[yaw]*4,target,yaw+.8,'world30_select')
    f,l=np.array(plan.delta[:2])*.24975892673356762*64
    beta=plan.source_yaw
    dx=-math.sin(beta)*f-math.cos(beta)*l
    dz=-math.cos(beta)*f+math.sin(beta)*l
    recovered=np.array(plan.source_position)+[dx,0,dz]
    np.testing.assert_allclose(recovered,target,atol=1e-12)
    assert float(wrap(beta+plan.delta[2]-plan.target_yaw))==pytest.approx(0)
