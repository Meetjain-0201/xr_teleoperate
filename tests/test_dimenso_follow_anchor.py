"""Offline clutch invariants; no robot imports or publishers."""
import numpy as np
import pytest
from teleop.utils.dimenso_follow_anchor import FollowAnchor, reset_ik_filter


def pose(x=0., yaw=0.):
    c,s=np.cos(yaw),np.sin(yaw)
    p=np.eye(4);p[:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];p[0,3]=x
    return p


def test_engage_and_reengage_hold_current_robot_pose():
    a=FollowAnchor(); robot=(pose(.2,.4),pose(-.3,-.6))
    for ctrl,head in [((pose(2,1),pose(3,-1)),pose(0,.3)),((pose(8,2),pose(9,-2)),pose(0,1.7))]:
        a.capture(ctrl,robot,head)
        for got,want in zip(a.targets(ctrl),robot):np.testing.assert_allclose(got,want,atol=1e-12)
        a.clear()


def test_translation_and_rotation_use_frozen_head_basis():
    a=FollowAnchor(); a.capture((pose(),pose()),(pose(.2),pose(-.2)),pose(0,np.pi/2))
    got=a.targets((pose(.1,np.pi/2),pose()))[0]
    np.testing.assert_allclose(got[:3,3],[.2,-.1,0],atol=1e-12)
    np.testing.assert_allclose(got[:3,:3],pose(0,np.pi/2)[:3,:3],atol=1e-12)


def test_invalid_pose_refused_and_unanchored_cannot_follow():
    a=FollowAnchor()
    with pytest.raises(RuntimeError):a.targets((pose(),pose()))
    bad=pose();bad[0,0]=float('nan')
    with pytest.raises(ValueError):a.capture((bad,pose()),(pose(),pose()),pose())


def test_reset_removes_paused_ik_history():
    from teleop.utils.weighted_moving_filter import WeightedMovingFilter
    from types import SimpleNamespace
    f=WeightedMovingFilter(np.array([.4,.3,.2,.1]),14)
    for n in range(4):f.add_data(np.ones(14)*(n+1))
    ik=SimpleNamespace(smooth_filter=f,init_data=np.ones(14))
    q=np.ones(14)*.1;reset_ik_filter(ik,q)
    f.add_data(q+.01)
    np.testing.assert_allclose(f.filtered_data,q+.01)
    np.testing.assert_allclose(ik.init_data,q)


def test_freshness_requires_both_event_and_head_and_valid_poses():
    from types import SimpleNamespace
    from teleop.utils.dimenso_follow_anchor import tracking_ready
    t=SimpleNamespace(controller_diagnostics={'sourceAgeSeconds':.01}, head_source_age_seconds=.02,
                      controller_poses_valid=True, motion_data_ready=True, head_pose=pose(),
                      left_controller_world_pose=pose(), right_controller_world_pose=pose())
    assert tracking_ready(t)
    t.head_source_age_seconds=1
    assert not tracking_ready(t)
    t.head_source_age_seconds=.01;t.controller_poses_valid=False
    assert not tracking_ready(t)


def test_follow_edges_bypass_paused_solver_and_reanchor_measured_state():
    from types import SimpleNamespace
    from teleop.utils.dimenso_follow_anchor import controller_command
    from teleop.utils.weighted_moving_filter import WeightedMovingFilter
    calls=[]
    ik=SimpleNamespace(smooth_filter=WeightedMovingFilter(np.array([1.]),14),init_data=np.zeros(14),
                       solve_ik=lambda *args:(calls.append(args) or (args[2],np.zeros(14))))
    data=SimpleNamespace(left_controller_world_pose=pose(4),right_controller_world_pose=pose(5),head_pose=pose())
    a=FollowAnchor();q=np.ones(14)*.3
    kw={'gravity':lambda ik,q:np.zeros(14),'fk':lambda ik,q:(pose(q[0]),pose(-q[0]))}
    first,_=controller_command(a,ik,data,q,q,following=True,**kw)
    np.testing.assert_equal(first,q);assert calls==[]
    controller_command(a,ik,data,q,q,following=True,**kw);assert len(calls)==1
    data.left_controller_world_pose=pose(100,2)
    controller_command(a,ik,data,q,q,following=False,**kw);assert len(calls)==1;assert not a.active
    next_q=q+.1
    resumed,_=controller_command(a,ik,data,next_q,q,following=True,**kw)
    np.testing.assert_equal(resumed,next_q);assert len(calls)==1
