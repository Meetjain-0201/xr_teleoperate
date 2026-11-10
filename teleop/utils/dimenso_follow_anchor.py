"""Clutch controller deltas onto measured robot wrists; never publish commands.

The main loop owns follow edges and FK. TeleVuer supplies controller world poses
in robot axes. This module freezes the head yaw for each engagement and resets
IK history so paused controller motion cannot leak into resumed commands.
"""
import numpy as np


def _pose(value):
    p=np.asarray(value,dtype=float)
    if (p.shape!=(4,4) or not np.isfinite(p).all() or
            not np.allclose(p[3],[0,0,0,1],atol=1e-5) or
            not np.allclose(p[:3,:3].T@p[:3,:3],np.eye(3),atol=1e-3) or
            not np.isclose(np.linalg.det(p[:3,:3]),1,atol=1e-3)):
        raise ValueError('follow reference requires finite rigid poses')
    return p.copy()


class FollowAnchor:
    def __init__(self):
        self.clear()

    def clear(self):
        self.controllers=None
        self.robot=None
        self.basis=None

    @property
    def active(self):
        return self.controllers is not None

    def capture(self, controllers, robot, head):
        c=tuple(_pose(p) for p in controllers)
        e=tuple(_pose(p) for p in robot)
        if len(c)!=2 or len(e)!=2:
            raise ValueError('both arm references are required')
        h=_pose(head); x=h[:3,0].copy();x[2]=0
        if np.linalg.norm(x)<1e-6:raise ValueError('head yaw is undefined')
        x/=np.linalg.norm(x);z=np.array([0.,0.,1.])
        self.basis=np.column_stack([x,np.cross(z,x),z])
        self.controllers=c;self.robot=e

    def targets(self, controllers):
        if not self.active:raise RuntimeError('capture references before following')
        result=[]
        for c,c0,e0 in zip(controllers,self.controllers,self.robot):
            c=_pose(c);p=e0.copy();b=self.basis
            p[:3,3]=e0[:3,3]+b.T@(c[:3,3]-c0[:3,3])
            p[:3,:3]=b.T@c[:3,:3]@c0[:3,:3].T@b@e0[:3,:3]
            result.append(p)
        return tuple(result)


def reset_ik_filter(ik,q):
    q=np.asarray(q,dtype=float).copy()
    if not np.isfinite(q).all():raise ValueError('invalid measured joint reference')
    ik.init_data=q.copy()
    ik.smooth_filter.reset(q)


def measured_wrist_poses(ik,q):
    """FK in the exact reduced model and EE frames used by the IK objective."""
    import pinocchio as pin
    data=ik.reduced_robot.model.createData()
    pin.framesForwardKinematics(ik.reduced_robot.model,data,q)
    return (data.oMf[ik.L_hand_id].homogeneous.copy(),
            data.oMf[ik.R_hand_id].homogeneous.copy())


def tracking_ready(tele_data, max_age=.5, *, require_head=True):
    """Only validated current receiver samples may establish/follow an anchor."""
    receipt=getattr(tele_data,'controller_diagnostics',None) or {}
    age=receipt.get('sourceAgeSeconds')
    head_age=getattr(tele_data,'head_source_age_seconds',None)
    try:
        for p in (tele_data.left_controller_world_pose, tele_data.right_controller_world_pose, tele_data.head_pose):
            _pose(p)
    except (AttributeError, TypeError, ValueError):
        return False
    return (bool(tele_data.motion_data_ready) and
            bool(getattr(tele_data,'controller_poses_valid',False)) and
            isinstance(age,(int,float)) and np.isfinite(age) and 0<=age<=max_age and
            (not require_head or (isinstance(head_age,(int,float)) and
                                  np.isfinite(head_age) and 0<=head_age<=max_age)))


def controller_command(anchor,ik,data,q,dq,*,following,gravity,fk=measured_wrist_poses):
    """One follow tick; pause skips IK entirely and every resume recaptures FK."""
    if not following:
        anchor.clear()
        return q.copy(),gravity(ik,q)
    controllers=(data.left_controller_world_pose,data.right_controller_world_pose)
    if not anchor.active:
        anchor.capture(controllers,fk(ik,q),data.head_pose)
        reset_ik_filter(ik,q)
        return q.copy(),gravity(ik,q)
    return ik.solve_ik(*anchor.targets(controllers),q,dq)
