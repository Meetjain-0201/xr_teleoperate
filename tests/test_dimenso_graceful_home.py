"""Offline state-machine checks; no SDK import, controller or DDS construction."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

PATH = Path(__file__).parents[1] / 'teleop/utils/dimenso_graceful_home.py'

def module():
    assert PATH.exists(), 'explicit bounded graceful home is missing'
    spec = importlib.util.spec_from_file_location('dimenso_graceful_home', PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

class Rig:
    def __init__(self, *, stuck=False, stale=False, hot=False, publish=True):
        self.t = 1.
        self.q = np.full(14, .2)
        self.target = self.q.copy()
        self.stuck, self.stale, self.hot, self.publish = stuck, stale, hot, publish
        self.commands, self.weights = [], []
        self.sequence = 0
        self.last_weight = 1.
    def read(self):
        return {'q': self.q.copy(), 'received_at': 0 if self.stale else self.t,
                'temperatures': [125 if self.hot else 30] * 14, 'faults': [0] * 14}
    def command(self, q):
        self.commands.append(np.array(q)); self.target = np.array(q)
    def weight(self, w):
        self.weights.append(w)
        if self.publish:
            self.sequence += 1; self.last_weight = w
    def published(self):
        return self.sequence, self.last_weight
    def sleep(self, dt):
        self.t += dt
        if not self.stuck:
            self.q = self.target.copy()
    def run(self, **kw):
        return module().graceful_home(read=self.read, command=self.command,
            set_weight=self.weight, published=self.published,
            clock=lambda: self.t, sleep=self.sleep, **kw)

def test_home_waits_for_measured_pose_then_published_zero_weight():
    rig = Rig()
    result = rig.run()
    assert result['completed'] and result['homeReached'] and result['handbackPublished']
    assert rig.weights[-1] == 0
    assert np.max(np.abs(rig.q)) < .05
    assert max(np.max(np.abs(b-a)) for a,b in zip(rig.commands,rig.commands[1:])) <= .011

def test_stuck_joints_cannot_report_success_or_release():
    rig = Rig(stuck=True)
    result = rig.run(timeout=2)
    assert not result['completed'] and not result['handbackPublished']
    assert not rig.weights
    assert 'timeout' in result['reason'].lower()

def test_stale_state_refuses_before_any_motion():
    rig = Rig(stale=True)
    result = rig.run()
    assert not result['completed'] and not rig.commands and not rig.weights
    assert 'stale' in result['reason'].lower()

def test_overheated_state_refuses_before_any_motion():
    rig = Rig(hot=True)
    result = rig.run()
    assert not result['completed'] and not rig.commands and not rig.weights
    assert 'temperature' in result['reason'].lower()

def test_missing_publisher_proof_does_not_claim_handback():
    rig = Rig(publish=False)
    result = rig.run(timeout=5)
    assert result['homeReached'] and not result['completed'] and not result['handbackPublished']

def test_nonfinite_position_refuses_before_motion():
    rig = Rig(); rig.q[3] = float('nan')
    result = rig.run()
    assert not result['completed'] and not rig.commands

def test_fault_during_return_interrupts_trajectory():
    rig = Rig()
    original = rig.read
    def read():
        state = original()
        if rig.t > 1.1: state['faults'][2] = 1
        return state
    rig.read = read
    result = rig.run()
    assert not result['completed'] and not rig.weights
    assert 'fault' in result['reason'].lower()

def controller_methods():
    import ast
    import threading
    import types
    source = Path(__file__).parents[1] / 'teleop/robot_control/robot_arm.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'G1_29_ArmController')
    names = {'ctrl_dual_arm_graceful_disconnect', '_recovery_snapshot', '_set_recovery_weight', '_recovery_publication'}
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), 'G1 adapter must wire bounded recovery and publication evidence'
    env = {'np':np,'G1_29_JointArmIndex':range(14),'G1_29_JointIndex':types.SimpleNamespace(kNotUsedJoint0=29)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    return env, threading

def test_debug_mode_cannot_claim_arm_sdk_handback():
    import types
    env,_ = controller_methods()
    robot = types.SimpleNamespace(motion_mode=False)
    result = env['ctrl_dual_arm_graceful_disconnect'](robot)
    assert not result['completed'] and not result['handbackPublished']

def test_recovery_snapshot_carries_sensor_age_and_max_motor_temperature():
    import types
    env,_ = controller_methods()
    data=types.SimpleNamespace(received_at=3.5,motor_state=[types.SimpleNamespace(q=.1,temperature=[40,125],motorstate=0) for _ in range(14)])
    robot=types.SimpleNamespace(lowstate_buffer=types.SimpleNamespace(GetData=lambda:data))
    result=env['_recovery_snapshot'](robot)
    assert result['received_at']==3.5 and result['temperatures']==[125]*14

def test_publication_evidence_requires_successful_sdk_write():
    import ast
    import threading
    import types
    import pytest
    source = Path(__file__).parents[1] / 'teleop/robot_control/robot_arm.py'
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='G1_29_ArmController')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_ctrl_motor_state')
    def stop(_): raise StopIteration
    env={'np':np,'time':types.SimpleNamespace(time=lambda:1.,monotonic=lambda:1.,sleep=stop),
         'G1_29_JointArmIndex':range(14),'G1_29_JointIndex':types.SimpleNamespace(kNotUsedJoint0=29)}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),env)
    for success in [False, True]:
        robot=types.SimpleNamespace(motion_mode=False, simulation_mode=True,
            ctrl_lock=threading.Lock(),q_target=np.zeros(14),tauff_target=np.zeros(14),
            msg=types.SimpleNamespace(motor_cmd=[types.SimpleNamespace(q=0.,dq=0.,tau=0.) for _ in range(35)]),
            crc=types.SimpleNamespace(Crc=lambda _:0),
            lowcmd_publisher=types.SimpleNamespace(Write=lambda _, ok=success:ok),
            _publish_sequence=0,_published_blend=None,_speed_gradual_max=False,control_dt=.004,
            _recovery_snapshot=lambda:{'q':np.zeros(14),'received_at':1.})
        with pytest.raises(StopIteration): env['_ctrl_motor_state'](robot)
        assert robot._publish_sequence == int(success)
        assert robot._published_blend == (0. if success else None)

def test_drift_away_from_home_during_handback_cannot_complete():
    rig=Rig(); original=rig.read
    def read():
        state=original()
        if rig.weights: state['q']=np.ones(14)
        return state
    rig.read=read
    result=rig.run()
    assert not result['completed'] and not result['handbackPublished']
    assert not result['homeReached']
    assert rig.weights[-1] > 0

def test_publisher_refuses_nonfinite_feedback_before_any_write():
    import ast, threading, types, pytest
    source=Path(__file__).parents[1]/'teleop/robot_control/robot_arm.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='G1_29_ArmController')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'_ctrl_motor_state','clip_arm_q_target'}]
    def stop(_): raise StopIteration
    env={'np':np,'time':types.SimpleNamespace(time=lambda:1.,monotonic=lambda:1.,sleep=stop),
         'logger_mp':types.SimpleNamespace(error=lambda *a:None),
         'G1_29_JointArmIndex':range(14),'G1_29_JointIndex':types.SimpleNamespace(kNotUsedJoint0=29)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    writes=[]
    robot=types.SimpleNamespace(motion_mode=False,simulation_mode=False,ctrl_lock=threading.Lock(),
        q_target=np.zeros(14),tauff_target=np.zeros(14),arm_velocity_limit=20,control_dt=.004,
        msg=types.SimpleNamespace(motor_cmd=[types.SimpleNamespace(q=0.,dq=0.,tau=0.) for _ in range(35)]),
        crc=types.SimpleNamespace(Crc=lambda _:0),lowcmd_publisher=types.SimpleNamespace(Write=lambda m:writes.append(m) or True),
        get_current_dual_arm_q=lambda:np.full(14,np.nan),
        _recovery_snapshot=lambda:{'q':np.full(14,np.nan),'received_at':1.},
        _publish_sequence=0,_published_blend=None,_speed_gradual_max=False)
    robot.clip_arm_q_target=types.MethodType(env['clip_arm_q_target'],robot)
    with pytest.raises(StopIteration):env['_ctrl_motor_state'](robot)
    assert writes==[]

def test_recovery_publisher_bounds_catchup_after_stall_and_speed_ramp():
    """Exercise the actual publisher against a lagging, then released fake arm."""
    import ast, threading, types, pytest
    source=Path(__file__).parents[1]/'teleop/robot_control/robot_arm.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='G1_29_ArmController')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'_ctrl_motor_state','clip_arm_q_target'}]
    now=[10.]; measured=[np.full(14, .8)]; writes=[]
    def write(msg):
        command=np.array([m.q for m in msg.motor_cmd[:14]])
        writes.append((measured[0].copy(), command))
        # Three frames of stall, then follow the received command.
        if len(writes)>3: measured[0]=command.copy()
        return True
    def sleep(dt):
        now[0]+=dt
        if len(writes)>=8: raise StopIteration
    env={'np':np,'time':types.SimpleNamespace(time=lambda:now[0],monotonic=lambda:now[0],sleep=sleep),
         'logger_mp':types.SimpleNamespace(error=lambda *a:None),
         'G1_29_JointArmIndex':range(14),'G1_29_JointIndex':types.SimpleNamespace(kNotUsedJoint0=29)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
    robot=types.SimpleNamespace(motion_mode=True,simulation_mode=False,ctrl_lock=threading.Lock(),
        q_target=np.zeros(14),tauff_target=np.zeros(14),arm_velocity_limit=30.,control_dt=.004,
        msg=types.SimpleNamespace(motor_cmd=[types.SimpleNamespace(q=0.,dq=0.,tau=0.) for _ in range(35)]),
        crc=types.SimpleNamespace(Crc=lambda _:0),lowcmd_publisher=types.SimpleNamespace(Write=write),
        _recovery_snapshot=lambda:{'q':measured[0].copy(),'received_at':now[0]},
        _publish_sequence=0,_published_blend=None,_speed_gradual_max=True,_gradual_start_time=0.,_recovery_active=True)
    robot.clip_arm_q_target=types.MethodType(env['clip_arm_q_target'],robot)
    with pytest.raises(StopIteration): env['_ctrl_motor_state'](robot)
    assert robot.arm_velocity_limit==30.  # positive control: ordinary ramp is active
    assert len(writes)==8
    assert measured[0][0]<.8  # positive control: arm eventually moves
    for actual,command in writes:
        assert np.max(np.abs(command-actual))<=.5*robot.control_dt+1e-12
