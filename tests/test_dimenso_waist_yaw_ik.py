"""DIM-752: the wired IK can solve waist yaw (robotics-onboard 9f44ff7's design) without changing its 14-joint API."""

import os
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

TELEOP = Path(__file__).resolve().parents[1] / "teleop"


@pytest.fixture(scope="module")
def iks():
    cwd = os.getcwd()
    os.chdir(TELEOP)          # robot_arm_ik resolves the URDF relative to teleop/
    try:
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
        yield G1_29_ArmIK(waist_yaw=True), G1_29_ArmIK()
    finally:
        os.chdir(cwd)


def _targets(ik, yaw):
    m = ik.reduced_robot.model
    d = m.createData()
    arm = np.array([0.2, 0.2, 0, 0.8, 0, 0, 0, 0.2, -0.2, 0, 0.8, 0, 0, 0])
    pin.framesForwardKinematics(m, d, np.concatenate(([yaw], arm)))
    return d.oMf[m.getFrameId("L_ee")].homogeneous.copy(), d.oMf[m.getFrameId("R_ee")].homogeneous.copy()


def test_waist_yaw_is_one_more_variable_and_the_api_stays_14(iks):
    ik, plain = iks
    assert ik.reduced_robot.model.nq == 15 and plain.reduced_robot.model.nq == 14
    L, R = _targets(ik, 0.0)
    q, tau = ik.solve_ik(L, R, np.zeros(14), np.zeros(14))
    assert q.shape == (14,) and tau.shape == (14,)


def test_the_torso_stays_upright_when_the_arms_reach(iks):
    from teleop.robot_control.robot_arm_ik import G1_29_ArmIK  # noqa: F401
    ik, _ = iks
    ik.seed_waist_yaw(0.0)
    L, R = _targets(ik, 0.0)
    cur = np.zeros(14)
    for _ in range(30):
        cur, _tau = ik.solve_ik(L, R, cur, np.zeros(14))
    assert abs(ik.waist_yaw) < 0.01


def test_the_torso_turns_gradually_when_the_arms_cannot_reach(iks):
    from teleop.robot_control.robot_arm_ik import MAX_WAIST_STEP_RAD
    ik, _ = iks
    ik.seed_waist_yaw(0.0)
    L, R = _targets(ik, 0.6)
    cur, prev, steps = np.zeros(14), 0.0, []
    for _ in range(60):
        cur, _tau = ik.solve_ik(L, R, cur, np.zeros(14))
        steps.append(ik.waist_yaw - prev)
        prev = ik.waist_yaw
    assert ik.waist_yaw > 0.05
    assert max(abs(s) for s in steps) <= MAX_WAIST_STEP_RAD + 1e-6
