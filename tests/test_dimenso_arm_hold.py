"""A paused Quest take keeps one fixed arm command until following resumes."""

from threading import Lock

import numpy as np

from teleop.utils.dimenso_arm_hold import ArmHold


class FakeArm:
    def __init__(self):
        self.ctrl_lock = Lock()
        self.q_target = np.array([0.4, -0.2])
        self.tauff_target = np.array([2.5, -1.5])


def test_pause_latches_commanded_pose_and_supporting_torque():
    arm = FakeArm()
    hold = ArmHold()
    hold.capture(arm)
    arm.q_target[:] = [0.0, 0.0]
    arm.tauff_target[:] = [0.0, 0.0]
    q, tau = hold.command()
    np.testing.assert_array_equal(q, [0.4, -0.2])
    np.testing.assert_array_equal(tau, [2.5, -1.5])
    hold.capture(arm)
    q, tau = hold.command()
    np.testing.assert_array_equal(q, [0.4, -0.2])
    np.testing.assert_array_equal(tau, [2.5, -1.5])
    hold.release()
    hold.capture(arm)
    q, tau = hold.command()
    np.testing.assert_array_equal(q, [0.0, 0.0])
    np.testing.assert_array_equal(tau, [0.0, 0.0])
