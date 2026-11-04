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


def test_saved_take_returns_home_slowly_and_waits_for_measured_arrival():
    arm = FakeArm()
    hold = ArmHold()
    hold.capture(arm)
    hold.begin_home()
    q, tau = hold.home_command(np.array([0.4, -0.2]),
                               lambda q: q + 3.0, dt=0.1)
    np.testing.assert_allclose(q, [0.32, -0.12])
    np.testing.assert_allclose(tau, [3.32, 2.88])
    assert hold.home_ready is False
    for _ in range(6):
        q, tau = hold.home_command(np.array([0.2, -0.1]),
                                   lambda q: q + 3.0, dt=0.1)
    np.testing.assert_allclose(q, [0.0, 0.0], atol=1e-7)
    assert hold.home_ready is False
    for _ in range(3):
        hold.home_command(np.array([0.01, -0.01]),
                          lambda q: q + 3.0, dt=0.1)
    assert hold.home_ready is True
    hold.release()
    assert hold.home_ready is False
