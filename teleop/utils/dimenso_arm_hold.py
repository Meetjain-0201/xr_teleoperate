"""Latch the last commanded arm pose and torque for Quest's scene-reset pause.

The Quest loop owns applying this command. This module never starts a controller
or publishes to the robot.
"""

import numpy as np


class ArmHold:
    def __init__(self, home_speed=0.8, home_tolerance=0.05):
        self._command = None
        self._home_target = None
        self._home_arrival_ticks = 0
        self.home_ready = False
        self.home_speed = home_speed
        self.home_tolerance = home_tolerance

    def capture(self, controller):
        """Capture once; later measured drift must not become the new target."""
        if self._command is None:
            with controller.ctrl_lock:
                self._command = (
                    controller.q_target.copy(),
                    controller.tauff_target.copy(),
                )

    def command(self):
        if self._command is None:
            raise RuntimeError("arm hold has not captured a command")
        return self._command

    def begin_home(self):
        """Use the latched command as the ramp origin, never a drifting pose."""
        if self._command is None:
            raise RuntimeError("capture the arm command before returning home")
        if self._home_target is None:
            self._home_target = self._command[0].copy()
            self.home_ready = False
            self._home_arrival_ticks = 0

    @property
    def returning_home(self):
        return self._home_target is not None

    def home_command(self, measured_q, gravity_torque, *, dt):
        """Rate-limit q toward zero and hold computed support torque at each step."""
        if self._home_target is None:
            raise RuntimeError("arm home has not started")
        step = self.home_speed * min(max(dt, 0.0), 0.1)
        self._home_target += np.clip(-self._home_target, -step, step)
        q = self._home_target.copy()
        tau = gravity_torque(q)
        self._command = (q, tau.copy())
        at_home = (np.all(np.isfinite(measured_q))
                   and np.max(np.abs(measured_q)) < self.home_tolerance
                   and np.max(np.abs(q)) < self.home_tolerance)
        self._home_arrival_ticks = self._home_arrival_ticks + 1 if at_home else 0
        self.home_ready = self._home_arrival_ticks >= 3
        return self._command

    def release(self):
        self._command = None
        self._home_target = None
        self._home_arrival_ticks = 0
        self.home_ready = False
