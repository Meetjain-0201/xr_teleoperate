"""Latch the last commanded arm pose and torque for Quest's scene-reset pause.

The Quest loop owns applying this command. This module never starts a controller
or publishes to the robot.
"""


class ArmHold:
    def __init__(self):
        self._command = None

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

    def release(self):
        self._command = None
