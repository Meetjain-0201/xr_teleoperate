"""DIMENSO ADDITION (DIM-927): bring the G1 waist upright after arm authority is taken.

Responsibility: produce the waist [yaw, roll, pitch] command for each rt/arm_sdk tick so the
waist moves linearly from the pose measured at engage to upright (0, 0, 0) over a fixed
duration, then stays upright.

Why: G1_29_ArmController locks every non-arm joint at the pose MEASURED when it starts. On
2026-11-13 that froze the waist 0.151 rad forward for a whole teleop session, so every take
recorded a bent torso (dimenso-context#717); the scoop takes of 2026-11-11 had 0.03-0.05 rad.
Unitree's own references drive the waist to zero instead: unitree_sdk2_python
example/g1/high_level/g1_arm7_sdk_dds_example.py (waist target 0, 0, 0) and upstream
xr_teleoperate 845b25a ("waist initialization": R1 `ctrl_head_and_waist_go_home`, linear to
zero over 3 s at startup).

Collaborators: `robot_control/robot_arm.py` G1_29_ArmController calls `next()` once per
publish tick. Pure: no DDS, no clock.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

UPRIGHT = (0.0, 0.0, 0.0)


class WaistUpright:
    def __init__(self, start_q: Sequence[float], duration_s: float, control_dt: float):
        start = [float(v) for v in start_q]
        if len(start) != 3 or not all(math.isfinite(v) for v in start):
            raise ValueError(f"waist start pose must be 3 finite angles [yaw, roll, pitch], got {list(start_q)}")
        self._start = np.array(start)
        self._goal = np.array(UPRIGHT)
        self._steps = max(1, int(round(float(duration_s) / float(control_dt))))
        self._step = 0

    @property
    def done(self) -> bool:
        return self._step >= self._steps

    def next(self) -> np.ndarray:
        if self._step < self._steps:
            self._step += 1
        frac = self._step / self._steps
        return self._start + (self._goal - self._start) * frac
