"""Per-task arm home pose for the Quest client (DIM-752).

Upstream homes both arms to all-zero joints (elbows bent ~90 deg on the G1). A task
can declare its own 14-joint home (left arm 7, then right arm 7, in
G1_29_JointArmIndex order); the backend passes it as `--home-q`. This module only
parses and validates; it never commands the robot.
"""

import numpy as np

ARM_JOINTS = 14


def parse_home_q(text):
    """`None`/empty -> zeros (upstream home). Otherwise 14 comma-separated finite radians."""
    if text is None or not str(text).strip():
        return np.zeros(ARM_JOINTS)
    try:
        q = np.array([float(v) for v in str(text).split(",")], dtype=float)
    except ValueError as exc:
        raise ValueError(f"--home-q must be {ARM_JOINTS} comma-separated numbers: {exc}") from exc
    if q.shape != (ARM_JOINTS,) or not np.all(np.isfinite(q)):
        raise ValueError(f"--home-q must be {ARM_JOINTS} finite numbers, got {q.size}")
    return q
