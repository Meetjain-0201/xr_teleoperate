"""The waist is brought upright after arm authority is taken, not frozen where it was."""

import numpy as np
import pytest

from teleop.utils.dimenso_waist_upright import WaistUpright


def test_bent_waist_reaches_upright_linearly_then_stays():
    waist = WaistUpright(start_q=[0.0, 0.01, 0.151], duration_s=3.0, control_dt=0.004)
    first = waist.next()
    assert first[2] < 0.151 and first[2] > 0.150
    for _ in range(749):
        q = waist.next()
    np.testing.assert_allclose(q, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(waist.next(), [0.0, 0.0, 0.0], atol=1e-12)
    assert waist.done


def test_rate_never_exceeds_start_over_duration():
    waist = WaistUpright(start_q=[0.2, -0.1, 0.35], duration_s=3.0, control_dt=0.004)
    prev = np.array([0.2, -0.1, 0.35])
    worst = 0.0
    while not waist.done:
        q = waist.next()
        worst = max(worst, float(np.max(np.abs(q - prev))) / 0.004)
        prev = q
    assert worst <= 0.35 / 3.0 + 1e-9


def test_upright_start_is_a_no_op():
    waist = WaistUpright(start_q=[0.0, 0.0, 0.0], duration_s=3.0, control_dt=0.004)
    np.testing.assert_allclose(waist.next(), [0.0, 0.0, 0.0])


def test_refuses_a_start_pose_that_is_not_three_finite_angles():
    with pytest.raises(ValueError):
        WaistUpright(start_q=[0.0, 0.1], duration_s=3.0, control_dt=0.004)
    with pytest.raises(ValueError):
        WaistUpright(start_q=[0.0, float("nan"), 0.1], duration_s=3.0, control_dt=0.004)
