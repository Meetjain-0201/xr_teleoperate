"""Bounded G1 arm home/handback, using injected telemetry and publisher evidence.

No DDS objects are constructed here. The G1 controller supplies fresh motor samples,
commands and publication counters. A transmitted zero blend is not a torque-off
acknowledgment. Limits are conservative software guards, not motor specifications.
"""
import time
import numpy as np


def _checked(sample, now):
    q = np.asarray(sample.get('q'), dtype=float)
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError('Invalid measured arm positions')
    age = now - sample.get('received_at', float('-inf'))
    if not np.isfinite(age) or age < 0 or age > .5:
        raise ValueError('Stale arm feedback; physical recovery required')
    temperatures = np.asarray(sample.get('temperatures'), dtype=float)
    if temperatures.shape != (14,) or not np.isfinite(temperatures).all():
        raise ValueError('Motor temperature unavailable')
    if np.max(temperatures) > 70:
        raise ValueError('Motor temperature exceeds software recovery limit (70 C)')
    faults = np.asarray(sample.get('faults'), dtype=float)
    if faults.shape != (14,) or not np.isfinite(faults).all() or np.any(faults != 0):
        raise ValueError('Motor fault or unknown fault state')
    return q


def graceful_home(*, read, command, set_weight, published, clock=time.monotonic,
                  sleep=time.sleep, timeout=20., velocity=.5):
    """Return only after measured home and a new successful zero-blend write.

    Never release authority after failed home. On an in-flight failure, replace
    the trajectory with the last valid measured pose; report failure and leave
    the controller available. This is not an emergency stop or collision planner.
    """
    result = dict(completed=False, reason=None, homeReached=False,
                  handbackPublished=False)
    last_q = None
    commanded = False
    try:
        if not np.isfinite(timeout) or timeout <= 0 or not 0 < velocity <= .5:
            raise ValueError('Invalid bounded recovery parameters')
        start = clock()
        deadline = start + timeout
        initial = _checked(read(), start)
        last_q = initial.copy()
        duration = max(1.5, float(np.max(np.abs(initial))) / velocity)
        if duration + 2.5 >= timeout:
            raise ValueError('Home trajectory cannot complete within recovery timeout')
        stable = 0
        while clock() < deadline:
            now = clock()
            last_q = _checked(read(), now)
            alpha = min(1., (now - start) / duration)
            command(initial * (1. - alpha))
            commanded = True
            stable = stable + 1 if alpha == 1 and np.max(np.abs(last_q)) < .05 else 0
            if stable >= 3:
                result['homeReached'] = True
                break
            sleep(.02)
        if not result['homeReached']:
            raise ValueError('Home timeout: measured arms did not reach home')
        release_start = clock()
        sequence_before = published()[0]
        while clock() < deadline:
            now = clock()
            last_q = _checked(read(), now)
            if np.max(np.abs(last_q)) >= .05:
                result["homeReached"] = False
                raise ValueError("Arms left home during handback; recovery unverified")
            weight = max(0., 1. - (now - release_start) / 2.)
            set_weight(weight)
            sequence, published_weight = published()
            if weight == 0 and sequence > sequence_before and published_weight == 0:
                result.update(completed=True, handbackPublished=True)
                return result
            sleep(.02)
        raise ValueError('Handback timeout: no successful zero-blend publication')
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        result['reason'] = str(exc)
        if commanded and last_q is not None:
            try:
                command(last_q)
            except Exception as hold_error:
                result['reason'] += f'; hold request failed: {hold_error}'
        return result
