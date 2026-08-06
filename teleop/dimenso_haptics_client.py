"""Collision haptics, CLIENT side -- the P2 vertical slice (DIM-555).

Consumes `rt/dimenso/haptics` (the §3.1 message published by the sim half, see
`../dimenso_haptics.py`) and turns it into the arguments of a Vuer haptic pulse, so the
operator's Quest controller taps when the robot touches something. Contract is
`docs/superpowers/specs/2026-08-06-collision-haptics-design.md`; nothing here redesigns it.

WHERE THIS FILE LIVES, AND WHY THERE ARE TWO COPIES
---------------------------------------------------
The importer is `teleop_hand_and_arm.py` in the operator-side checkout at
`/home/meet/projects/g1-quest-teleop/xr_teleoperate/teleop/`, which is upstream Unitree
code we carry a fork of -- see `README.md` in this directory. So a deployed copy sits next
to that file and THIS is the canonical one:

  * canonical, tested: `robotics-api/docker/isaac-sim-teleop/client-overrides/` (here)
  * deployed, imported: `<XR_TELEOPERATE_ROOT>/xr_teleoperate/teleop/dimenso_haptics_client.py`

`tests/test_haptics_client.py` runs every logic test against THIS copy -- unconditionally,
no checkout required -- and separately asserts the deployed copy is byte-identical. That
split is deliberate: `tests/test_isaac_teleop_client_patch.py` skips all 15 of its guards
when the checkout is absent, and per GOTCHAS 2026-08-03 a skip and a pass are
indistinguishable in the summary line. The logic must not inherit that.

THE BIG CORRECTION: VUER'S HAPTICS PROPS DO NOT WORK. THE EVENT DOES.
--------------------------------------------------------------------
The spike that preceded this work concluded that haptics reaches the browser as six props
on the `MotionControllers` element -- `pulseLeftStrength`, `pulseLeftDuration`,
`pulseLeftHash`, and the Right three -- edge-triggered by a change in the hash. That is
what Vuer's own docs show and what `vuer/schemas/schema.dial` declares.

**It is wrong for vuer 0.1.6, and following it would have produced a silent no-op.** Read
directly out of the shipped bundle
(`vuer/client_build/assets/chunks/chunk-B_1l0x81.js`), the client's `MotionControllers`
component destructures all six props into locals and then **never reads any of them** --
verified as zero further occurrences of each bound identifier in the function body. They
are not forwarded to the rest-spread either, so they are swallowed, not passed through.
Sending them ships bytes that nothing acts on, with no error anywhere.

The path that IS implemented is a first-class server event:

    from vuer.events import HapticActuatorPulse
    session @ HapticActuatorPulse(left={"strength": 0.8, "duration": 40})

`HapticActuatorPulse.etype == "HAPTIC_ACTUATOR_PULSE"` (`vuer/events.py:276-302`), the
websocket receive path publishes every inbound event onto the downlink bus by etype, and
`MotionControllers` holds the only subscriber:

    downlink.subscribe("HAPTIC_ACTUATOR_PULSE", ({data:{left, right}}) => {
        left  && pulse(leftGamepad,  left.strength,  left.duration);
        right && pulse(rightGamepad, right.strength, right.duration);
    })

where `pulse(gamepad, strength, duration)` is
`gamepad?.hapticActuators?.length && gamepad.hapticActuators[0].pulse(strength, duration)`
-- intensity first, which is what Quest implements. Vuer 0.1.6 ships its own tests for the
event (`vuer/__tests__/test_server_events.py:241-260`); it ships none for the props.

Three consequences that shape everything below:

1. **The payload keys are `strength` and `duration`** -- not `pulseLeftStrength`, and not
   the message's own `duration_ms`. `DEAD_PROP_NAMES` exists so a test can assert we never
   emit the plausible-looking names. Rule 49: pin the ABSENCE of the broken thing, or
   someone "fixes" this from the Vuer docs and silently turns haptics off.
2. **There is no hash, so there is no free edge-trigger.** The event is fire-and-forget:
   every one sent is a pulse played. Onset de-duplication is therefore OURS to own and is
   load-bearing rather than a nicety -- see `PulseGate`. Repeated `pulse()` calls preempt
   rather than queue, so re-sending every tick would make a 40 ms tap into a weaker,
   buzzier 33 ms one. That is the spec's §3.3 "continuous mush" with a mechanism.
3. **The import is the version gate.** `HapticActuatorPulse` does not exist in the
   installed vuer 0.0.60. So `ImportError` means "this vuer is too old", loudly, at
   startup -- strictly better than the props path, whose failure on an old vuer is
   indistinguishable from success. This module stays import-clean of vuer (it is pure
   stdlib and must be testable in the backend's 3.13 venv); the caller does the import and
   owns the log line.

WHY LEFT AND NOT RIGHT
----------------------
Spec §6 P2 says "one zone (`finger`, right hand)". This slice uses the **LEFT** hand
instead, deliberately. A shipped Quest Browser regression routed right-hand pulses to the
left controller and dropped left-hand pulses entirely (Meta investigation 938861405320634,
fixed 2026-03-09). Right-hand-only would therefore be exercising the exact path that
misfires, and "I felt it, but in the wrong hand" would send the next person hunting
through the §3.2 side mapping instead of the browser version. `P2_SIDES` is a tuple and
`select_pulses` takes `sides`, so both hands are reachable for the operator check the
spike calls mandatory -- test each hand independently and record the browser version.

WHAT THIS MODULE CANNOT KNOW
----------------------------
Whether a pulse is ever FELT. Three things gate that and none are checkable from here:
the on-device Quest Browser's controller-haptics permission (added in 42.3, denial is
expected to be silent), the browser version relative to the 2026-03-09 routing fix, and
the fact that haptics does nothing at all over Quest Link -- validation must be in the
on-device browser. See `README.md` §7 in this directory.
"""
from __future__ import annotations

import json

# The DDS topic. Must equal `dimenso_haptics.TOPIC` on the sim side; asserted by
# tests/test_haptics_client.py. Two hardcoded copies of a topic name is how a subscriber
# ends up listening to a route nobody publishes.
TOPIC = "rt/dimenso/haptics"

# §3.1 sides and zones. Mirrors dimenso_haptics.py rather than importing it: that module
# ships to the GPU box on a container PYTHONPATH mount and is not on the laptop.
SIDE_LEFT = "left"
SIDE_RIGHT = "right"
SIDE_BOTH = "both"
ZONE_FINGER = "finger"
ZONE_WRIST = "wrist"
ZONE_BODY = "body"

# The etype Vuer's client actually subscribes to, and the two payload keys it reads.
EVENT_ETYPE = "HAPTIC_ACTUATOR_PULSE"
KEY_STRENGTH = "strength"
KEY_DURATION = "duration"

# Names that look like the right answer and are inert in vuer 0.1.6 -- the six real-but-
# unread props plus the two the Vuer docs misspell. Asserted absent from what we emit.
DEAD_PROP_NAMES = (
    "pulseLeftStrength", "pulseLeftDuration", "pulseLeftHash",
    "pulseRightStrength", "pulseRightDuration", "pulseRightHash",
    # docs.vuer.ai's own example drops the `l`. An unknown prop serialises, ships and is
    # ignored, so the typo is a silent no-op on top of an already-dead path.
    "puseLeftHash", "puseRightHash",
)

# Vuer's declared bounds for the equivalent props (schema.dial), which are also the sane
# bounds for the event: strength 0..1 step .01, duration 0..5000 ms step 10. The 5000 is
# VUER's choice, not a platform cap -- Gamepad Extensions specifies no maximum for
# `pulse()`. Clamping here anyway: `value` out of 0..1 is unspecified behaviour and
# nothing downstream clamps for us.
MIN_STRENGTH = 0.0
MAX_STRENGTH = 1.0
MIN_DURATION_MS = 0
MAX_DURATION_MS = 5000

# P2 slice: one zone, one hand. See "WHY LEFT AND NOT RIGHT" above.
P2_SIDES = (SIDE_LEFT,)
P2_ZONES = (ZONE_FINGER,)


def clamp_strength(value):
    """0..1, and a non-number becomes 0.0 rather than raising.

    0.0 is the safe degenerate value: `pulse(0, ms)` is a no-pulse, so a garbled
    intensity produces silence instead of a full-strength jolt the operator did not ask
    for. Rule 33's "never fake a zero" is about DERIVED FIGURES shown to a human; this is
    an actuator command, where the quiet failure is the safe one.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return MIN_STRENGTH
    if value != value:  # NaN, which every comparison below would pass through
        return MIN_STRENGTH
    return max(MIN_STRENGTH, min(MAX_STRENGTH, value))


def clamp_duration_ms(value):
    """0..5000 ms as an int, and a non-number becomes 0.

    A 0 ms duration is a no-op pulse, i.e. the same fail-quiet choice as
    `clamp_strength`. Note the useful MINIMUM is undocumented on Quest -- the spec's
    10 ms floor is an untested hypothesis, not a measured figure, and belongs on the
    on-device checklist rather than in a clamp here.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return MIN_DURATION_MS
    if value != value:
        return MIN_DURATION_MS
    return int(max(MIN_DURATION_MS, min(MAX_DURATION_MS, value)))


def parse_message(payload):
    """The §3.1 message as a dict, or None if it is not one. NEVER raises.

    Returns None -- not a partially-filled dict -- for anything that is not a JSON object
    with a list `events`. The caller's only correct response to a malformed message is to
    ignore it, so there is one way to say so.

    An EMPTY `events` list is valid and returns a dict with `events: []`. §3.1 is explicit
    that the sim emits a quiet tick rather than omitting the message, so silence on the
    topic means the publisher is dead. Collapsing "no contact" into the same None as
    "garbage" would throw that distinction away.
    """
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = payload.decode("utf-8")
        except Exception:
            return None
    if not isinstance(payload, str):
        return None
    try:
        msg = json.loads(payload)
    except Exception:
        return None
    if not isinstance(msg, dict):
        return None
    events = msg.get("events")
    if not isinstance(events, list):
        return None

    def _int(key):
        try:
            return int(msg.get(key))
        except (TypeError, ValueError):
            return None

    return {
        # None rather than 0 when absent or unparseable: `seq` drives drop detection and
        # `sim_tick` is differenced by consumers, so a fabricated 0 would read as real
        # data and manufacture a gap or a reset (rule 33; GOTCHAS 2026-08-03 on tick
        # arithmetic).
        "seq": _int("seq"),
        "sim_tick": _int("sim_tick"),
        "source": msg.get("source"),
        # Non-dict entries dropped here so every consumer below can assume dicts.
        "events": [e for e in events if isinstance(e, dict)],
    }


def select_pulses(payload, sides=P2_SIDES, zones=P2_ZONES, onset_only=True):
    """`{side: {"strength": float, "duration": int}}` for the pulses this client should
    play. NEVER raises: a malformed message returns `{}`.

    * **One pulse per side per message.** Several fingers hitting one object is ONE tap to
      the operator. Strongest wins, which is also what a preempting `pulse()` would have
      left anyway -- decided here rather than by arrival order.
    * **`onset_only` defaults True** because P2 is onset-only (§6) and every event the sim
      publishes today carries `onset: true`. When P3 starts sending sustained events with
      `onset: false` they are IGNORED rather than buzzing -- the client needs a real
      shaping decision at that point, and doing nothing until it is made beats a 30 Hz
      rattle that the operator will disable and then not have when it matters.
    * **`SIDE_BOTH` is not expanded.** §3.2 gives torso/waist/pelvis `side: "both"`, and
      those are the `body` zone which P2 does not emit. Turning one torso contact into two
      simultaneous pulses is a design decision for P3, not a default that arrives by
      accident the day `body` is switched on. It is filtered out here because "both" is
      not in `sides`.
    """
    msg = parse_message(payload)
    if msg is None:
        return {}
    sides = tuple(sides)
    zones = tuple(zones)
    out = {}
    for event in msg["events"]:
        if event.get("side") not in sides:
            continue
        if event.get("zone") not in zones:
            continue
        if onset_only and not event.get("onset"):
            continue
        strength = clamp_strength(event.get("intensity"))
        duration = clamp_duration_ms(event.get("duration_ms"))
        # A pulse that cannot be felt is not a pulse. Dropping it here keeps the
        # per-side counter from advancing, so the emit side does not spend a controller
        # tick sending a guaranteed no-op.
        if strength <= MIN_STRENGTH or duration <= MIN_DURATION_MS:
            continue
        prev = out.get(event.get("side"))
        if prev is None or strength > prev[KEY_STRENGTH]:
            out[event.get("side")] = {KEY_STRENGTH: strength, KEY_DURATION: duration}
    return out


def pulse_kwargs(pulses):
    """`select_pulses` output -> kwargs for `vuer.events.HapticActuatorPulse`.

    Only sides with a pulse appear. The client checks `left && pulse(...)`, so an absent
    or None side is skipped rather than played at zero -- passing every side explicitly
    would be harmless today and is exactly the kind of thing a future client tightens.
    """
    kwargs = {}
    for side, spec in pulses.items():
        if side not in (SIDE_LEFT, SIDE_RIGHT):
            # `both` and anything unrecognised. The event has no channel for it.
            continue
        kwargs[side] = {
            KEY_STRENGTH: clamp_strength(spec.get(KEY_STRENGTH)),
            KEY_DURATION: clamp_duration_ms(spec.get(KEY_DURATION)),
        }
    return kwargs


class PulseGate:
    """The edge-trigger, on the emit side. Pure: it holds ints, no shared memory.

    Two processes are involved, because `TeleVuer.__init__` forks a child that runs the
    Vuer server (`televuer.py:176-178`). The DDS subscriber runs in the PARENT and
    advances a per-side counter in a `multiprocessing.Value`; the wrapped
    `on_controller_move` runs in the CHILD at ~30 Hz and asks this gate whether the
    counter has moved since it last sent. Nothing but plain ints crosses between them.

    This is what replaces `pulseLeftHash`. With the props dead (see the module docstring)
    the event is fire-and-forget -- every send is a played pulse -- so without a gate the
    child would re-send the same contact 30 times a second, each call preempting the last
    at ~33 ms. `take()` existing is the difference between a tap and a rattle.

    Coalescing is intended: if three onsets land between two controller ticks the counter
    advances three times and ONE pulse plays, carrying the latest values. Three overlapping
    taps 10 ms apart is not three sensations.
    """

    def __init__(self, sides=P2_SIDES):
        self.sides = tuple(sides)
        # 0 is "nothing published yet", matching a fresh Value('l', 0). So the first real
        # message (counter 1) reads as advanced, and a publisher that never fires never
        # pulses.
        self._sent = {side: 0 for side in self.sides}
        self.pulses_sent = 0

    def take(self, counters):
        """`{side: counter}` -> the subset whose counter MOVED since the last `take`.

        Compared with `!=`, not `>`. A monotonic counter in one process cannot go
        backwards, so an inequality that is not an increase means state was reset
        underneath us; one extra tap is a better response to that than going permanently
        silent, which is what `>` would do if a counter ever wrapped or was rezeroed.

        Records the new value even for sides the caller ends up unable to send. A retry
        loop that re-sent the same contact on every subsequent tick would be the rattle
        this class exists to prevent, and the next contact is at most one tick away.
        """
        fresh = {}
        for side in self.sides:
            try:
                current = int(counters.get(side, 0))
            except (TypeError, ValueError, AttributeError):
                continue
            if current != self._sent.get(side, 0):
                self._sent[side] = current
                fresh[side] = current
        self.pulses_sent += len(fresh)
        return fresh
