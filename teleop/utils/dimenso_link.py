"""DIMENSO ADDITION (DIM-752): drive a real G1 over its robot link instead of DDS.

Responsibility: give `teleop_hand_and_arm.py --link` the SAME controller interfaces it uses on the
wired path, so the Quest page, IK, buttons, overlay and recording logic stay byte-identical, and only
the output changes:

  wired (Ethernet, laptop on 192.168.123.x)     --link (Wi-Fi or Ethernet robot link)
  G1_29_ArmController   -> rt/arm_sdk (DDS)     LinkArmController   -> backend /ws/robots/{id}/arm-stream
  Brainco_Controller_ctrl -> rt/brainco/* (DDS)  LinkBraincoHand     -> backend /ws/robots/{id}/hand-stream
  LocoClientWrapper     -> loco service (DDS)    LinkLoco            -> backend /ws/robots/{id}/teleop-stream
  lowstate subscriber   <- rt/lowstate (DDS)     LinkTelemetry       <- backend /ws/robots/{id}/telemetry

The backend relays every target over the robot's own link, and the robot re-checks each one (boot,
seq, ttl <= 250 ms, e-stop, motion grant, initiation, one arm writer; robotics-onboard
motion/{arm,hand,teleop}_stream.py). Nothing here opens a DDS participant.

Safety shape, matching the wired controller:
  * the arm target is rate limited toward the newest command at the wired limit (20 -> 30 rad/s);
  * every frame carries a 200 ms ttl, so a dead laptop, socket or Wi-Fi link stops new motion within
    a quarter second and the robot holds, then releases, on its own clock;
  * no target is ever sent before a FRESH measured arm pose exists (the ramp starts from it).

Collaborators: teleop_hand_and_arm.py (constructs these behind --link), the robotics-api teleop stream
routes (routers/robot_teleop.py) and robot telemetry socket (routers/robot_connect.py).
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np
import logging_mp

logger_mp = logging_mp.getLogger(__name__)

#: G1 29-DoF body indices of the 14 arm joints, left 7 then right 7 (G1_29_JointArmIndex order).
ARM_INDICES = tuple(range(15, 29))
#: Same as the hub and the robot (robotics-api teleop_stream_hub.MAX_TTL_MS = 250).
TTL_MS = 200
#: A measured pose older than this is not a measurement (robotics-api LinkRobotStateSource uses 0.5 s).
MEASURED_STALE_S = 0.5
SEND_HZ = 50.0
HAND_FINGERS = 6


def _ws_url(base: str, path: str, **query: str) -> str:
    q = "&".join(f"{k}={quote(str(v))}" for k, v in query.items() if v)
    return f"{base.rstrip('/')}{path}" + (f"?{q}" if q else "")


class LinkTelemetry:
    """The robot's newest measured joints, from the backend's telemetry socket (read only)."""

    def __init__(self, base_ws: str, robot_id: str) -> None:
        self._url = _ws_url(base_ws, f"/ws/robots/{robot_id}/telemetry")
        self._lock = threading.Lock()
        self._q: Optional[np.ndarray] = None
        self._dq: Optional[np.ndarray] = None
        self._at = 0.0
        self.frames = 0
        threading.Thread(target=self._run, name="dimenso-link-telemetry", daemon=True).start()

    def _run(self) -> None:
        from websockets.sync.client import connect
        while True:
            try:
                with connect(self._url, max_size=None, open_timeout=5) as ws:
                    for raw in ws:
                        self._ingest(raw)
            except Exception as exc:  # reconnect; staleness is what callers see
                logger_mp.warning("[dimenso-link] telemetry socket: %s; reconnecting", exc)
                time.sleep(0.5)

    def _ingest(self, raw) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        joints = msg.get("joints") if isinstance(msg, dict) else None
        if not isinstance(joints, list):
            return
        by = {j.get("index"): j for j in joints if isinstance(j, dict)}
        try:
            q = np.array([float(by[i]["q"]) for i in ARM_INDICES])
            dq = np.array([float(by[i].get("dq", 0.0)) for i in ARM_INDICES])
        except (KeyError, TypeError, ValueError):
            return
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            return
        with self._lock:
            self._q, self._dq, self._at = q, dq, time.monotonic()
            self.frames += 1

    def arm(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """(q14, dq14, age_s), or Nones when nothing fresh has arrived."""
        with self._lock:
            if self._q is None:
                return None, None, None
            age = time.monotonic() - self._at
            if age > MEASURED_STALE_S:
                return None, None, age
            return self._q.copy(), self._dq.copy(), age

    def wait_fresh(self, timeout_s: float) -> np.ndarray:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            q, _, _ = self.arm()
            if q is not None:
                return q
            time.sleep(0.05)
        raise RuntimeError(f"no fresh arm joints from the robot link within {timeout_s:.0f} s ({self._url})")


class LinkStream:
    """One backend latest-value stream (arm, hand or base). Sends are fire-and-forget; acks are counted."""

    def __init__(self, base_ws: str, robot_id: str, session_id: str, path: str, name: str) -> None:
        self.name = name
        self._url = _ws_url(base_ws, f"/ws/robots/{robot_id}/{path}", sessionId=session_id)
        self._lock = threading.Lock()
        self._ws = None
        self._seq = 0
        self.stream_id: Optional[str] = None
        self.refused: Optional[str] = None
        self.admitted = 0
        self.rejected = 0
        self.last_reject: Optional[str] = None

    def _open(self):
        from websockets.sync.client import connect
        ws = connect(self._url, max_size=None, open_timeout=5)
        hello = json.loads(ws.recv(timeout=5))
        if hello.get("type") != "opened":
            ws.close()
            self.refused = f"{hello.get('code')}: {hello.get('reason')}"
            raise RuntimeError(f"{self.name} stream refused -- {self.refused}")
        self.stream_id, self.refused = hello.get("streamId"), None
        threading.Thread(target=self._read_acks, args=(ws,), name=f"dimenso-link-{self.name}-acks", daemon=True).start()
        logger_mp.info("[dimenso-link] %s stream open (%s)", self.name, self.stream_id)
        return ws

    def _read_acks(self, ws) -> None:
        try:
            for raw in ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if msg.get("type") != "ack":
                    continue
                if msg.get("admitted"):
                    self.admitted += 1
                else:
                    self.rejected += 1
                    reason = f"{msg.get('code')}: {msg.get('reason')}"
                    if reason != self.last_reject:
                        logger_mp.warning("[dimenso-link] %s target refused -- %s", self.name, reason)
                    self.last_reject = reason
        except Exception:
            pass
        with self._lock:
            if self._ws is ws:
                self._ws = None

    def send(self, payload: dict) -> bool:
        with self._lock:
            try:
                if self._ws is None:
                    self._ws = self._open()
                self._seq += 1
                self._ws.send(json.dumps({"type": "target", "seq": self._seq, "ttlMs": TTL_MS, **payload}))
                return True
            except Exception as exc:
                if self._ws is not None:
                    try:
                        self._ws.close()
                    except Exception:
                        pass
                self._ws = None
                if str(exc) != self.last_reject:
                    logger_mp.warning("[dimenso-link] %s send failed: %s", self.name, exc)
                self.last_reject = str(exc)
                return False

    def close(self) -> None:
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    pass
            self._ws = None


class LinkArmController:
    """Duck-types G1_29_ArmController for the loop: q_target/tauff_target/ctrl_lock, ctrl_dual_arm,
    get_current_dual_arm_q/_dq, get_current_motor_q, speed_gradual_max, ctrl_dual_arm_go_home.

    On construction it ramps both arms from the measured pose to home (zeros, the wired go-home target) at
    `home_rad_s`, then HOLDS home until the loop commands otherwise. Torques are not sent: the robot's arm
    stream takes joint targets only and its sidecar applies its own gains."""

    def __init__(self, telemetry: LinkTelemetry, stream: LinkStream, *, home_rad_s: float = 0.6) -> None:
        self._tm, self._stream = telemetry, stream
        q0 = telemetry.wait_fresh(10.0)
        logger_mp.info("[dimenso-link] measured arms %s -> home over the link", np.round(q0, 3).tolist())
        self.ctrl_lock = threading.Lock()
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)
        self._sent = q0.copy()
        self.arm_velocity_limit = home_rad_s
        self._gradual_start: Optional[float] = None
        self.all_motor_q = None
        self.sent_frames = 0
        threading.Thread(target=self._pump, name="dimenso-link-arm", daemon=True).start()

    def _pump(self) -> None:
        dt = 1.0 / SEND_HZ
        while True:
            t0 = time.monotonic()
            with self.ctrl_lock:
                target = np.asarray(self.q_target, dtype=float).copy()
            if self._gradual_start is not None:   # wired: 20 -> 30 rad/s over 5 s after START
                self.arm_velocity_limit = 20.0 + 10.0 * min(1.0, (t0 - self._gradual_start) / 5.0)
            if np.all(np.isfinite(target)) and target.shape == (14,):
                step = self.arm_velocity_limit * dt
                self._sent = self._sent + np.clip(target - self._sent, -step, step)
                if self._stream.send({"q14": [float(v) for v in self._sent]}):
                    self.sent_frames += 1
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    # ---- the G1_29_ArmController surface the loop and ArmHold use ----
    def ctrl_dual_arm(self, q_target, tauff_target) -> None:
        with self.ctrl_lock:
            self.q_target = np.asarray(q_target, dtype=float)
            self.tauff_target = np.asarray(tauff_target, dtype=float)

    def _measured(self) -> tuple[np.ndarray, np.ndarray]:
        q, dq, _ = self._tm.arm()
        if q is None:   # stale: the IK seeds from the last sent target, never from zeros
            return self._sent.copy(), np.zeros(14)
        return q, dq

    def get_current_dual_arm_q(self) -> np.ndarray:
        return self._measured()[0]

    def get_current_dual_arm_dq(self) -> np.ndarray:
        return self._measured()[1]

    def get_current_motor_q(self) -> np.ndarray:
        raise RuntimeError("--link records on the backend from the robot link; the client never records")

    def speed_gradual_max(self, t: float = 5.0) -> None:
        self._gradual_start = time.monotonic()

    def speed_instant_max(self) -> None:
        self._gradual_start, self.arm_velocity_limit = None, 30.0

    def ctrl_dual_arm_go_home(self) -> None:
        """Exit path: ramp home at 0.6 rad/s and wait (<= 10 s); the robot holds, then releases, after we stop."""
        self._gradual_start, self.arm_velocity_limit = None, 0.6
        self.ctrl_dual_arm(np.zeros(14), np.zeros(14))
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and np.max(np.abs(self._sent)) > 1e-6:
            time.sleep(0.05)
        q, _, _ = self._tm.arm()
        logger_mp.info("[dimenso-link] go-home done; measured %s", None if q is None else np.round(q, 3).tolist())


def brainco_controller_targets(trigger10: float, squeeze: float) -> np.ndarray:
    """The wired controller mapping (robot_hand_brainco.py, DIMENSO ADDITION DIM-657), one hand:
    trigger in xr_teleoperate's [10 open .. 0 pressed], squeeze [0..1]; out thumb-aux, thumb, index..pinky
    in [0 open .. 1 closed]."""
    t = (10.0 - float(trigger10)) / 10.0
    s = float(squeeze)
    return np.array([
        np.clip((s - 0.5) / 0.5, 0.0, 0.98),   # thumb-aux <- GRIP
        np.clip(s / 0.5, 0.0, 0.7),            # thumb     <- GRIP
        np.clip(t, 0.0, 0.98),                 # index     <- TRIGGER
        np.clip(t, 0.0, 0.98),                 # middle    <- TRIGGER
        np.clip(t, 0.0, 0.98),                 # ring      <- TRIGGER
        np.clip(t, 0.0, 0.98),                 # pinky     <- TRIGGER
    ])


class LinkBraincoHand:
    """Duck-types Brainco_Controller_ctrl: reads the same four shared Values and sends 6+6 finger targets."""

    def __init__(self, left_trigger, left_squeeze, right_trigger, right_squeeze, dual_hand_data_lock=None,
                 dual_hand_state_array=None, dual_hand_action_array=None, *, stream: LinkStream,
                 xr_motion_data_ready_in=None, fps: float = 30.0) -> None:
        self._v = (left_trigger, left_squeeze, right_trigger, right_squeeze)
        self._lock, self._action = dual_hand_data_lock, dual_hand_action_array
        self._ready, self._stream, self._fps = xr_motion_data_ready_in, stream, fps
        self.sent_frames = 0
        threading.Thread(target=self._pump, name="dimenso-link-hands", daemon=True).start()

    def _pump(self) -> None:
        left = right = np.zeros(HAND_FINGERS)
        while True:
            t0 = time.monotonic()
            ready = True
            if self._ready is not None:
                with self._ready.get_lock():
                    ready = bool(self._ready.value)
            if ready:
                vals = []
                for v in self._v:
                    with v.get_lock():
                        vals.append(float(v.value))
                left = brainco_controller_targets(vals[0], vals[1])
                right = brainco_controller_targets(vals[2], vals[3])
            if self._lock is not None and self._action is not None:
                with self._lock:
                    self._action[:] = np.concatenate((left, right))
            if self._stream.send({"left": [float(x) for x in left], "right": [float(x) for x in right]}):
                self.sent_frames += 1
            time.sleep(max(0.0, 1.0 / self._fps - (time.monotonic() - t0)))


class LinkLoco:
    """Duck-types LocoClientWrapper.Move/StopMove with the robot link's base-velocity stream (capped 0.3)."""

    def __init__(self, stream: LinkStream) -> None:
        self._stream = stream

    def Move(self, vx: float, vy: float, vyaw: float) -> None:
        cap = lambda v: max(-0.3, min(0.3, float(v)))   # the hub's VELOCITY_CAPS; it refuses over-cap targets
        self._stream.send({"velocity": {"vx": cap(vx), "vy": cap(vy), "vyaw": cap(vyaw)}})

    def StopMove(self) -> None:
        self._stream.send({"velocity": {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}})


class LinkSession:
    """Everything --link needs, built from the backend's env (DIMENSO_LINK_WS, _ROBOT, _SESSION)."""

    def __init__(self, base_ws: str, robot_id: str, session_id: str) -> None:
        if not (base_ws and robot_id and session_id):
            raise RuntimeError("--link needs DIMENSO_LINK_WS, DIMENSO_LINK_ROBOT and DIMENSO_LINK_SESSION from the backend")
        self.telemetry = LinkTelemetry(base_ws, robot_id)
        self.arm_stream = LinkStream(base_ws, robot_id, session_id, "arm-stream", "arm")
        self.hand_stream = LinkStream(base_ws, robot_id, session_id, "hand-stream", "hand")
        self.base_stream = LinkStream(base_ws, robot_id, session_id, "teleop-stream", "base")

    def diagnostics(self) -> dict:
        _, _, age = self.telemetry.arm()
        return {s.name: {"admitted": s.admitted, "rejected": s.rejected, "lastReject": s.last_reject,
                         "refused": s.refused}
                for s in (self.arm_stream, self.hand_stream, self.base_stream)} | {
            "telemetryFrames": self.telemetry.frames, "measuredAgeS": None if age is None else round(age, 3)}

    def close(self) -> None:
        for s in (self.arm_stream, self.hand_stream, self.base_stream):
            s.close()
