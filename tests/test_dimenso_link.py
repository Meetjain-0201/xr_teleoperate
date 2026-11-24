"""--link sends the wired loop's commands over the backend's robot-link streams (DIM-752).

A fake backend (websockets server on loopback) stands in for robotics-api; nothing reaches a robot.
"""

import json
import threading
import time
from multiprocessing import Value

import numpy as np
import pytest
from websockets.sync.server import serve

from teleop.utils import dimenso_link as dl


class FakeBackend:
    """/ws/robots/r/{telemetry,arm-stream,hand-stream,teleop-stream}, recording every target."""

    def __init__(self, arm_q):
        self.arm_q = list(arm_q)
        self.targets = {"arm-stream": [], "hand-stream": [], "teleop-stream": []}
        self.refuse = None
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.port = self.server.socket.getsockname()[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"ws://127.0.0.1:{self.port}"

    def _handle(self, ws):
        path = ws.request.path.split("?")[0].rsplit("/", 1)[-1]
        if path == "telemetry":
            while True:
                joints = [{"index": i, "q": 0.0, "dq": 0.0} for i in range(15)]
                joints += [{"index": 15 + i, "q": q, "dq": 0.0} for i, q in enumerate(self.arm_q)]
                ws.send(json.dumps({"type": "telemetry", "joints": joints}))
                time.sleep(0.02)
        if self.refuse:
            ws.send(json.dumps({"type": "refused", "code": self.refuse, "reason": "test"}))
            return
        assert "sessionId=ctl-test" in ws.request.path
        ws.send(json.dumps({"type": "opened", "streamId": f"s-{path}"}))
        for raw in ws:
            msg = json.loads(raw)
            self.targets[path].append(msg)
            ws.send(json.dumps({"type": "ack", "seq": msg["seq"], "admitted": True}))

    def close(self):
        self.server.shutdown()


@pytest.fixture
def backend():
    b = FakeBackend([0.3] * 7 + [-0.3] * 7)
    yield b
    b.close()


def wait_for(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_brainco_mapping_matches_the_wired_controller():
    # trigger fully pressed (0 in xr_teleoperate's 10..0), grip released: four fingers close, thumb open
    np.testing.assert_allclose(dl.brainco_controller_targets(0.0, 0.0), [0, 0, 0.98, 0.98, 0.98, 0.98])
    # grip half: thumb flexes to its 0.7 cap, thumb-aux not yet rotated; trigger released keeps fingers open
    np.testing.assert_allclose(dl.brainco_controller_targets(10.0, 0.5), [0, 0.7, 0, 0, 0, 0])
    np.testing.assert_allclose(dl.brainco_controller_targets(10.0, 1.0), [0.98, 0.7, 0, 0, 0, 0])


def test_arm_ramps_home_from_the_measured_pose_never_jumps(backend):
    s = dl.LinkSession(backend.base, "r", "ctl-test")
    t_start = time.time()
    arm = dl.LinkArmController(s.telemetry, s.arm_stream, home_rad_s=0.6)
    assert wait_for(lambda: len(backend.targets["arm-stream"]) >= 5)
    first = np.array(backend.targets["arm-stream"][0]["q14"])
    # the first target is one rate-limited step from the MEASURED pose, toward zero (timer slack allowed)
    assert np.max(np.abs(first - np.array(backend.arm_q))) <= 0.6 * 1.5 / dl.SEND_HZ
    # and the whole ramp never beats 0.6 rad/s of real time
    time.sleep(0.3)
    last = np.array(backend.targets["arm-stream"][-1]["q14"])
    assert np.max(np.abs(last - np.array(backend.arm_q))) <= 0.6 * (time.time() - t_start) + 1e-6
    assert all(t["ttlMs"] == dl.TTL_MS for t in backend.targets["arm-stream"])
    seqs = [t["seq"] for t in backend.targets["arm-stream"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    np.testing.assert_allclose(arm.get_current_dual_arm_q(), backend.arm_q)


def test_engage_follows_the_command_at_the_wired_speed(backend):
    s = dl.LinkSession(backend.base, "r", "ctl-test")
    arm = dl.LinkArmController(s.telemetry, s.arm_stream)
    arm.speed_gradual_max()
    arm.ctrl_dual_arm(np.full(14, 0.2), np.zeros(14))
    assert wait_for(lambda: bool(backend.targets["arm-stream"]) and np.allclose(backend.targets["arm-stream"][-1]["q14"], 0.2))


def test_no_target_without_a_fresh_measured_pose():
    b = FakeBackend([0.0] * 14)
    b.arm_q = []          # telemetry frames carry no arm joints
    try:
        s = dl.LinkSession(b.base, "r", "ctl-test")
        with pytest.raises(RuntimeError, match="no fresh arm joints"):
            s.telemetry.wait_fresh(0.5)
        assert b.targets["arm-stream"] == []
    finally:
        b.close()


def test_hands_and_walking_use_their_own_streams(backend):
    s = dl.LinkSession(backend.base, "r", "ctl-test")
    vals = [Value("d", 0.0), Value("d", 0.0), Value("d", 10.0), Value("d", 1.0)]
    dl.LinkBraincoHand(*vals, stream=s.hand_stream)
    assert wait_for(lambda: backend.targets["hand-stream"])
    h = backend.targets["hand-stream"][-1]
    np.testing.assert_allclose(h["left"], [0, 0, 0.98, 0.98, 0.98, 0.98])
    np.testing.assert_allclose(h["right"], [0.98, 0.7, 0, 0, 0, 0])
    loco = dl.LinkLoco(s.base_stream)
    loco.Move(0.5, -0.1, 0.0)
    loco.StopMove()
    assert wait_for(lambda: len(backend.targets["teleop-stream"]) == 2)
    assert backend.targets["teleop-stream"][0]["velocity"] == {"vx": 0.3, "vy": -0.1, "vyaw": 0.0}
    assert backend.targets["teleop-stream"][1]["velocity"] == {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}


def test_a_refused_stream_is_named_and_sends_nothing():
    b = FakeBackend([0.0] * 14)
    b.refuse = "arm_writer_busy"
    try:
        s = dl.LinkSession(b.base, "r", "ctl-test")
        assert s.arm_stream.send({"q14": [0.0] * 14}) is False
        assert "arm_writer_busy" in s.arm_stream.refused
        assert s.diagnostics()["arm"]["refused"].startswith("arm_writer_busy")
    finally:
        b.close()


def test_solved_waist_yaw_rides_on_q14_with_upright_roll_and_pitch(backend):
    s = dl.LinkSession(backend.base, "r", "ctl-test")
    arm = dl.LinkArmController(s.telemetry, s.arm_stream)
    assert wait_for(lambda: len(backend.targets["arm-stream"]) >= 3)
    assert all("waist3" not in t for t in backend.targets["arm-stream"])    # nothing until the IK solves one
    arm.set_waist_yaw(0.1)
    assert wait_for(lambda: any("waist3" in t for t in backend.targets["arm-stream"]))
    sent = [t["waist3"] for t in backend.targets["arm-stream"] if "waist3" in t]
    assert sent[0] == [0.1, 0.0, 0.0] and len(sent) == 1     # sent once, not every frame
