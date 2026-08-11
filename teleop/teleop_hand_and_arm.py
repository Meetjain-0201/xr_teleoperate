import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize # dds 
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController, H2_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK, H2_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

# DIMENSO ADDITION: shared between the rt/reset_pose/cmd callback thread and the
# main loop. Module level because the loop is module-level code, so a closure
# cell would not be writable from the callback.
_DIMENSO_RESET_FLAG = 0.0

# ---------------- DIMENSO ADDITION: collision haptics (DIM-555) ----------------
# Closes the chain the sim half opened: the robot touches something, PhysX reports
# it, the sim publishes rt/dimenso/haptics, and the operator's Quest controller taps.
# Contract is robotics-api docs/superpowers/specs/2026-08-06-collision-haptics-design.md
# (§3.1 message schema). Full rationale, including the two-copy arrangement for
# dimenso_haptics_client.py, is in robotics-api
# docker/isaac-sim-teleop/client-overrides/README.md §6.
#
# OFF BY DEFAULT. `DIMENSO_HAPTICS=1` -- the SAME env var that arms the sim-side
# publisher. One flag for one chain, deliberately: a half-armed chain (client on,
# sim off) looks exactly like a broken client and is not a state worth being able
# to reach.
#
# THREE THINGS THIS BLOCK HAS TO GET RIGHT, all of them ordering or fail-open:
#
# 1. IT IS PRE-FORK, AND THAT IS THE WHOLE CORRECTNESS ARGUMENT.
#    TeleVuer.__init__ binds the CONTROLLER_MOVE handler (televuer.py:96) and then
#    forks a daemon child that runs the Vuer server (televuer.py:176-178). So the
#    monkey-patch and these multiprocessing Values must both exist BEFORE
#    TeleVuerWrapper(...) is constructed below. Patch it afterwards and only the
#    parent's copy changes -- the child keeps running the unpatched method and
#    haptics silently never fires. Same class as "a running backend is not the code
#    you just committed" (robotics-api GOTCHAS 2026-07-28). Guarded by
#    tests/test_haptics_client.py::
#      test_the_client_patches_on_controller_move_BEFORE_constructing_the_wrapper
#
# 2. THE PULSE ARRIVES AS AN EVENT, NOT AS PROPS ON MotionControllers.
#    vuer 0.1.6's browser client destructures pulseLeftStrength / pulseLeftDuration /
#    pulseLeftHash and the Right three, and then NEVER READS THEM -- they are dead
#    code, and sending them is a silent no-op. The path that works is
#    `session @ HapticActuatorPulse(left={"strength":.., "duration":..})`, whose
#    etype HAPTIC_ACTUATOR_PULSE is the one MotionControllers actually subscribes
#    to. Because that event is fire-and-forget, onset de-duplication is OURS
#    (PulseGate) rather than a free edge-trigger off the hash.
#    The import below is therefore also the VERSION GATE: HapticActuatorPulse does
#    not exist in vuer 0.0.60, so an old vuer fails loudly here instead of
#    pretending to work.
#
# 3. FAIL-OPEN, NON-NEGOTIABLE. This runs on the same handler that delivers wrist
#    pose to the arms. Upstream's method is awaited FIRST and outside our try, and
#    every fault is caught and logged ONCE rather than per-tick at 30 Hz. An
#    observability hook has already hung a sim control loop (GOTCHAS 2026-08-06).
_DIMENSO_HAPTICS_ENABLED = os.environ.get("DIMENSO_HAPTICS", "") == "1"
# Which hand(s). Default LEFT, not right, and not for symmetry with the spec: a
# shipped Quest Browser regression routed right-hand pulses to the LEFT controller
# and dropped left entirely (Meta investigation 938861405320634, fixed 2026-03-09).
# Right-hand-only would be exercising exactly the path that misfires. Set
# DIMENSO_HAPTICS_SIDES=left,right for the both-hands check, which is mandatory
# before believing any "wrong hand" report is ours rather than the browser's.
_DIMENSO_HAPTICS_SIDES = tuple(
    # P3: BOTH hands. "left" was the P2 vertical slice, and reverting an unrelated change
    # silently took the operator back to it -- one hand, whatever he touched.
    s.strip() for s in os.environ.get("DIMENSO_HAPTICS_SIDES", "left,right").split(",") if s.strip()
)
_dimenso_haptics = None            # the pure module, or None when off/unavailable
_DIMENSO_HAPTIC_STATE = None       # {side: {"seq","strength","ms"}} of shared Values
_dimenso_haptics_gate = None       # PulseGate; the child gets its own copy at fork
_DimensoHapticPulse = None         # vuer.events.HapticActuatorPulse
_dimenso_haptics_faults = 0        # so the log line happens once, not 30x a second
_dimenso_haptics_rx_faults = 0
_dimenso_haptics_sent = 0

if _DIMENSO_HAPTICS_ENABLED:
    try:
        # Sibling of this file, copied from robotics-api client-overrides/. Byte
        # identity is asserted by tests/test_haptics_client.py so the two cannot
        # drift the way a regenerated fork silently did in GOTCHAS 2026-08-03.
        import dimenso_haptics_client as _dimenso_haptics
        from vuer.events import HapticActuatorPulse as _DimensoHapticPulse

        _DIMENSO_HAPTIC_STATE = {
            side: {
                # Monotonic per-side pulse counter -- the edge trigger. Advanced by
                # the DDS callback in THIS process, read by the Vuer child.
                "seq": Value('l', 0, lock=True),
                "strength": Value('d', 0.0, lock=True),
                "ms": Value('i', 0, lock=True),
            }
            for side in _DIMENSO_HAPTICS_SIDES
        }
        _dimenso_haptics_gate = _dimenso_haptics.PulseGate(sides=_DIMENSO_HAPTICS_SIDES)
        logger_mp.info(
            f"[dimenso] collision haptics ARMED for {list(_DIMENSO_HAPTICS_SIDES)} "
            f"on {_dimenso_haptics.TOPIC}"
        )
    except Exception as _e:
        # Explicitly NOT fatal. DIMENSO_HAPTICS=1 against a checkout without the
        # module, or against vuer 0.0.60, must degrade to a teleop session with no
        # haptics -- never to no teleop session. But it is an ERROR, not a debug
        # line: the operator asked for haptics and is not getting any.
        logger_mp.error(
            f"[dimenso] DIMENSO_HAPTICS=1 but haptics could not initialise ({_e!r}). "
            "Continuing WITHOUT haptics. If this is an ImportError on "
            "vuer.events.HapticActuatorPulse, the installed vuer is too old -- pin "
            "vuer[all]==0.1.6 (0.0.60 has no such event; 0.0.69-0.0.72 vendor a "
            "pre-haptics browser client and fail silently)."
        )
        _dimenso_haptics = None
        _DIMENSO_HAPTIC_STATE = None
        _dimenso_haptics_gate = None
        _DimensoHapticPulse = None


def _dimenso_haptics_ready():
    return (_dimenso_haptics is not None and _DIMENSO_HAPTIC_STATE is not None
            and _dimenso_haptics_gate is not None and _DimensoHapticPulse is not None)


def _dimenso_haptics_ingest(payload):
    """Latest-wins. Turn one rt/dimenso/haptics message into shared-Value writes.

    Runs on the DDS reader thread, which is the only writer. Pure decisions live in
    dimenso_haptics_client.select_pulses; this function only moves numbers, so that
    everything decidable is unit-tested off-headset.

    Returns the sides it advanced, for the tests and for the log line.
    """
    global _dimenso_haptics_rx_faults
    if not _dimenso_haptics_ready():
        return ()
    try:
        # onset_only=False: P3 sends sustain and release events too, so held contact
        # keeps a low buzz instead of going silent after the first tap.
        pulses = _dimenso_haptics.select_pulses(
            payload, sides=_DIMENSO_HAPTICS_SIDES, onset_only=False)
        advanced = []
        for side, spec in pulses.items():
            slot = _DIMENSO_HAPTIC_STATE.get(side)
            if slot is None:
                continue
            # Values BEFORE the counter. The reader gates on the counter, so bumping
            # it first would let one controller tick read a fresh seq against a stale
            # strength -- a pulse at the previous contact's amplitude.
            slot["strength"].value = float(spec[_dimenso_haptics.KEY_STRENGTH])
            slot["ms"].value = int(spec[_dimenso_haptics.KEY_DURATION])
            with slot["seq"].get_lock():
                slot["seq"].value += 1
            advanced.append(side)
        return tuple(advanced)
    except Exception as _e:
        _dimenso_haptics_rx_faults += 1
        if _dimenso_haptics_rx_faults == 1:
            logger_mp.error(f"[dimenso] haptics ingest fault (logged once): {_e!r}")
        return ()


def _dimenso_haptics_emit(session):
    """Send at most one HapticActuatorPulse per side, only on a fresh onset.

    Runs in the VUER CHILD process, ~30 Hz, on the handler that also carries wrist
    pose to the arms. Never raises.

    The gate is what keeps this a tap. Repeated pulse() calls PREEMPT rather than
    queue, so re-sending an unchanged contact every tick would truncate a 40 ms tap
    to ~33 ms, every tick -- quieter and buzzier than sending it once.
    """
    global _dimenso_haptics_faults, _dimenso_haptics_sent
    if not _dimenso_haptics_ready():
        return
    try:
        counters = {}
        for side in _DIMENSO_HAPTICS_SIDES:
            slot = _DIMENSO_HAPTIC_STATE.get(side)
            if slot is not None:
                counters[side] = slot["seq"].value
        fresh = _dimenso_haptics_gate.take(counters)
        if not fresh:
            return
        pulses = {}
        for side in fresh:
            slot = _DIMENSO_HAPTIC_STATE[side]
            pulses[side] = {
                _dimenso_haptics.KEY_STRENGTH: slot["strength"].value,
                _dimenso_haptics.KEY_DURATION: slot["ms"].value,
            }
        kwargs = _dimenso_haptics.pulse_kwargs(pulses)
        if not kwargs:
            return
        session @ _DimensoHapticPulse(**kwargs)
        _dimenso_haptics_sent += 1
        # The FIRST send is the single most valuable line in this log. It splits
        # "we never sent anything" from "we sent and nothing was felt" -- and the
        # latter is a browser question (the 42.3+ controller-haptics permission,
        # the 2026-03-09 per-hand routing fix, or a Link session, all of which fail
        # silently), not a question about this code.
        if _dimenso_haptics_sent == 1:
            logger_mp.info(f"[dimenso] FIRST haptic pulse sent to the headset: {kwargs}")
        elif _dimenso_haptics_sent % 100 == 0:
            logger_mp.info(f"[dimenso] haptic pulses sent: {_dimenso_haptics_sent}")
    except Exception as _e:
        _dimenso_haptics_faults += 1
        if _dimenso_haptics_faults == 1:
            logger_mp.error(f"[dimenso] haptics emit fault (logged once): {_e!r}")


def _dimenso_patch_televuer_for_haptics():
    """Wrap TeleVuer.on_controller_move. MUST be called before TeleVuerWrapper(...).

    A wrap rather than a second add_handler("CONTROLLER_MOVE") registration: Vuer
    stores handlers as a dict of dicts so both work, but wrapping guarantees our
    send happens AFTER the pose has been written to shared memory on the very same
    tick, with no second dispatch to order against.
    """
    if not _dimenso_haptics_ready():
        return False
    from televuer import TeleVuer as _DimensoTeleVuer
    _orig = _DimensoTeleVuer.on_controller_move

    async def _dimenso_on_controller_move(self, event, session, fps=60):
        # Upstream FIRST, and outside the try: this is the arm/pose path, and its
        # own behaviour (including how it handles a bad event) must be untouched.
        await _orig(self, event, session, fps=fps)
        _dimenso_haptics_emit(session)

    _DimensoTeleVuer.on_controller_move = _dimenso_on_controller_move
    return True
# ------------------------- END DIMENSO ADDITION -----------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1', 'H2'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    # network parameters
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.debug(f"args: {args}")

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # ---- DIMENSO ADDITION (DIM-547): wrist views on their OWN thread -------------
        # NEVER in the control loop -- an inline fetch stalled teleop on a live session.
        def _dimenso_wrist_pump(client, wrapper, cfg):
            import threading, time as _t
            enabled = {"left":  bool(cfg.get("left_wrist_camera", {}).get("enable_zmq")),
                       "right": bool(cfg.get("right_wrist_camera", {}).get("enable_zmq"))}
            if not any(enabled.values()):
                # 2026-08-11: SAY SO. This used to be a bare `return`.
                #
                # Meet, in the headset: "why are wrist cameras not appearing in teleop, check
                # and this error should not repeat". The cause was not placement and not this
                # client -- it was `ISAAC_WRIST_CAMERAS` unset on the backend, which leaves the
                # sim's cam_config_server.yaml at `enable_zmq: false` for both wrists and never
                # spawns the CameraCfgs. `cfg` here comes from the IMAGE SERVER over ZMQ
                # (img_client.get_cam_config()), so this dict is the box's own answer.
                #
                # The defect worth fixing permanently is that the operator saw one frame and
                # NOTHING anywhere said why -- not this log, not the sim's, not the backend's.
                # A silent `return` on a feature the operator is actively looking for is the
                # same class as every other "correct code, absent call site" trap here.
                logger_mp.warning(
                    "[dimenso] WRIST PANELS OFF: the image server reports enable_zmq=false "
                    "for BOTH wrist cameras, so no wrist frames exist to draw and only the "
                    "head view will appear. This is a LAUNCH-TIME setting and cannot be "
                    "changed on a running session: set ISAAC_WRIST_CAMERAS=1 in "
                    "robotics-api/.env and relaunch the Isaac session (one flag drives both "
                    "the CameraCfg spawn and this ZMQ publish). Teleop is otherwise "
                    "unaffected."
                )
                return
            if not all(enabled.values()):
                logger_mp.warning(
                    "[dimenso] only the %s wrist stream is published; the other panel will "
                    "stay absent. Both come from one flag, so this means the box's "
                    "cam_config_server.yaml was edited by hand rather than rendered.",
                    ", ".join(k for k, v in enabled.items() if v),
                )
            getters = {"left": client.get_left_wrist_frame, "right": client.get_right_wrist_frame}
            faults = {"left": 0, "right": 0}
            MAX_FAULTS = 30

            # 2026-08-11. A frame that never ARRIVES raises nothing, so the fault counter above
            # cannot see it: `getters[side]()` simply returns None forever and the panel stays
            # blank. That is the exact half-configured state isaac_cam_config's docstring warns
            # about -- "the image server advertises a stream that never produces a frame, and
            # the client waits on it" -- i.e. ZMQ published but the CameraCfg not spawned. One
            # line, once per side, so a blank panel is never mistaken for a placement bug again.
            seen = {"left": False, "right": False}
            starved_reported = {"left": False, "right": False}
            STARVED_AFTER = 100          # x 0.1s sleep = ~10s, well past Isaac's first frames

            def run():
                logger_mp.info("[dimenso] wrist panel pump up (%s)",
                               ", ".join(k for k, v in enabled.items() if v))
                loops = 0
                while True:
                    loops += 1
                    for side, on in enabled.items():
                        if not on or faults[side] >= MAX_FAULTS:
                            continue
                        if (loops >= STARVED_AFTER and not seen[side]
                                and not starved_reported[side]):
                            starved_reported[side] = True
                            logger_mp.warning(
                                "[dimenso] %s wrist stream is PUBLISHED but has produced no "
                                "frame in ~%.0fs -- the panel will be blank. This is the "
                                "half-configured state: enable_zmq is true on the box while "
                                "the wrist CameraCfg was not spawned. Both halves come from "
                                "ISAAC_WRIST_CAMERAS; a mismatch means the box's config was "
                                "hand-edited instead of rendered.", side, STARVED_AFTER * 0.1)
                        try:
                            f = getters[side]()
                            if f is not None and f.bgr is not None:
                                if not seen[side]:
                                    seen[side] = True
                                    logger_mp.info(
                                        "[dimenso] %s wrist panel: first frame received", side)
                                wrapper.render_wrist_to_xr(side, f.bgr)
                                faults[side] = 0
                        except Exception:
                            faults[side] += 1
                            if faults[side] == MAX_FAULTS:
                                logger_mp.warning("[dimenso] %s wrist panel disabled after %d "
                                                  "errors; teleop unaffected", side, MAX_FAULTS)
                    _t.sleep(0.1)
            threading.Thread(target=run, daemon=True, name="dimenso-wrist-pump").start()
        # ---------------------- END DIMENSO ADDITION --------------------------------

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # ---- DIMENSO ADDITION: collision haptics -- patch BEFORE the fork ----
        # TeleVuer.__init__ binds the CONTROLLER_MOVE handler and then starts a daemon
        # child process (televuer.py:96, :176-178). Applied one statement earlier than
        # the construction below and NOT a line later: after the fork this would only
        # rebind the parent's copy of the method, the Vuer child would keep running
        # upstream's, and haptics would silently never fire with nothing in any log to
        # say so. See the module-level block for the full rationale.
        #
        # Controller input only -- on_controller_move is not even registered in
        # hand-tracking mode (televuer.py:93-96), and bare hands have no haptic
        # actuator to pulse (inputSource.gamepad is null for optical hand tracking).
        if _DIMENSO_HAPTICS_ENABLED:
            if args.input_mode != "controller":
                logger_mp.warning(
                    "[dimenso] DIMENSO_HAPTICS=1 ignored: --input-mode is "
                    f"'{args.input_mode}', and hand tracking exposes no haptic actuator. "
                    "Run with --input-mode controller."
                )
            elif _dimenso_patch_televuer_for_haptics():
                logger_mp.info("[dimenso] TeleVuer.on_controller_move wrapped for haptics")
        # ---------------------- END DIMENSO ADDITION ----------------------

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     arm_reference_mode="head_yaw"
                                     )

        # DIM-547: pump starts AFTER the wrapper, never inside the loop.
        if os.environ.get("DIMENSO_WRIST_PANELS", "1").strip().lower() not in ("0","false","no"):
            _dimenso_wrist_pump(img_client, tv_wrapper, camera_config)

        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)
        elif args.arm == "H2":
            arm_ik = H2_ArmIK()
            arm_ctrl = H2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

        # end-effector
        xr_motion_data_ready = Value('b', False, lock=True)        # [input] whether XR hand/controller motion data has arrived
        if args.ee in ("dex3", "inspire_ftp") and args.input_mode == "controller":
            raise ValueError(f"{args.ee} does not support controller input mode.")
        elif args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_dfx" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX_ctrl(left_gripper_trigger_in, right_gripper_trigger_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "hand":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_hand
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_hand(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                                dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        elif args.ee == "brainco" and args.input_mode == "controller":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller_ctrl
            left_gripper_trigger_in = Value('d', 10.0, lock=True)  # [input]
            left_gripper_squeeze_in = Value('d', 0.0, lock=True)   # [input]
            right_gripper_trigger_in = Value('d', 10.0, lock=True) # [input]
            right_gripper_squeeze_in = Value('d', 0.0, lock=True)  # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller_ctrl(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                                                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim, xr_motion_data_ready_in=xr_motion_data_ready)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        # ---------------- DIMENSO ADDITION: sim locomotion publisher ----------------
        # rt/run_command/cmd is what unitree_sim_isaaclab's wholebody action provider
        # reads as [vx, vy, yaw, height] (action_provider_wh_dds.py:313-326, default
        # [0,0,0,0.8]). Nothing in this client published it, so the thumbsticks did
        # nothing in sim. Published from HERE, not the backend, because this process
        # already owns a CycloneDDS 0.10.2 participant on domain 1 -- an 11.x
        # participant on the same domain segfaults this client during XTypes
        # discovery. See robotics-api docker/isaac-sim-teleop/client-overrides/.
        _dimenso_body_pub = None
        if args.sim:
            try:
                from unitree_sdk2py.core.channel import ChannelPublisher as _DimensoPub
                from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_ as _DimensoStr
                _dimenso_body_pub = _DimensoPub("rt/run_command/cmd", _DimensoStr)
                _dimenso_body_pub.Init()
                logger_mp.info("[dimenso] locomotion publisher up on rt/run_command/cmd")
            except Exception as _e:
                logger_mp.error(f"[dimenso] could not create locomotion publisher: {_e}")

        _DIMENSO_STAND_HEIGHT = 0.8
        # Full-stick command, scaled to what THIS policy was trained to track.
        #
        # Upstream's sign convention is kept verbatim, but not its 0.3 cap. That cap comes
        # from xr_teleoperate issue #135 and applies to the REAL robot driven by Unitree's
        # own locomotion controller (`loco_wrapper.Move`, further down this file). The sim
        # path publishes to AGILE's Velocity-G1-History-v0, whose CommandsCfg trained on:
        #
        #     lin_vel_x  (-0.5, 0.5)    lin_vel_y  (-0.5, 0.5)    ang_vel_z  (-1.0, 1.0)
        #
        # So 0.3 left 40% of the linear range and 70% of the YAW range unreachable, which
        # is why turning felt disproportionately slower than walking. These are the trained
        # ceilings: going past them does not go faster, it goes out of distribution, and a
        # policy given a command it never saw tracks it worse rather than better.
        #
        # The real-robot path below still uses 0.3, deliberately. Do not unify them.
        _DIMENSO_VX_MAX  = 0.5
        _DIMENSO_VY_MAX  = 0.5
        _DIMENSO_YAW_MAX = 1.0

        def _dimenso_body_from_sticks(d):
            return (-d.left_ctrl_thumbstickValue[1]  * _DIMENSO_VX_MAX,
                    -d.left_ctrl_thumbstickValue[0]  * _DIMENSO_VY_MAX,
                    -d.right_ctrl_thumbstickValue[0] * _DIMENSO_YAW_MAX)

        def _dimenso_publish_body(vx, vy, yaw):
            if _dimenso_body_pub is None:
                return
            try:
                _dimenso_body_pub.Write(_DimensoStr(
                    data=str([float(vx), float(vy), float(yaw), _DIMENSO_STAND_HEIGHT])))
            except Exception:
                pass  # a publish failure must never take the control loop down

        # Both A buttons (left X + right A) held for ~0.4s engages. NOT the triggers
        # -- those already drive the Inspire hands, so a trigger gesture would clench
        # them at the instant of engage. NOT the thumbstick clicks -- upstream uses
        # both of those for Damp(). The hold requirement is so a brushed button
        # cannot engage a robot.
        _DIMENSO_ENGAGE_TICKS = 12          # x 0.033s ~= 0.4s
        _dimenso_engage_held = 0

        # ---- DIMENSO ADDITION: arms home on scene reset ----
        # A whole-scene reset restores the robot and the objects, but the ARMS are
        # commanded by this client at ~30Hz from the operator's wrist pose, so the sim
        # resets them and the very next tick drags them straight back. The reset has to
        # be honoured HERE, in the loop that owns the arm command.
        #
        # Done as a WINDOW the main loop reads, not by calling
        # arm_ctrl.ctrl_dual_arm_go_home() from the subscriber callback: that method
        # blocks for up to 100 attempts, and the main loop would overwrite q_target with
        # the IK solution on its next tick anyway. A flag has no contention.
        #
        # Deliberately does NOT disengage: after the window the arms follow the
        # operator's hands again, so a reset never silently drops them out of teleop.
        _DIMENSO_HOME_SECONDS = 1.5   # measured peak ~3.5 rad/s, so this reaches zero
        # numpy is NOT imported by this file, and `np.zeros(14)` would have raised
        # NameError inside the control loop -- a runtime-only failure py_compile cannot
        # see. An ndarray is genuinely required: clip_arm_q_target() does
        # `current_q + delta`, and upstream's own ctrl_dual_arm_go_home() passes
        # np.zeros(14).
        import numpy as _dimenso_np

        if args.sim:
            try:
                from unitree_sdk2py.core.channel import ChannelSubscriber as _DimensoSub

                def _dimenso_on_reset(msg):
                    global _DIMENSO_RESET_FLAG
                    try:
                        if str(msg.data).strip() == "2":     # whole scene only
                            _DIMENSO_RESET_FLAG = time.time() + _DIMENSO_HOME_SECONDS
                            logger_mp.info("[dimenso] whole-scene reset seen -- sending arms home")
                    except Exception:
                        pass

                _dimenso_reset_sub = _DimensoSub("rt/reset_pose/cmd", _DimensoStr)
                _dimenso_reset_sub.Init(_dimenso_on_reset, 10)
                logger_mp.info("[dimenso] watching rt/reset_pose/cmd for arms-home")
            except Exception as _e:
                logger_mp.error(f"[dimenso] could not subscribe to rt/reset_pose/cmd: {_e}")
        # ------------------------- END DIMENSO ADDITION -----------------------------

        # ---- DIMENSO ADDITION: collision haptics subscriber (DIM-555) ----
        # Deliberately created AFTER TeleVuerWrapper, unlike the monkey-patch above.
        # A DDS reader spawns its own listener thread, and threads do not survive
        # fork() -- creating it earlier would leave the Vuer child holding a copy of
        # the reader's file descriptors with nothing servicing them. The reset
        # subscriber above already proves this position works. Only the shared Values
        # and the patch have to be pre-fork; the SUBSCRIBER must not be.
        #
        # Domain is untouched: this reuses the participant ChannelFactoryInitialize
        # already created for domain 1 (sim). There is no ChannelFactoryInitialize
        # here, so no code path in this hunk can reach domain 0 -- the physical G1.
        if args.sim and _DIMENSO_HAPTICS_ENABLED and _dimenso_haptics_ready():
            try:
                from unitree_sdk2py.core.channel import ChannelSubscriber as _DimensoSub

                def _dimenso_on_haptics(msg):
                    # Never raises: _dimenso_haptics_ingest catches everything and
                    # logs once. A DDS callback that throws is a fail-open violation.
                    _dimenso_haptics_ingest(getattr(msg, "data", None))

                _dimenso_haptics_sub = _DimensoSub(_dimenso_haptics.TOPIC, _DimensoStr)
                # queue depth 1: latest-wins. A contact is only interesting while it is
                # current, and a backlog of onsets would replay stale taps after the
                # operator has already moved on.
                _dimenso_haptics_sub.Init(_dimenso_on_haptics, 1)
                logger_mp.info(
                    f"[dimenso] subscribed to {_dimenso_haptics.TOPIC} for collision haptics"
                )
            except Exception as _e:
                logger_mp.error(
                    f"[dimenso] could not subscribe to {_dimenso_haptics.TOPIC}: {_e!r} "
                    "-- continuing WITHOUT haptics"
                )
        # -------------- END DIMENSO ADDITION --------------

        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            # -------- DIMENSO ADDITION: engage from inside the headset --------
            # This loop previously read NO controller state, so the only way to
            # engage was a laptop keypress/IPC command -- impossible while holding
            # both controllers in the pose the robot is about to snap to.
            if args.input_mode == "controller":
                try:
                    _d = tv_wrapper.get_tele_data()
                    if _d.left_ctrl_aButton and _d.right_ctrl_aButton:
                        _dimenso_engage_held += 1
                        if _dimenso_engage_held >= _DIMENSO_ENGAGE_TICKS:
                            logger_mp.info("[dimenso] both A buttons held -- engaging")
                            START = True
                    else:
                        _dimenso_engage_held = 0
                except Exception:
                    pass
            # ---------------------- END DIMENSO ADDITION ----------------------
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()

        head_img = None
        left_wrist_img = None
        right_wrist_img = None

        # Loop-rate diagnostic (2026-07-21, investigating reported walking
        # latency/"just moves forward" symptom) -- logs once/sec whether the
        # control loop is keeping up with --frequency. Doesn't change behavior,
        # only visibility.
        _loop_diag_window_elapsed = []
        _loop_diag_last_log = time.time()

        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if args.ee in ("dex3", "inspire_ftp", "inspire_dfx", "brainco")  and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "brainco" and args.input_mode == "controller":
                with left_gripper_trigger_in.get_lock():
                    left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                with left_gripper_squeeze_in.get_lock():
                    left_gripper_squeeze_in.value = tele_data.left_ctrl_squeezeValue
                with right_gripper_trigger_in.get_lock():
                    right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                with right_gripper_squeeze_in.get_lock():
                    right_gripper_squeeze_in.value = tele_data.right_ctrl_squeezeValue
            elif args.ee == "inspire_dfx" and args.input_mode == "controller":
                # ---- DIMENSO ADDITION: the hand swap is PHYSICAL-ROBOT ONLY ----
                # The cross below was confirmed on the real robot (2026-07-21 live
                # test): right controller's trigger drives the left hand and vice
                # versa, matching PC2's own stack, which carries a
                # G1_TELEOP_SWAP_HANDS env var for apparently the same
                # characteristic of that setup.
                #
                # It is WRONG IN SIM. unitree_sim_isaaclab wires rt/inspire/cmd
                # straight through, so applying the hardware workaround here
                # produced exactly the symptom it exists to fix -- reported live
                # (2026-07-28): "the right one closes left palm and vice versa".
                #
                # Kept for the physical robot rather than deleted, because that
                # observation was made on hardware and this session has no way to
                # re-test it. --sim is the discriminator.
                if args.sim:
                    with left_gripper_trigger_in.get_lock():
                        left_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                    with right_gripper_trigger_in.get_lock():
                        right_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                else:
                    with left_gripper_trigger_in.get_lock():
                        left_gripper_trigger_in.value = tele_data.right_ctrl_triggerValue
                    with right_gripper_trigger_in.get_lock():
                        right_gripper_trigger_in.value = tele_data.left_ctrl_triggerValue
                # -------------------- END DIMENSO ADDITION --------------------
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready

            # ---- DIMENSO ADDITION: thumbstick locomotion (sim) ----
            # Ungated by --motion and --record on purpose: --motion redirects arms to
            # rt/arm_sdk which the sim never subscribes to, and current_body_action --
            # which computes this exact mapping -- is only built inside `if
            # args.record:`, so it is never available on our path.
            if args.sim and args.input_mode == "controller":
                _dv = _dimenso_body_from_sticks(tele_data)
                _dimenso_publish_body(*_dv)
            # -------------- END DIMENSO ADDITION --------------
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
            # ---- DIMENSO ADDITION: arms home on scene reset ----
            # Inside the window opened by a whole-scene reset, command ZERO instead of
            # the IK solution. Same target as ctrl_dual_arm_go_home(), but applied
            # through the loop that already owns q_target, so nothing fights it.
            if _DIMENSO_RESET_FLAG and time.time() < _DIMENSO_RESET_FLAG:
                arm_ctrl.ctrl_dual_arm(_dimenso_np.zeros(14), _dimenso_np.zeros(14))
            else:
                if _DIMENSO_RESET_FLAG:
                    _DIMENSO_RESET_FLAG = 0.0
                    logger_mp.info("[dimenso] arms-home window ended; following again")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            # -------------- END DIMENSO ADDITION --------------

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "inspire_dfx" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "brainco" and args.input_mode == "controller"):
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

            _loop_diag_window_elapsed.append(time_elapsed)
            if current_time - _loop_diag_last_log >= 1.0:
                target = 1.0 / args.frequency
                overruns = sum(1 for e in _loop_diag_window_elapsed if e > target)
                avg_ms = (sum(_loop_diag_window_elapsed) / len(_loop_diag_window_elapsed)) * 1000
                max_ms = max(_loop_diag_window_elapsed) * 1000
                actual_hz = len(_loop_diag_window_elapsed) / (current_time - _loop_diag_last_log)
                logger_mp.info(
                    f"[loop-diag] target={args.frequency:.1f}Hz actual={actual_hz:.1f}Hz "
                    f"avg_iter={avg_ms:.1f}ms max_iter={max_ms:.1f}ms overruns={overruns}/{len(_loop_diag_window_elapsed)}"
                )
                _loop_diag_window_elapsed = []
                _loop_diag_last_log = current_time

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        # ---- DIMENSO ADDITION: locomotion deadman ----
        # sharedmemorymanager.read_data() returns the last value FOREVER with no
        # staleness check, so a stale walk command keeps the robot walking after the
        # operator has gone. Zero it before anything else in teardown.
        try:
            if args.sim:
                _dimenso_publish_body(0.0, 0.0, 0.0)
                logger_mp.info("[dimenso] published zero body command (deadman)")
        except Exception as _e:
            logger_mp.error(f"[dimenso] deadman publish failed: {_e}")
        # -------------- END DIMENSO ADDITION --------------
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
