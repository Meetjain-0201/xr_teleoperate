"""Graceful completion acknowledgment never precedes measured home/handback."""
import pytest
from teleop.utils.dimenso_shutdown import Shutdown


def test_ack_refused_until_complete_and_failure_retained():
    s=Shutdown();s.request()
    with pytest.raises(RuntimeError):s.ack()
    s.finish({'completed':False,'reason':'stale','homeReached':False,'handbackPublished':False})
    assert s.snapshot()['SHUTDOWN_STATE']=='failed'
    with pytest.raises(RuntimeError):s.ack()
    assert not s.acknowledged.is_set()


def test_ack_only_after_both_evidence_fields():
    s=Shutdown();s.request()
    s.finish({'completed':True,'homeReached':True,'handbackPublished':True})
    assert s.snapshot()['ARM_HOME_REACHED']
    assert s.snapshot()['ARM_HANDBACK_PUBLISHED']
    s.ack();assert s.acknowledged.is_set()


def test_inconsistent_completion_refused():
    s=Shutdown();s.request()
    s.finish({'completed':True,'homeReached':False,'handbackPublished':True})
    assert s.snapshot()['SHUTDOWN_STATE']=='failed'


def test_shutdown_locks_out_start_and_record_commands():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    tree = ast.parse((Path(__file__).parents[1] / 'teleop/teleop_hand_and_arm.py').read_text())
    on_press = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'on_press')
    shutdown = Shutdown()
    shutdown.supported = True
    scope = dict(_DIMENSO_SHUTDOWN=shutdown, START=False, STOP=False,
                 RECORD_TOGGLE=False, RECORD_DISCARD=False,
                 logger_mp=SimpleNamespace(warning=lambda *a: None))
    exec(compile(ast.Module(body=[on_press], type_ignores=[]), '<on_press>', 'exec'), scope)
    press = scope['on_press']
    press('graceful_disconnect')
    assert scope['STOP'] and not scope['START']
    for command in ['r', 'q', 's', 'd']:
        with pytest.raises(RuntimeError):
            press(command)
    assert not scope['START']
    with pytest.raises(RuntimeError):
        press('shutdown_ack')
    shutdown.finish(dict(completed=True, homeReached=True, handbackPublished=True))
    press('shutdown_ack')
    assert shutdown.acknowledged.is_set()


def test_ipc_routes_request_and_rejects_early_ack_without_hardware_imports():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    source = Path(__file__).parents[1] / 'teleop/utils/ipc.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IPC_Server')
    cls.body = [n for n in cls.body if isinstance(n, ast.Assign) or
                isinstance(n, ast.FunctionDef) and n.name == '_handle_message']
    scope = {}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), scope)
    state = Shutdown()
    def press(command):
        {'graceful_disconnect': state.request, 'shutdown_ack': state.ack}[command]()
    server = scope['IPC_Server']()
    server.on_press = press
    assert server._handle_message({'reqid': '1', 'cmd': 'CMD_DISCONNECT_GRACEFUL'})['status'] == 'ok'
    assert state.requested
    assert server._handle_message({'reqid': '2', 'cmd': 'CMD_SHUTDOWN_ACK'})['status'] == 'error'
    state.finish(dict(completed=True, homeReached=True, handbackPublished=True))
    assert server._handle_message({'reqid': '3', 'cmd': 'CMD_SHUTDOWN_ACK'})['status'] == 'ok'
    assert state.acknowledged.is_set()


def test_body_stop_requires_ack_and_restores_timeout():
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).parents[1] / 'teleop/utils/motion_switcher.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LocoClientWrapper')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'stop_for_shutdown')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    for code in [0, None, 3104]:
        calls = []
        client = SimpleNamespace(SetTimeout=lambda t: calls.append(t),
                                 SetVelocity=lambda *v: calls.append(v) or code)
        wrapper = SimpleNamespace(client=client)
        if code == 0:
            scope['stop_for_shutdown'](wrapper)
        else:
            with pytest.raises(RuntimeError):
                scope['stop_for_shutdown'](wrapper)
        assert calls == [1.0, (0.0, 0.0, 0.0), 0.0001]


@pytest.mark.parametrize("stop_error", [None, RuntimeError("stop timeout")])
def test_failed_finalizer_stops_body_without_homing_or_clearing_failure(stop_error):
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).parents[1] / 'teleop/teleop_hand_and_arm.py'
    tree = ast.parse(path.read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == '_DIMENSO_SHUTDOWN.requested')
    state = Shutdown()
    state.request()
    state.finish({'completed': False, 'reason': 'invalid controller input'})
    state.acknowledged = SimpleNamespace(wait=lambda _: True)
    calls = []
    def stop_body():
        calls.append('stop')
        if stop_error is not None:
            raise stop_error
    scope = dict(_DIMENSO_SHUTDOWN=state,
                 arm_ctrl=SimpleNamespace(ctrl_dual_arm_go_home=lambda: calls.append('legacy'),
                     ctrl_dual_arm_graceful_disconnect=lambda **kw: calls.append('home')),
                 loco_wrapper=SimpleNamespace(stop_for_shutdown=stop_body),
                 logger_mp=SimpleNamespace(exception=lambda *a: None, error=lambda *a: None))
    exec(compile(ast.Module(body=[branch], type_ignores=[]), str(path), 'exec'), scope)
    assert calls == ['stop']
    assert state.snapshot()['SHUTDOWN_STATE'] == 'failed'
    assert 'invalid controller input' in state.snapshot()['SHUTDOWN_ERROR']
    if stop_error is not None:
        assert 'stop timeout' in state.snapshot()['SHUTDOWN_ERROR']
