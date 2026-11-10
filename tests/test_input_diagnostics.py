"""Pure heartbeat diagnostics: sampled provenance and mapped/held values."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('diagnostics',Path(__file__).parents[1]/'teleop/utils/dimenso_input_diagnostics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def sample(seq=1):return {'schemaVersion':1,'source':'vuer-controller-event','streamId':'one','sequence':seq,'receivedAtUnixMs':1000,'sourceAgeSeconds':0.5,'left':{'triggerValue':.2,'squeezeValue':.3},'right':{'triggerValue':.9,'squeezeValue':.8}}

def test_heartbeat_ages_event_and_hold_retains_applied_source():
    state=module.InputDiagnostics();assert state.snapshot(now=10) is None
    state.update(sample(),left=(8,.3),right=(1,.8),mode='controller',paused=False,engaged=True,now=10)
    first=state.snapshot(now=11)
    assert first['sourceAgeSeconds']==1.5 and first['applied']['sourceSequence']==1
    state.update(sample(2),left=(8,.3),right=(1,.8),mode='hold',paused=True,engaged=True,now=12)
    held=state.snapshot(now=12)
    assert held['sequence']==2 and held['applied']['sourceSequence']==1
    assert held['applied']['left']['triggerValue']==8
    state.update(sample(3),left=(10,0),right=(10,0),mode='home-open',paused=True,engaged=False,now=13)
    assert state.snapshot(now=13)['applied']['sourceSequence'] is None


def test_hold_after_source_restart_has_unknown_source_sequence():
    state=module.InputDiagnostics()
    state.update(sample(),left=(8,.3),right=(1,.8),mode='controller',paused=False,engaged=True,now=10)
    restarted=sample();restarted['streamId']='two'
    state.update(restarted,left=(8,.3),right=(1,.8),mode='hold',paused=True,engaged=True,now=11)
    assert state.snapshot(now=11)['applied']['sourceSequence'] is None
