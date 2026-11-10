"""Sample controller provenance and applied mapping for the existing IPC heartbeat.

Does not command hardware. Collaborators: TeleData event receipt and the main
loop's already-written hand inputs. Heartbeats are sampled, not an event log.
"""
import copy
import math
import threading
import time


class InputDiagnostics:
    def __init__(self):
        self._lock=threading.Lock()
        self._sample=None
        self._at=None

    def update(self,raw,*,left,right,mode,paused,engaged,now=None):
        with self._lock:
            if raw is None:
                self._sample=None
                return
            previous=(self._sample or {}).get('applied') or {}
            sequence=raw['sequence'] if mode=='controller' else previous.get('sourceSequence') if mode=='hold' and (self._sample or {}).get('streamId')==raw['streamId'] else None
            def hand(values):
                return {key:float(value) if type(value) in (int,float) and math.isfinite(value) else None
                        for key,value in zip(('triggerValue','squeezeValue'),values)}
            self._sample=copy.deepcopy(raw)
            self._sample['applied']={'sourceSequence':sequence,'mode':mode,'left':hand(left),'right':hand(right),
                                     'followPaused':bool(paused),'engaged':bool(engaged)}
            self._at=time.monotonic() if now is None else now

    def snapshot(self,*,now=None):
        with self._lock:
            if self._sample is None:return None
            result=copy.deepcopy(self._sample)
            result['sourceAgeSeconds']+=max(0,(time.monotonic() if now is None else now)-self._at)
            return result
