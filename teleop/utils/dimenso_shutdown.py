"""Thread-safe shutdown evidence shared by the control finalizer and IPC.

Completion means home measured and handback published, never a torque-off claim.
The process remains observable until the backend acknowledges this evidence.
"""
import threading


class Shutdown:
    def __init__(self):
        self.lock=threading.Lock()
        self.requested=False
        self.acknowledged=threading.Event()
        self.state='idle'
        self.error=None
        self.home=False
        self.handback=False
        self.supported=False

    def request(self):
        with self.lock:
            self.requested=True
            if self.state=='idle':self.state='requested'

    def finish(self,result):
        with self.lock:
            self.home=result.get('homeReached') is True
            self.handback=result.get('handbackPublished') is True
            complete=result.get('completed') is True and self.home and self.handback
            self.state='complete' if complete else 'failed'
            self.error=None if complete else result.get('reason') or 'home or handback evidence missing'

    def ack(self):
        with self.lock:
            if self.state!='complete' or not self.home or not self.handback:
                raise RuntimeError('shutdown has not completed with home and handback evidence')
            self.acknowledged.set()

    def snapshot(self):
        with self.lock:
            return {'GRACEFUL_DISCONNECT_SUPPORTED':self.supported,
                    'SHUTDOWN_STATE':self.state,'SHUTDOWN_ERROR':self.error,
                    'ARM_HOME_REACHED':self.home,'ARM_HANDBACK_PUBLISHED':self.handback}
