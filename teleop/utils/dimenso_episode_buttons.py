"""Translate Quest face buttons into replayable real-robot recording intents.

The teleop loop owns sampling. Its IPC heartbeat publishes these numbered events;
the website recorder owns whether each intent is accepted. No episode data is
written here, and a button never sends a robot mode or motion command.
"""

from collections import deque
from threading import Lock


class EpisodeButtons:
    """Edge-triggered B and A, held Y and X+A, with no engage-release quit."""

    def __init__(self, hold_ticks=12):
        self.hold_ticks = hold_ticks
        self.paused = False
        self.events = deque(maxlen=64)
        self._events_lock = Lock()
        self._seq = 0
        self._b_ticks = 0
        self._a_down = False
        self._a_neutral = False
        self._y_ticks = 0
        self._resume_ticks = 0

    def _emit(self, kind):
        with self._events_lock:
            self._seq += 1
            self.events.append({"seq": self._seq, "kind": kind})

    def snapshot(self):
        with self._events_lock:
            return list(self.events)

    def sample(self, *, x, a, y, b, recording=False):
        """Return intents fired this tick; a single press cannot fire twice."""
        before = self._seq
        if not x and not a:
            self._a_neutral = True

        if self.paused:
            self._resume_ticks = self._resume_ticks + 1 if x and a else 0
            if self._resume_ticks == self.hold_ticks:
                self.paused = False
                self._a_neutral = False
                self._emit("resume")
        elif a and not x and not self._a_down and self._a_neutral:
            self.paused = True
            self._a_neutral = False
            self._resume_ticks = 0
            self._emit("pause")

        if b:
            self._b_ticks += 1
            if self._b_ticks == self.hold_ticks * 5:
                self._emit("end")
        elif self._b_ticks:
            if self._b_ticks < self.hold_ticks * 5 and not self.paused:
                self._emit("toggle")
                if recording:
                    self.paused = True
            self._b_ticks = 0
        self._a_down = a

        self._y_ticks = self._y_ticks + 1 if y else 0
        if self._y_ticks == self.hold_ticks:
            self._emit("discard")
            self.paused = True
        return [event["kind"] for event in self.events if event["seq"] > before]
