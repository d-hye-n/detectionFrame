"""Camera-independent gesture state and half-open pixel geometry."""
from dataclasses import dataclass
from enum import Enum
from math import dist


def rectangle(a, b, width, height):
    x1, x2 = sorted((max(0, min(width, a[0])), max(0, min(width, b[0]))))
    y1, y2 = sorted((max(0, min(height, a[1])), max(0, min(height, b[1]))))
    return x1, y1, x2, y2


def global_box(local, roi):
    x1, y1, x2, y2 = roi
    a, b, c, d = local
    return (max(x1, min(x2, round(a+x1))), max(y1, min(y2, round(b+y1))),
            max(x1, min(x2, round(c+x1))), max(y1, min(y2, round(d+y1))))


class State(str, Enum):
    IDLE = "IDLE"
    DRAWING = "DRAWING"
    LOCKED = "LOCKED"


@dataclass
class ROIController:
    min_size: int = 40
    debounce: float = 0.10
    lost_timeout: float = 0.40
    state: State = State.IDLE
    anchor: tuple | None = None
    cursor: tuple | None = None
    locked: tuple | None = None
    _pinched: bool = False
    _pending: bool | None = None
    _pending_since: float = 0.0
    _last_seen: float | None = None
    _smooth: tuple | None = None

    def reset(self):
        self.state = State.IDLE
        self.anchor = self.cursor = self.locked = None
        self._pinched = False
        self._pending = None
        self._last_seen = self._smooth = None

    def begin(self, point):
        self.locked = None
        self.anchor = self.cursor = point
        self.state = State.DRAWING

    def finish(self, point, width, height):
        if self.anchor is None:
            return
        box = rectangle(self.anchor, point, width, height)
        if box[2]-box[0] < self.min_size or box[3]-box[1] < self.min_size:
            self.reset()
            return
        self.cursor, self.locked, self.state = point, box, State.LOCKED

    def update(self, landmarks, width, height, now):
        if self.state == State.LOCKED:
            return
        if not landmarks:
            self._pending = None
            if self._last_seen is not None and now-self._last_seen >= self.lost_timeout:
                self.reset()  # Tracking loss must not confirm the ROI.
            return
        self._last_seen = now
        points = [(p.x*width, p.y*height) for p in landmarks]
        ratio = dist(points[4], points[8]) / max(dist(points[0], points[9]), 1.0)
        pinched = ratio < (0.45 if self._pinched else 0.28)
        midpoint = tuple((points[4][i]+points[8][i])/2 for i in (0, 1))
        self._smooth = midpoint if self._smooth is None else tuple(
            0.4*n+0.6*old for n, old in zip(midpoint, self._smooth))
        point = tuple(round(v) for v in self._smooth)
        # Freeze the corner during finger release to prevent release drift.
        if self.state == State.DRAWING and pinched:
            self.cursor = point
        if pinched == self._pinched:
            self._pending = None
            return
        if self._pending != pinched:
            self._pending, self._pending_since = pinched, now
            return
        if now-self._pending_since < self.debounce:
            return
        self._pinched, self._pending = pinched, None
        if pinched:
            self.begin(point)
        elif self.state == State.DRAWING:
            self.finish(self.cursor, width, height)

    def preview(self, width, height):
        if self.locked:
            return self.locked
        if self.anchor is not None and self.cursor is not None:
            return rectangle(self.anchor, self.cursor, width, height)
        return None
