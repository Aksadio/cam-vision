"""Hand gestures and air drawing.

* One index finger pointing  -> draws a line in the air (names, letters, shapes).
* Two hands forming a heart  -> shows a big animated heart.
* Closed fist (held briefly) -> clears everything that was drawn.

The logic only needs MediaPipe hand landmarks (objects with a ``.landmark`` list
whose items have ``x`` and ``y`` between 0 and 1), so it is easy to test without
a camera.
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .config import CONFIG

# (knuckle/MCP index, fingertip index) for the four long fingers.
_FINGERS = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}

# BGR colours.
_PEN_CORE = (255, 240, 120)
_PEN_GLOW = (200, 110, 20)
_HEART_FILL = (120, 60, 255)
_HEART_EDGE = (215, 175, 255)
_MINI_HEART = (190, 130, 255)


def _xy(hand, index: int, w: int, h: int):
    point = hand.landmark[index]
    return point.x * w, point.y * h


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_size(hand, w: int, h: int) -> float:
    """Wrist to middle-finger knuckle distance in pixels (a size reference)."""
    return _dist(_xy(hand, 0, w, h), _xy(hand, 9, w, h))


def finger_ratios(hand, w: int, h: int) -> dict:
    """For each finger: fingertip-to-wrist distance divided by knuckle-to-wrist distance.

    A stretched finger gives about 1.7 to 2.0, a folded finger about 0.6 to 1.0.
    The ratio does not depend on hand rotation or distance from the camera.
    """
    wrist = _xy(hand, 0, w, h)
    ratios = {}
    for name, (knuckle, tip) in _FINGERS.items():
        d_knuckle = _dist(_xy(hand, knuckle, w, h), wrist)
        d_tip = _dist(_xy(hand, tip, w, h), wrist)
        ratios[name] = d_tip / d_knuckle if d_knuckle > 1e-6 else 0.0
    return ratios


def is_pointing(ratios: dict) -> bool:
    """Index finger out, middle/ring/pinky folded (thumb is ignored)."""
    return ratios["index"] > CONFIG.pointing_ratio and all(
        ratios[name] < CONFIG.folded_ratio for name in ("middle", "ring", "pinky")
    )


def is_fist(ratios: dict) -> bool:
    """All four long fingers folded."""
    return all(ratios[name] < CONFIG.folded_ratio for name in _FINGERS)


def heart_pose(hand_a, hand_b, w: int, h: int):
    """Return ``(center_x, center_y, size)`` if two hands make a heart, else ``None``.

    Heart = index fingertips touching at the top, thumb tips touching at the
    bottom, and the two wrists clearly apart (this separates it from praying hands).
    """
    size = (hand_size(hand_a, w, h) + hand_size(hand_b, w, h)) / 2.0
    if size < 8:
        return None
    idx_a, idx_b = _xy(hand_a, 8, w, h), _xy(hand_b, 8, w, h)
    thb_a, thb_b = _xy(hand_a, 4, w, h), _xy(hand_b, 4, w, h)
    wr_a, wr_b = _xy(hand_a, 0, w, h), _xy(hand_b, 0, w, h)
    idx_gap = _dist(idx_a, idx_b)
    thumb_gap = _dist(thb_a, thb_b)
    wrist_gap = _dist(wr_a, wr_b)
    top = (idx_a[1] + idx_b[1]) / 2.0
    bottom = (thb_a[1] + thb_b[1]) / 2.0

    if idx_gap > CONFIG.heart_idx_gap * size:
        return None
    if thumb_gap > CONFIG.heart_thumb_gap * size:
        return None
    if bottom - top < CONFIG.heart_min_height * size:
        return None
    if wrist_gap < CONFIG.heart_min_wrist_gap * size or wrist_gap < idx_gap * 1.3:
        return None
    cx = (idx_a[0] + idx_b[0] + thb_a[0] + thb_b[0]) / 4.0
    cy = (top + bottom) / 2.0
    return cx, cy, size


def _heart_points(cx: float, cy: float, width: float):
    t = np.linspace(0.0, 2.0 * np.pi, 90)
    x = 16.0 * np.sin(t) ** 3
    y = 13.0 * np.cos(t) - 5.0 * np.cos(2 * t) - 2.0 * np.cos(3 * t) - np.cos(4 * t)
    k = width / 32.0
    pts = np.stack([cx + x * k, cy - y * k], axis=1)
    return pts.astype(np.int32).reshape(-1, 1, 2)


class AirCanvas:
    """Keeps the drawn strokes and the gesture state between video frames."""

    def __init__(self) -> None:
        self.strokes: list[list[tuple[float, float]]] = []  # normalized (0..1) points
        self._active = False          # a stroke is currently being extended
        self._pen_down = False
        self._smooth = None           # smoothed fingertip in pixels
        self._cursor = None
        self._pen_frames = 0
        self._pen_missing = 0
        self._fist_frames = 0
        self._fist_latched = False
        self._heart_frames = 0
        self._heart_until = 0.0
        self._heart_pose = None
        self._notice = ""
        self._notice_until = 0.0

    # ------------------------------------------------------------------ state
    def clear(self) -> None:
        self.strokes = []
        self._active = False
        self._pen_down = False
        self._smooth = None
        self._cursor = None
        self._pen_frames = 0
        self._heart_frames = 0
        self._heart_until = 0.0
        self._heart_pose = None

    def update(self, hands, w: int, h: int) -> str:
        """Process the hands seen in one frame and return a short status label."""
        now = time.monotonic()
        hands = list(hands or [])
        ratios = [finger_ratios(hand, w, h) for hand in hands]

        heart = heart_pose(hands[0], hands[1], w, h) if len(hands) >= 2 else None
        if heart:
            self._heart_frames += 1
            if self._heart_frames >= CONFIG.heart_hold_frames:
                self._heart_pose = heart
                self._heart_until = now + CONFIG.heart_seconds
        else:
            self._heart_frames = 0

        fist_now = any(is_fist(r) for r in ratios) and not heart
        if fist_now:
            self._fist_frames += 1
            if self._fist_frames >= CONFIG.fist_hold_frames and not self._fist_latched:
                self.clear()
                self._fist_latched = True
                self._notice = "CLEARED"
                self._notice_until = now + 1.0
        else:
            self._fist_frames = 0
            if not any(is_fist(r) for r in ratios):
                self._fist_latched = False

        tip = None
        if not heart:
            candidates = [
                _xy(hand, 8, w, h) for hand, r in zip(hands, ratios) if is_pointing(r)
            ]
            if candidates:
                if self._active and self._smooth is not None:
                    tip = min(candidates, key=lambda p: _dist(p, self._smooth))
                else:
                    tip = candidates[0]
        self._update_pen(tip, w, h)

        if now < self._notice_until:
            return self._notice
        if now < self._heart_until:
            return "LOVE"
        if self._pen_down:
            return "DRAWING"
        if self._fist_frames > 0:
            return "FIST"
        return "HAND" if hands else "NO HAND"

    def _update_pen(self, tip, w: int, h: int) -> None:
        if tip is None:
            self._pen_frames = 0
            self._pen_missing += 1
            if self._pen_missing > CONFIG.pen_grace_frames:
                self._active = False
                self._pen_down = False
                self._smooth = None
                self._cursor = None
            return

        self._pen_missing = 0
        self._pen_frames += 1
        if self._smooth is None:
            smooth = tip
        else:
            a = CONFIG.pen_smoothing
            smooth = (a * tip[0] + (1 - a) * self._smooth[0],
                      a * tip[1] + (1 - a) * self._smooth[1])
            if _dist(smooth, self._smooth) > 0.3 * w:
                # Big jump (other hand took over): start a fresh stroke.
                smooth = tip
                self._active = False
        self._smooth = smooth
        self._cursor = smooth
        if self._pen_frames < CONFIG.pen_on_frames:
            return

        self._pen_down = True
        point = (smooth[0] / w, smooth[1] / h)
        if not self._active:
            self.strokes.append([point])
            self._active = True
        else:
            last = self.strokes[-1][-1]
            if math.hypot((point[0] - last[0]) * w, (point[1] - last[1]) * h) >= 2.0:
                self.strokes[-1].append(point)
        self._trim()

    def _trim(self) -> None:
        total = sum(len(stroke) for stroke in self.strokes)
        while total > CONFIG.max_stroke_points and len(self.strokes) > 1:
            total -= len(self.strokes.pop(0))

    # ---------------------------------------------------------------- drawing
    def render(self, frame) -> None:
        """Draw strokes, the heart effect, the pen cursor and notices into ``frame``."""
        h, w = frame.shape[:2]
        now = time.monotonic()
        self._render_strokes(frame, w, h)
        if self._heart_pose is not None and now < self._heart_until:
            self._render_heart(frame, now, w, h)
        if self._cursor is not None:
            center = (int(self._cursor[0]), int(self._cursor[1]))
            if self._pen_down:
                cv2.circle(frame, center, 9, _PEN_CORE, -1, cv2.LINE_AA)
            cv2.circle(frame, center, 13, (255, 255, 255), 2, cv2.LINE_AA)
        if now < self._notice_until:
            scale = 1.0
            (tw, th), _ = cv2.getTextSize(self._notice, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
            cv2.putText(frame, self._notice, ((w - tw) // 2, 70),
                        cv2.FONT_HERSHEY_DUPLEX, scale, (140, 220, 255), 2, cv2.LINE_AA)

    def _render_strokes(self, frame, w: int, h: int) -> None:
        thickness = max(2, CONFIG.pen_thickness)
        for stroke in self.strokes:
            if not stroke:
                continue
            pts = np.array([(int(x * w), int(y * h)) for x, y in stroke], dtype=np.int32)
            if len(pts) == 1:
                cv2.circle(frame, tuple(int(v) for v in pts[0]), thickness, _PEN_CORE, -1, cv2.LINE_AA)
                continue
            pts = pts.reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, _PEN_GLOW, thickness + 6, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, _PEN_CORE, thickness, cv2.LINE_AA)

    def _render_heart(self, frame, now: float, w: int, h: int) -> None:
        cx, cy, size = self._heart_pose
        alpha = max(0.0, min(1.0, (self._heart_until - now) / 0.6))
        pulse = 1.0 + 0.07 * math.sin(now * 8.0)
        width = max(150.0, min(4.2 * size, 0.6 * w)) * pulse

        overlay = frame.copy()
        pts = _heart_points(cx, cy, width)
        cv2.fillPoly(overlay, [pts], _HEART_FILL, cv2.LINE_AA)
        cv2.polylines(overlay, [pts], True, _HEART_EDGE, 4, cv2.LINE_AA)

        for i in range(6):
            phase = (now * 0.5 + i / 6.0) % 1.0
            mx = cx + (i - 2.5) * width * 0.28 + math.sin(now * 3.0 + i) * 12.0
            my = cy + width * 0.3 - phase * width * 1.3
            mini = _heart_points(mx, my, width * 0.14)
            cv2.fillPoly(overlay, [mini], _MINI_HEART, cv2.LINE_AA)

        label = "LOVE"
        font_scale = max(0.8, width / 220.0)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        cv2.putText(overlay, label, (int(cx - tw / 2), int(cy + th / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        blend = 0.85 * alpha
        cv2.addWeighted(overlay, blend, frame, 1.0 - blend, 0, frame)
