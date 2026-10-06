"""Laptop webcam access with resolution negotiation and readable errors."""

from __future__ import annotations

import cv2

from .config import CONFIG


class CameraError(RuntimeError):
    """Raised when the requested webcam cannot produce a valid frame."""


class Camera:
    def __init__(self, index: int = CONFIG.camera_index) -> None:
        self.index = index
        self.capture: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0

    def open(self) -> None:
        # DirectShow is generally the most reliable backend for laptop webcams on Windows.
        if hasattr(cv2, "CAP_DSHOW"):
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.index)
        else:
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                "Could not open camera 0. Check that the built-in webcam is connected, "
                "not in use by another app, and allowed in Windows camera privacy settings."
            )

        self.capture = cap
        # Try HD first. If the driver negotiates another supported size, accept it; if
        # the first read fails, retry at the common laptop resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.requested_height)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.fallback_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.fallback_height)
            ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            self.close()
            raise CameraError(
                "The webcam opened, but did not provide a valid frame. Check camera permission "
                "and try closing other camera applications."
            )
        self.height, self.width = frame.shape[:2]

    def read(self):
        if self.capture is None:
            return False, None
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            return False, None
        return True, frame

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
