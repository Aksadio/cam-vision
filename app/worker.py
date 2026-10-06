"""Background camera/detection worker; the Tk event loop never blocks on inference."""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2
import mediapipe as mp

from .camera import Camera, CameraError
from .config import CONFIG
from .detectors.face import FaceDetector
from .detectors.hands import HandDetector
from .detectors.pose import PoseDetector
from .gestures import AirCanvas
from .renderer import OverlayRenderer


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _enhance_contrast(rgb):
    """Local contrast boost (on brightness only) so hands stand out in washed-out video."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class VisionWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(name="cam-vision-worker", daemon=True)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._settings = {"face": True, "hands": True, "pose": True, "mirror": True, "draw": True}
        self._clear_requested = False
        self._frame = None
        self._raw_frame = None
        self._metrics = {
            "camera": "STARTING", "engine": "STARTING", "fps": 0.0,
            "latency": 0.0, "faces": 0, "hands": 0, "pose": 0,
            "confidence": None, "width": 0, "height": 0, "error": "", "gesture": "--",
        }

    def set_setting(self, key: str, value: bool) -> None:
        with self._lock:
            if key in self._settings:
                self._settings[key] = bool(value)

    def request_clear(self) -> None:
        """Ask the worker thread to erase everything drawn in the air."""
        with self._lock:
            self._clear_requested = True

    def snapshot(self):
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
            metrics = dict(self._metrics)
        return frame, metrics

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        camera = None
        detectors = []
        try:
            camera = Camera()
            camera.open()
            with self._lock:
                self._metrics.update(camera="CONNECTED", width=camera.width, height=camera.height)

            # Model creation may be slow on first run; it happens here, never in Tk's UI thread.
            face = FaceDetector()
            hands = HandDetector()
            pose = PoseDetector()
            detectors = [face, hands, pose]
            renderer = OverlayRenderer()
            canvas = AirCanvas()
            previous_mirror = True
            with self._lock:
                self._metrics.update(engine="ACTIVE", error="")

            frame_times = deque(maxlen=20)
            frame_index = 0
            last_face = None
            last_pose = None
            while not self._stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    with self._lock:
                        self._metrics.update(camera="ERROR", error="Camera stopped returning valid frames.")
                    time.sleep(0.08)
                    continue

                started = time.perf_counter()
                with self._lock:
                    settings = dict(self._settings)
                    clear_requested = self._clear_requested
                    self._clear_requested = False
                if clear_requested or settings["mirror"] != previous_mirror:
                    # Flipping the mirror would leave old drawings on the wrong side.
                    canvas.clear()
                previous_mirror = settings["mirror"]
                # Mirror before inference so normalized landmarks stay aligned with the
                # natural mirror-view image rendered for the user.
                if settings["mirror"]:
                    frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                # Downscale inference input on larger webcams to keep CPU usage modest.
                scale = min(1.0, CONFIG.detector_width / float(width))
                if scale < 1.0:
                    infer = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
                else:
                    infer = frame
                rgb = cv2.cvtColor(infer, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                frame_index += 1
                # The face and pose models are slow, so they run on every Nth frame (taking
                # turns) and the latest result is reused. Hands run on every frame because
                # drawing needs them to be smooth.
                if settings["face"]:
                    if last_face is None or frame_index % CONFIG.face_every == 0:
                        last_face = face.process(rgb)
                    face_result = last_face
                else:
                    last_face = face_result = None
                if settings["pose"]:
                    if last_pose is None or frame_index % CONFIG.pose_every == CONFIG.pose_every // 2:
                        last_pose = pose.process(rgb)
                    pose_result = last_pose
                else:
                    last_pose = pose_result = None
                # Hands are also needed for air drawing, even if the skeleton is hidden.
                need_hands = settings["hands"] or settings["draw"]
                if need_hands:
                    hand_input = _enhance_contrast(rgb) if CONFIG.hand_enhance_contrast else rgb
                    hand_result = hands.process(hand_input)
                else:
                    hand_result = None
                rgb.flags.writeable = True

                faces_count = len(face_result.multi_face_landmarks or []) if face_result else 0
                hands_count = (
                    len(hand_result.multi_hand_landmarks or [])
                    if hand_result and settings["hands"] else 0
                )
                pose_count = int(bool(pose_result and pose_result.pose_landmarks))
                # FaceMesh does not expose per-face confidence; report available hand/pose
                # classifier/visibility scores instead and use an em dash when unavailable.
                scores = []
                if hand_result and hand_result.multi_handedness:
                    for item in hand_result.multi_handedness:
                        if item.classification:
                            scores.append(float(item.classification[0].score))
                if pose_result and pose_result.pose_landmarks:
                    visible = [lm.visibility for lm in pose_result.pose_landmarks.landmark if lm.visibility is not None]
                    if visible:
                        scores.append(sum(visible) / len(visible))
                confidence = (sum(scores) / len(scores)) if scores else None

                frame = renderer.draw(frame, face_result, hand_result, pose_result, settings)
                if settings["draw"]:
                    try:
                        hand_list = hand_result.multi_hand_landmarks if hand_result else None
                        gesture = canvas.update(hand_list, width, height)
                        canvas.render(frame)
                    except Exception as exc:  # keep the live video running no matter what
                        gesture = "ERROR"
                        with self._lock:
                            self._metrics.update(error=f"Gesture error: {type(exc).__name__}: {exc}")
                else:
                    canvas.clear()
                    gesture = "OFF"
                renderer.add_frame_label(frame, "LIVE  /  CAM VISION")
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                frame_times.append(time.perf_counter())
                fps = 0.0
                if len(frame_times) > 1:
                    span = frame_times[-1] - frame_times[0]
                    fps = (len(frame_times) - 1) / span if span > 0 else 0.0
                with self._lock:
                    self._frame = frame
                    self._metrics.update(
                        camera="CONNECTED", engine="ACTIVE", fps=fps, latency=elapsed_ms,
                        faces=faces_count, hands=hands_count, pose=pose_count,
                        confidence=confidence, width=width, height=height, error="", gesture=gesture,
                    )
        except CameraError as exc:
            self._set_error("CAMERA UNAVAILABLE", str(exc))
        except Exception as exc:  # Surface model initialization/runtime failures to the UI.
            self._set_error("ENGINE ERROR", f"{type(exc).__name__}: {exc}")
        finally:
            for detector in detectors:
                try:
                    detector.close()
                except Exception:
                    pass
            if camera is not None:
                camera.close()

    def _set_error(self, camera_status: str, message: str) -> None:
        with self._lock:
            self._metrics.update(camera=camera_status, engine="ERROR", error=message)
