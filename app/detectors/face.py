"""MediaPipe face mesh detector."""

import mediapipe as mp

from ..config import CONFIG


class FaceDetector:
    def __init__(self) -> None:
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=CONFIG.max_faces,
            refine_landmarks=True,
            min_detection_confidence=CONFIG.detection_confidence,
            min_tracking_confidence=CONFIG.tracking_confidence,
        )

    def process(self, rgb_frame):
        return self.mesh.process(rgb_frame)

    def close(self) -> None:
        self.mesh.close()
