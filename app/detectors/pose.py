"""MediaPipe lightweight full-body pose detector."""

import mediapipe as mp

from ..config import CONFIG


class PoseDetector:
    def __init__(self) -> None:
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=CONFIG.detection_confidence,
            min_tracking_confidence=CONFIG.tracking_confidence,
        )

    def process(self, rgb_frame):
        return self.pose.process(rgb_frame)

    def close(self) -> None:
        self.pose.close()
