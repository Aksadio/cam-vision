"""MediaPipe two-hand landmark detector."""

import mediapipe as mp

from ..config import CONFIG


class HandDetector:
    def __init__(self) -> None:
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=CONFIG.max_hands,
            model_complexity=0,
            min_detection_confidence=CONFIG.hand_detection_confidence,
            min_tracking_confidence=CONFIG.hand_tracking_confidence,
        )

    def process(self, rgb_frame):
        return self.hands.process(rgb_frame)

    def close(self) -> None:
        self.hands.close()
