"""Render MediaPipe landmarks in a restrained neon dashboard style."""

from __future__ import annotations

import cv2
import mediapipe as mp

from .config import CONFIG


class OverlayRenderer:
    def __init__(self) -> None:
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

    def draw(self, frame, face_result, hand_result, pose_result, enabled):
        """Draw selected landmark sets into a BGR frame in place and return it."""
        if enabled.get("face") and face_result and face_result.multi_face_landmarks:
            for landmarks in face_result.multi_face_landmarks:
                if CONFIG.face_full_mesh:
                    # Optional dense mesh (many lines); off by default.
                    self.mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        self.mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(
                            color=(115, 72, 42), thickness=1, circle_radius=1
                        ),
                    )
                # Clean look: only the face outline, eyes, brows, lips and irises.
                for contour in (
                    self.mp_face.FACEMESH_CONTOURS,
                    self.mp_face.FACEMESH_IRISES,
                ):
                    self.mp_draw.draw_landmarks(
                        frame,
                        landmarks,
                        contour,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_draw.DrawingSpec(
                            color=(255, 130, 235), thickness=1, circle_radius=1
                        ),
                    )

        if enabled.get("hands") and hand_result and hand_result.multi_hand_landmarks:
            for landmarks in hand_result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(255, 240, 190), thickness=-1, circle_radius=3
                    ),
                    connection_drawing_spec=self.mp_draw.DrawingSpec(
                        color=(80, 235, 255), thickness=2, circle_radius=2
                    ),
                )

        if enabled.get("pose") and pose_result and pose_result.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(210, 240, 255), thickness=-1, circle_radius=3
                ),
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(255, 155, 75), thickness=3, circle_radius=2
                ),
            )

        return frame

    @staticmethod
    def add_frame_label(frame, text: str) -> None:
        """Add a readable, unobtrusive live indicator to the preview/capture."""
        cv2.rectangle(frame, (14, 14), (174, 46), (18, 20, 36), thickness=-1)
        cv2.putText(
            frame, text, (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (190, 245, 255), 1, cv2.LINE_AA
        )
