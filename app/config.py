"""Shared application defaults."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    requested_width: int = 640
    requested_height: int = 480
    fallback_width: int = 640
    fallback_height: int = 480
    detector_width: int = 640       # smaller = faster (good for low-RAM/old laptops)
    max_faces: int = 2
    max_hands: int = 2
    detection_confidence: float = 0.5
    tracking_confidence: float = 0.5
    hand_detection_confidence: float = 0.4  # lower = hands are found more easily
    hand_tracking_confidence: float = 0.4
    hand_enhance_contrast: bool = True      # boosts local contrast before hand detection (helps washed-out/backlit video)
    face_every: int = 4             # run the (heavy) face model on every Nth frame and reuse the result in between
    pose_every: int = 4             # same for body pose; hands are checked on every frame
    window_title: str = "CAM VISION — AI Computer Vision System"
    window_min_width: int = 1100
    window_min_height: int = 700

    # --- Face overlay -------------------------------------------------------
    # False = only the clean outline (face edge, eyes, brows, lips, irises).
    # True  = also draw the dense full mesh (many extra lines).
    face_full_mesh: bool = False

    # --- Air drawing (index finger = pen) ----------------------------------
    pen_thickness: int = 4          # line width in pixels
    pen_smoothing: float = 0.55     # 0..1, lower = smoother but laggier
    pen_on_frames: int = 2          # frames the pointing pose must stay before ink starts
    pen_grace_frames: int = 4       # tracking dropouts shorter than this keep the stroke
    max_stroke_points: int = 8000   # oldest strokes are dropped beyond this
    pointing_ratio: float = 1.5     # fingertip-to-wrist / knuckle-to-wrist: above this = finger stretched out
    folded_ratio: float = 1.3       # below this = finger folded

    # --- Closed fist = clear everything ------------------------------------
    fist_hold_frames: int = 10      # hold the fist this many frames (about 0.4 s)

    # --- Two-hand heart -----------------------------------------------------
    heart_hold_frames: int = 5      # hold the heart pose this many frames
    heart_seconds: float = 3.0      # how long the heart stays after the pose
    heart_idx_gap: float = 1.0      # max gap between index tips (x hand size)
    heart_thumb_gap: float = 1.2    # max gap between thumb tips (x hand size)
    heart_min_height: float = 0.35  # index tips must be this much above thumb tips
    heart_min_wrist_gap: float = 0.9  # wrists must be at least this far apart


CONFIG = AppConfig()
