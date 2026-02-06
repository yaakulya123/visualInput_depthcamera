#!/usr/bin/env python3
"""
Keypoint Adapter - Converts YOLO COCO keypoints to MediaPipe-compatible landmarks.

YOLO11-pose outputs 17 COCO keypoints per person.
The existing BreathingDetector and StillnessDetector expect MediaPipe's 33-landmark format.
This adapter bridges that gap so we don't need to change any existing detector code.

COCO 17 keypoints:
  0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
  5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
  9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
  13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

MediaPipe 33 landmarks (subset that matters):
  0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow,
  14: right_elbow, 15: left_wrist, 16: right_wrist, 23: left_hip,
  24: right_hip, 25: left_knee, 26: right_knee, 27: left_ankle,
  28: right_ankle
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# COCO index -> MediaPipe index mapping
COCO_TO_MEDIAPIPE = {
    0: 0,    # nose -> nose
    1: 2,    # left_eye -> left_eye (inner)
    2: 5,    # right_eye -> right_eye (inner)
    3: 7,    # left_ear -> left_ear
    4: 8,    # right_ear -> right_ear
    5: 11,   # left_shoulder -> left_shoulder
    6: 12,   # right_shoulder -> right_shoulder
    7: 13,   # left_elbow -> left_elbow
    8: 14,   # right_elbow -> right_elbow
    9: 15,   # left_wrist -> left_wrist
    10: 16,  # right_wrist -> right_wrist
    11: 23,  # left_hip -> left_hip
    12: 24,  # right_hip -> right_hip
    13: 25,  # left_knee -> left_knee
    14: 26,  # right_knee -> right_knee
    15: 27,  # left_ankle -> left_ankle
    16: 28,  # right_ankle -> right_ankle
}

NUM_MEDIAPIPE_LANDMARKS = 33


class FakeLandmark:
    """Mimics a single MediaPipe NormalizedLandmark."""
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0,
                 visibility: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class FakeLandmarks:
    """
    Mimics MediaPipe's pose_landmarks object.

    Provides .landmark list with 33 entries so existing detectors
    can access landmarks.landmark[11], etc. without changes.
    """

    def __init__(self):
        self.landmark = [FakeLandmark() for _ in range(NUM_MEDIAPIPE_LANDMARKS)]


def coco_to_mediapipe(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    frame_width: int,
    frame_height: int,
    confidence_threshold: float = 0.3,
) -> Optional[FakeLandmarks]:
    """
    Convert YOLO COCO keypoints to a MediaPipe-compatible landmarks object.

    Args:
        keypoints: (17, 2) array of pixel coordinates [x, y] from YOLO.
        confidences: (17,) array of keypoint confidences from YOLO.
        frame_width: Frame width in pixels (for normalization to 0-1).
        frame_height: Frame height in pixels (for normalization to 0-1).
        confidence_threshold: Minimum confidence to mark a landmark as visible.

    Returns:
        FakeLandmarks object compatible with existing detectors, or None if
        too few keypoints are visible.
    """
    if keypoints is None or len(keypoints) < 17:
        return None

    # Count visible keypoints
    visible_count = np.sum(confidences > confidence_threshold)
    if visible_count < 4:
        return None

    landmarks = FakeLandmarks()

    for coco_idx, mp_idx in COCO_TO_MEDIAPIPE.items():
        conf = float(confidences[coco_idx])
        if conf > confidence_threshold:
            x_pixel, y_pixel = keypoints[coco_idx]
            landmarks.landmark[mp_idx] = FakeLandmark(
                x=float(x_pixel) / frame_width,
                y=float(y_pixel) / frame_height,
                z=0.0,
                visibility=conf,
            )

    return landmarks


def get_shoulder_center(keypoints: np.ndarray, confidences: np.ndarray,
                        confidence_threshold: float = 0.3) -> Optional[tuple]:
    """
    Get the center point between shoulders in pixel coordinates.
    Useful for quick breathing signal extraction without full adapter.

    Returns:
        (x, y) pixel coordinates or None if shoulders not visible.
    """
    left_conf = confidences[5]
    right_conf = confidences[6]

    if left_conf < confidence_threshold and right_conf < confidence_threshold:
        return None

    if left_conf >= confidence_threshold and right_conf >= confidence_threshold:
        cx = (keypoints[5][0] + keypoints[6][0]) / 2
        cy = (keypoints[5][1] + keypoints[6][1]) / 2
    elif left_conf >= confidence_threshold:
        cx, cy = keypoints[5]
    else:
        cx, cy = keypoints[6]

    return (float(cx), float(cy))
