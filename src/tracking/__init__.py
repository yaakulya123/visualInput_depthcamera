# Multi-Person Tracking Module
from .keypoint_adapter import coco_to_mediapipe, FakeLandmarks, FakeLandmark
from .person_state import (
    PersonState,
    MultiPersonState,
    PersonManager,
    PERSON_COLORS,
)
from .person_tracker import PersonTracker

# RealSense 3D Skeleton Tracking
try:
    from .realsense_skeleton import (
        RealSenseSkeletonTracker,
        Skeleton3D,
        Landmark3D,
        OneEuroFilter3D,
        draw_skeleton_3d,
    )
except ImportError:
    # pyrealsense2 not available
    pass
