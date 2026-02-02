#!/usr/bin/env python3
"""
Stillness Detector - Body Stillness/Jitter Detection for Liquid Stillness

This module provides comprehensive stillness detection for the biofeedback system.
It detects body movement and restlessness, outputting a "jitter score" that drives
the fluid turbulence in the visualization.

Key Features:
- Multi-method detection (pose landmarks + optical flow fallback)
- One-Euro filtering for smooth, responsive tracking
- Regional body analysis (arms, torso, legs)
- Temporal analysis for sustained stillness detection
- Adaptive thresholds for different sensitivity levels

Based on research:
- Motion Tracker (PLOS ONE): https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130293
- One-Euro Filter: https://gery.casiez.net/1euro/
- Body Posture Stability Detection: https://pmc.ncbi.nlm.nih.gov/articles/PMC3571857/

Output:
- jitter_score: 0.0 (perfectly still) to 1.0 (highly restless)
- This score drives fluid turbulence in the visualization
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time

from .one_euro_filter import LandmarkSmoother, get_preset


class BodyRegion(Enum):
    """Body regions for regional motion analysis."""
    HEAD = "head"
    TORSO = "torso"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    LEFT_LEG = "left_leg"
    RIGHT_LEG = "right_leg"
    FULL_BODY = "full_body"


# MediaPipe landmark indices by body region
REGION_LANDMARKS = {
    BodyRegion.HEAD: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Face landmarks
    BodyRegion.TORSO: [11, 12, 23, 24],  # Shoulders + hips
    BodyRegion.LEFT_ARM: [11, 13, 15, 17, 19, 21],  # Shoulder to fingers
    BodyRegion.RIGHT_ARM: [12, 14, 16, 18, 20, 22],
    BodyRegion.LEFT_LEG: [23, 25, 27, 29, 31],  # Hip to toes
    BodyRegion.RIGHT_LEG: [24, 26, 28, 30, 32],
    BodyRegion.FULL_BODY: list(range(33)),  # All landmarks
}

# Importance weights for jitter calculation (movement sensitivity)
REGION_WEIGHTS = {
    BodyRegion.HEAD: 0.5,       # Head movement less important
    BodyRegion.TORSO: 1.0,      # Core stability most important
    BodyRegion.LEFT_ARM: 1.5,   # Arm movement very noticeable
    BodyRegion.RIGHT_ARM: 1.5,
    BodyRegion.LEFT_LEG: 1.2,   # Leg movement noticeable
    BodyRegion.RIGHT_LEG: 1.2,
}


@dataclass
class StillnessState:
    """Current stillness/jitter state output."""
    jitter_score: float         # Overall jitter 0-1 (0 = still, 1 = restless)
    raw_motion: float           # Raw motion magnitude before smoothing
    smoothed_motion: float      # Motion after temporal smoothing
    stillness_duration: float   # Seconds of sustained stillness
    motion_type: str            # "still", "fidgeting", "moving", "restless"
    regional_motion: Dict[str, float]  # Per-region motion scores
    confidence: float           # Detection confidence 0-1
    timestamp: float


class StillnessDetector:
    """
    Real-time body stillness and jitter detection.

    Combines multiple detection methods:
    1. Pose landmark tracking (primary) - Tracks 33 body points
    2. Optical flow (fallback) - Pixel-level motion when pose fails
    3. Frame differencing (simple backup) - Basic motion detection

    The system uses One-Euro filtering for smooth, jitter-free tracking,
    and temporal analysis to distinguish sustained stillness from momentary pauses.

    Usage:
        detector = StillnessDetector()

        # In frame loop:
        state = detector.update(frame, landmarks=pose_landmarks)
        print(f"Jitter: {state.jitter_score:.2f}, Type: {state.motion_type}")
    """

    def __init__(
        self,
        # Sensitivity settings
        motion_threshold_low: float = 0.02,   # Below this = "still"
        motion_threshold_high: float = 0.15,  # Above this = "restless"

        # Timing settings
        stillness_confirm_time: float = 0.5,  # Seconds to confirm stillness
        buffer_size: int = 30,                # Frame buffer (~1 second at 30fps)

        # Smoothing settings
        temporal_smoothing: float = 0.15,     # Exponential smoothing factor
        filter_preset: str = "stillness",     # One-Euro filter preset

        # Processing settings
        frame_scale: float = 0.5,             # Scale factor for optical flow
    ):
        """
        Initialize the stillness detector.

        Args:
            motion_threshold_low: Motion below this is "still"
            motion_threshold_high: Motion above this is "restless"
            stillness_confirm_time: Seconds to confirm sustained stillness
            buffer_size: Number of frames to buffer for analysis
            temporal_smoothing: Smoothing factor for motion signal
            filter_preset: One-Euro filter preset name
            frame_scale: Scale factor for optical flow processing
        """
        self.motion_threshold_low = motion_threshold_low
        self.motion_threshold_high = motion_threshold_high
        self.stillness_confirm_time = stillness_confirm_time
        self.buffer_size = buffer_size
        self.temporal_smoothing = temporal_smoothing
        self.frame_scale = frame_scale

        # Initialize One-Euro landmark smoother
        preset = get_preset(filter_preset)
        self.landmark_smoother = LandmarkSmoother(
            num_landmarks=33,
            **preset
        )

        # Previous frame data
        self.prev_landmarks: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None

        # Motion buffers
        self.motion_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)

        # Smoothed values
        self.smoothed_jitter = 0.0
        self.smoothed_regional = {region.value: 0.0 for region in BodyRegion}

        # Stillness tracking
        self.stillness_start_time: Optional[float] = None
        self.last_significant_motion_time = time.time()

        # Optical flow parameters (Farneback)
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }

        # Calibration
        self.calibration_frames = 30
        self.calibration_data: List[float] = []
        self.is_calibrated = False
        self.baseline_motion = 0.0

    def update(
        self,
        frame: np.ndarray,
        landmarks=None,  # MediaPipe pose landmarks
        depth_frame: Optional[np.ndarray] = None,
    ) -> StillnessState:
        """
        Process a frame and return current stillness state.

        Args:
            frame: RGB/BGR frame from camera
            landmarks: MediaPipe pose landmarks (optional, preferred)
            depth_frame: Depth frame for enhanced detection (optional)

        Returns:
            StillnessState with jitter score and metrics
        """
        current_time = time.time()
        confidence = 0.0
        raw_motion = 0.0
        regional_motion = {region.value: 0.0 for region in BodyRegion}

        # Method 1: Pose-based detection (preferred)
        if landmarks is not None:
            motion_data = self._detect_from_landmarks(landmarks, current_time)
            if motion_data is not None:
                raw_motion = motion_data['total_motion']
                regional_motion = motion_data['regional_motion']
                confidence = motion_data['confidence']

        # Method 2: Optical flow fallback
        if confidence < 0.5:
            flow_motion, flow_conf = self._detect_from_optical_flow(frame)
            if flow_conf > confidence:
                raw_motion = flow_motion
                confidence = flow_conf
                # Optical flow doesn't provide regional data
                regional_motion = {region.value: flow_motion for region in BodyRegion}

        # Calibration phase
        if not self.is_calibrated:
            return self._calibrate(raw_motion, regional_motion, confidence, current_time)

        # Subtract baseline noise
        adjusted_motion = max(0, raw_motion - self.baseline_motion)

        # Apply temporal smoothing
        alpha = self.temporal_smoothing
        self.smoothed_jitter = alpha * adjusted_motion + (1 - alpha) * self.smoothed_jitter

        # Update regional smoothing
        for region in BodyRegion:
            key = region.value
            self.smoothed_regional[key] = (
                alpha * regional_motion.get(key, 0) +
                (1 - alpha) * self.smoothed_regional[key]
            )

        # Store in buffer
        self.motion_buffer.append(self.smoothed_jitter)
        self.time_buffer.append(current_time)

        # Calculate jitter score (0-1)
        jitter_score = self._calculate_jitter_score()

        # Determine motion type
        motion_type = self._classify_motion(jitter_score)

        # Track stillness duration
        stillness_duration = self._track_stillness_duration(jitter_score, current_time)

        return StillnessState(
            jitter_score=jitter_score,
            raw_motion=raw_motion,
            smoothed_motion=self.smoothed_jitter,
            stillness_duration=stillness_duration,
            motion_type=motion_type,
            regional_motion=self.smoothed_regional.copy(),
            confidence=confidence,
            timestamp=current_time
        )

    def _detect_from_landmarks(
        self, landmarks, current_time: float
    ) -> Optional[Dict]:
        """
        Detect motion from MediaPipe pose landmarks.

        Uses landmark displacement between frames to calculate motion.
        Applies One-Euro filtering for smooth tracking.
        """
        # Extract landmark coordinates
        try:
            coords = np.array([
                [lm.x, lm.y, lm.z] for lm in landmarks.landmark
            ])
            visibilities = np.array([lm.visibility for lm in landmarks.landmark])
        except (AttributeError, IndexError):
            return None

        # Check minimum visibility
        avg_visibility = np.mean(visibilities)
        if avg_visibility < 0.3:
            return None

        # Apply One-Euro smoothing
        smoothed_coords = self.landmark_smoother.smooth(current_time, coords)

        # No previous landmarks = no motion calculation
        if self.prev_landmarks is None:
            self.prev_landmarks = smoothed_coords.copy()
            return {
                'total_motion': 0.0,
                'regional_motion': {r.value: 0.0 for r in BodyRegion},
                'confidence': avg_visibility
            }

        # Calculate displacement for each landmark
        displacement = np.linalg.norm(smoothed_coords - self.prev_landmarks, axis=1)

        # Calculate regional motion
        regional_motion = {}
        for region, indices in REGION_LANDMARKS.items():
            valid_indices = [i for i in indices if visibilities[i] > 0.5]
            if valid_indices:
                region_displacement = displacement[valid_indices]
                # Use weighted average (higher = more motion)
                regional_motion[region.value] = float(np.mean(region_displacement))
            else:
                regional_motion[region.value] = 0.0

        # Calculate weighted total motion
        total_motion = 0.0
        total_weight = 0.0
        for region in BodyRegion:
            if region != BodyRegion.FULL_BODY:
                weight = REGION_WEIGHTS.get(region, 1.0)
                total_motion += regional_motion.get(region.value, 0) * weight
                total_weight += weight

        if total_weight > 0:
            total_motion /= total_weight

        # Update previous landmarks
        self.prev_landmarks = smoothed_coords.copy()

        return {
            'total_motion': total_motion,
            'regional_motion': regional_motion,
            'confidence': avg_visibility
        }

    def _detect_from_optical_flow(
        self, frame: np.ndarray
    ) -> Tuple[float, float]:
        """
        Detect motion using dense optical flow.

        Fallback method when pose detection fails or has low confidence.
        """
        # Preprocess frame
        h, w = frame.shape[:2]
        new_w = int(w * self.frame_scale)
        new_h = int(h * self.frame_scale)

        resized = cv2.resize(frame, (new_w, new_h))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = blurred
            return 0.0, 0.3

        # Calculate optical flow
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, blurred, None, **self.flow_params
            )

            # Calculate flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

            # Use mean magnitude as motion measure
            motion = float(np.mean(magnitude))

            # Normalize (empirically tuned)
            normalized_motion = np.clip(motion / 3.0, 0, 1)

            # Update previous frame
            self.prev_gray = blurred

            return normalized_motion, 0.6  # Moderate confidence for optical flow

        except Exception:
            self.prev_gray = blurred
            return 0.0, 0.0

    def _calibrate(
        self,
        raw_motion: float,
        regional_motion: Dict,
        confidence: float,
        current_time: float
    ) -> StillnessState:
        """Collect calibration data to establish baseline noise level."""
        if confidence > 0.3:
            self.calibration_data.append(raw_motion)

        if len(self.calibration_data) >= self.calibration_frames:
            # Use 75th percentile as baseline (allows for some movement during calibration)
            self.baseline_motion = float(np.percentile(self.calibration_data, 75))
            self.is_calibrated = True
            print(f"[StillnessDetector] Calibration complete. Baseline: {self.baseline_motion:.4f}")

        progress = len(self.calibration_data) / self.calibration_frames

        return StillnessState(
            jitter_score=0.0,
            raw_motion=raw_motion,
            smoothed_motion=0.0,
            stillness_duration=0.0,
            motion_type="calibrating",
            regional_motion=regional_motion,
            confidence=progress,  # Use progress as confidence during calibration
            timestamp=current_time
        )

    def _calculate_jitter_score(self) -> float:
        """
        Calculate normalized jitter score from smoothed motion.

        Returns score from 0 (still) to 1 (restless).
        Uses a sigmoid-like curve for natural feeling.
        """
        motion = self.smoothed_jitter

        # Map to 0-1 range using thresholds
        if motion <= self.motion_threshold_low:
            # Below low threshold = nearly still
            score = motion / self.motion_threshold_low * 0.2
        elif motion >= self.motion_threshold_high:
            # Above high threshold = fully restless
            score = 0.8 + 0.2 * min(1, (motion - self.motion_threshold_high) /
                                     self.motion_threshold_high)
        else:
            # Middle range - linear interpolation
            range_size = self.motion_threshold_high - self.motion_threshold_low
            position = (motion - self.motion_threshold_low) / range_size
            score = 0.2 + position * 0.6

        return float(np.clip(score, 0, 1))

    def _classify_motion(self, jitter_score: float) -> str:
        """Classify motion type from jitter score."""
        if jitter_score < 0.1:
            return "still"
        elif jitter_score < 0.3:
            return "fidgeting"
        elif jitter_score < 0.6:
            return "moving"
        else:
            return "restless"

    def _track_stillness_duration(
        self, jitter_score: float, current_time: float
    ) -> float:
        """Track how long the user has been still."""
        is_still = jitter_score < 0.15

        if is_still:
            if self.stillness_start_time is None:
                self.stillness_start_time = current_time
            duration = current_time - self.stillness_start_time
        else:
            self.stillness_start_time = None
            self.last_significant_motion_time = current_time
            duration = 0.0

        return duration

    def get_stillness_quality(self) -> str:
        """
        Get quality rating of stillness based on sustained duration.

        Returns quality level for potential rewards/feedback.
        """
        if not self.motion_buffer:
            return "unknown"

        recent_avg = np.mean(list(self.motion_buffer)[-15:]) if len(self.motion_buffer) >= 15 else 0

        if self.stillness_start_time is None:
            return "moving"

        duration = time.time() - self.stillness_start_time

        if duration < 5:
            return "settling"
        elif duration < 15:
            return "focused"
        elif duration < 30:
            return "deep_focus"
        else:
            return "transcendent"  # 30+ seconds of stillness

    def reset(self):
        """Reset detector state."""
        self.prev_landmarks = None
        self.prev_gray = None
        self.motion_buffer.clear()
        self.time_buffer.clear()
        self.smoothed_jitter = 0.0
        self.smoothed_regional = {region.value: 0.0 for region in BodyRegion}
        self.stillness_start_time = None
        self.landmark_smoother.reset()
        self.calibration_data = []
        self.is_calibrated = False


# Convenience function for quick setup
def create_stillness_detector(sensitivity: str = "normal") -> StillnessDetector:
    """
    Create a stillness detector with preset sensitivity.

    Args:
        sensitivity: "low", "normal", or "high"

    Returns:
        Configured StillnessDetector
    """
    presets = {
        "low": {
            "motion_threshold_low": 0.03,
            "motion_threshold_high": 0.20,
            "temporal_smoothing": 0.1,
        },
        "normal": {
            "motion_threshold_low": 0.02,
            "motion_threshold_high": 0.15,
            "temporal_smoothing": 0.15,
        },
        "high": {
            "motion_threshold_low": 0.01,
            "motion_threshold_high": 0.10,
            "temporal_smoothing": 0.2,
        },
    }

    params = presets.get(sensitivity, presets["normal"])
    return StillnessDetector(**params)
