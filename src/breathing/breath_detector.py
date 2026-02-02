#!/usr/bin/env python3
"""
Breathing Detector - Core breathing detection for Liquid Stillness

Supports two modes:
1. DEPTH mode: Uses RealSense depth stream for precise Z-axis tracking (5-10mm)
2. POSE mode: Uses MediaPipe pose landmarks as fallback (Y-axis approximation)

The detector outputs a normalized breathing signal [-1, 1]:
  -1 = Full exhale (chest at lowest)
   0 = Neutral
  +1 = Full inhale (chest at highest)
"""

import numpy as np
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
import time


class DetectionMode(Enum):
    DEPTH = "depth"       # RealSense depth stream
    POSE = "pose"         # MediaPipe pose landmarks
    UNKNOWN = "unknown"


@dataclass
class BreathingState:
    """Current breathing state output."""
    signal: float           # Normalized breathing signal [-1, 1]
    raw_value: float        # Raw measurement (mm for depth, pixels for pose)
    phase: str              # "inhale", "exhale", or "hold"
    breath_rate: float      # Breaths per minute (BPM)
    amplitude: float        # Breathing depth (normalized)
    confidence: float       # Detection confidence [0, 1]
    timestamp: float        # Unix timestamp


class BreathingDetector:
    """
    Real-time breathing detection from depth or pose data.

    Usage:
        detector = BreathingDetector(mode=DetectionMode.POSE)

        # In frame loop:
        state = detector.update(frame, landmarks=pose_landmarks)
        print(f"Breathing: {state.signal:.2f}, Phase: {state.phase}")
    """

    def __init__(
        self,
        mode: DetectionMode = DetectionMode.POSE,
        buffer_size: int = 100,         # Frames to buffer (~3.3s at 30fps)
        smoothing_alpha: float = 0.3,   # Exponential smoothing factor
        min_amplitude_mm: float = 3.0,  # Minimum detectable breath (mm)
    ):
        self.mode = mode
        self.buffer_size = buffer_size
        self.smoothing_alpha = smoothing_alpha
        self.min_amplitude_mm = min_amplitude_mm

        # Signal buffer for breathing wave analysis
        self.signal_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)

        # Smoothed values
        self.smoothed_value = None
        self.baseline = None  # Running baseline (neutral chest position)

        # Phase detection
        self.last_phase = "hold"
        self.last_peak_time = time.time()
        self.breath_intervals = deque(maxlen=10)  # For BPM calculation

        # Calibration
        self.calibration_frames = 60  # Frames to calibrate baseline
        self.calibration_values = []
        self.is_calibrated = False

        # ROI for depth mode (will be set during detection)
        self.chest_roi = None  # (x, y, w, h)

    def update(
        self,
        frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        landmarks=None,  # MediaPipe pose landmarks
    ) -> BreathingState:
        """
        Process a frame and return current breathing state.

        Args:
            frame: RGB/BGR frame from camera
            depth_frame: 16-bit depth frame (DEPTH mode only)
            landmarks: MediaPipe pose landmarks (POSE mode only)

        Returns:
            BreathingState with current breathing metrics
        """
        current_time = time.time()

        # Get raw measurement based on mode
        if self.mode == DetectionMode.DEPTH and depth_frame is not None:
            raw_value, confidence = self._measure_depth(depth_frame, frame)
        elif self.mode == DetectionMode.POSE and landmarks is not None:
            raw_value, confidence = self._measure_pose(landmarks, frame)
        else:
            # No valid input
            return BreathingState(
                signal=0.0,
                raw_value=0.0,
                phase="hold",
                breath_rate=0.0,
                amplitude=0.0,
                confidence=0.0,
                timestamp=current_time
            )

        # Calibration phase
        if not self.is_calibrated:
            return self._calibrate(raw_value, confidence, current_time)

        # Apply exponential smoothing
        if self.smoothed_value is None:
            self.smoothed_value = raw_value
        else:
            self.smoothed_value = (
                self.smoothing_alpha * raw_value +
                (1 - self.smoothing_alpha) * self.smoothed_value
            )

        # Store in buffer
        self.signal_buffer.append(self.smoothed_value)
        self.time_buffer.append(current_time)

        # Calculate breathing metrics
        signal = self._calculate_normalized_signal()
        phase = self._detect_phase(signal)
        breath_rate = self._calculate_breath_rate()
        amplitude = self._calculate_amplitude()

        return BreathingState(
            signal=signal,
            raw_value=raw_value,
            phase=phase,
            breath_rate=breath_rate,
            amplitude=amplitude,
            confidence=confidence,
            timestamp=current_time
        )

    def _measure_depth(
        self, depth_frame: np.ndarray, rgb_frame: np.ndarray
    ) -> Tuple[float, float]:
        """
        Measure chest height from depth frame.

        Returns: (height_mm, confidence)
        """
        h, w = depth_frame.shape[:2]

        # If no ROI set, use center region
        if self.chest_roi is None:
            # Default: center 30% of frame (assuming person is centered)
            roi_w = int(w * 0.3)
            roi_h = int(h * 0.2)
            roi_x = (w - roi_w) // 2
            roi_y = int(h * 0.3)  # Upper-middle (chest area from top-down)
            self.chest_roi = (roi_x, roi_y, roi_w, roi_h)

        rx, ry, rw, rh = self.chest_roi
        roi = depth_frame[ry:ry+rh, rx:rx+rw]

        # Filter out invalid depth values (0 = no data)
        valid_mask = roi > 0
        if not np.any(valid_mask):
            return 0.0, 0.0

        valid_depths = roi[valid_mask]

        # Use median of top 20% closest points (chest surface)
        sorted_depths = np.sort(valid_depths)
        top_20_percent = sorted_depths[:len(sorted_depths) // 5]

        if len(top_20_percent) == 0:
            return 0.0, 0.0

        chest_depth = np.median(top_20_percent)
        confidence = min(1.0, len(top_20_percent) / 100)

        return float(chest_depth), confidence

    def _measure_pose(
        self, landmarks, rgb_frame: np.ndarray
    ) -> Tuple[float, float]:
        """
        Measure chest position from MediaPipe pose landmarks.
        Uses shoulder landmarks (11, 12) to approximate chest movement.

        Returns: (y_position, confidence)
        """
        if landmarks is None:
            return 0.0, 0.0

        # Get shoulder landmarks
        # MediaPipe landmark indices: 11 = left shoulder, 12 = right shoulder
        try:
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
        except (IndexError, AttributeError):
            return 0.0, 0.0

        # Average visibility as confidence
        confidence = (left_shoulder.visibility + right_shoulder.visibility) / 2

        if confidence < 0.5:
            return 0.0, 0.0

        # Average Y position of shoulders (inverted: lower Y = higher in frame)
        # For top-down view, we use Y as proxy for chest height
        avg_y = (left_shoulder.y + right_shoulder.y) / 2

        # Convert to pixel coordinates for consistency
        h, w = rgb_frame.shape[:2]
        y_pixels = avg_y * h

        return y_pixels, confidence

    def _calibrate(
        self, raw_value: float, confidence: float, current_time: float
    ) -> BreathingState:
        """Collect calibration data to establish baseline."""
        if confidence > 0.5:
            self.calibration_values.append(raw_value)

        if len(self.calibration_values) >= self.calibration_frames:
            # Calculate baseline as median
            self.baseline = np.median(self.calibration_values)
            self.smoothed_value = self.baseline
            self.is_calibrated = True
            print(f"[BreathDetector] Calibration complete. Baseline: {self.baseline:.2f}")

        progress = len(self.calibration_values) / self.calibration_frames
        return BreathingState(
            signal=0.0,
            raw_value=raw_value,
            phase="calibrating",
            breath_rate=0.0,
            amplitude=progress,  # Use amplitude to show calibration progress
            confidence=confidence,
            timestamp=current_time
        )

    def _calculate_normalized_signal(self) -> float:
        """Normalize signal to [-1, 1] based on recent range."""
        if len(self.signal_buffer) < 10:
            return 0.0

        # Get recent range
        recent = list(self.signal_buffer)
        min_val = np.min(recent)
        max_val = np.max(recent)
        range_val = max_val - min_val

        if range_val < 1e-6:  # Avoid division by zero
            return 0.0

        # Normalize: -1 (exhale/min) to +1 (inhale/max)
        current = self.smoothed_value
        normalized = 2 * (current - min_val) / range_val - 1

        # For POSE mode, invert (lower Y = inhale = higher chest)
        if self.mode == DetectionMode.POSE:
            normalized = -normalized

        return float(np.clip(normalized, -1, 1))

    def _detect_phase(self, signal: float) -> str:
        """Detect breathing phase from signal derivative."""
        if len(self.signal_buffer) < 3:
            return "hold"

        # Calculate derivative (signal change)
        recent = list(self.signal_buffer)[-5:]
        if len(recent) < 2:
            return "hold"

        derivative = recent[-1] - recent[0]

        # Thresholds for phase detection
        threshold = 0.5  # Minimum change to register as inhale/exhale

        if derivative > threshold:
            new_phase = "inhale" if self.mode == DetectionMode.DEPTH else "exhale"
        elif derivative < -threshold:
            new_phase = "exhale" if self.mode == DetectionMode.DEPTH else "inhale"
        else:
            new_phase = "hold"

        # Track breath cycles for BPM
        if self.last_phase == "inhale" and new_phase == "exhale":
            # Completed one breath cycle
            current_time = time.time()
            interval = current_time - self.last_peak_time
            if 0.5 < interval < 10:  # Valid breath interval (6-120 BPM)
                self.breath_intervals.append(interval)
            self.last_peak_time = current_time

        self.last_phase = new_phase
        return new_phase

    def _calculate_breath_rate(self) -> float:
        """Calculate breaths per minute from intervals."""
        if len(self.breath_intervals) < 2:
            return 0.0

        avg_interval = np.mean(self.breath_intervals)
        if avg_interval > 0:
            return 60.0 / avg_interval
        return 0.0

    def _calculate_amplitude(self) -> float:
        """Calculate normalized breathing amplitude."""
        if len(self.signal_buffer) < 30:
            return 0.0

        recent = list(self.signal_buffer)
        amplitude = np.max(recent) - np.min(recent)

        # Normalize based on expected range
        if self.mode == DetectionMode.DEPTH:
            # Expect 5-15mm for normal breathing
            normalized = np.clip(amplitude / 15.0, 0, 1)
        else:
            # Pose mode: pixels, depends on resolution
            normalized = np.clip(amplitude / 50.0, 0, 1)

        return float(normalized)

    def reset(self):
        """Reset detector state (for recalibration)."""
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.smoothed_value = None
        self.baseline = None
        self.is_calibrated = False
        self.calibration_values = []
        self.breath_intervals.clear()
        self.chest_roi = None

    def set_chest_roi(self, x: int, y: int, w: int, h: int):
        """Manually set chest region of interest (for depth mode)."""
        self.chest_roi = (x, y, w, h)
