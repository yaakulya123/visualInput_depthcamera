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
        detector = BreathingDetector(mode=DetectionMode.DEPTH)

        # In frame loop:
        state = detector.update(frame, depth_frame=depth_data)
        print(f"Breathing: {state.signal:.2f}, Phase: {state.phase}")
    """

    def __init__(
        self,
        mode: DetectionMode = DetectionMode.POSE,
        buffer_size: int = 100,         # Frames to buffer (~3.3s at 30fps)
        smoothing_alpha: float = 0.2,   # Exponential smoothing factor
        min_amplitude_mm: float = 3.0,  # Minimum detectable breath (mm)
        depth_scale: float = 0.001,     # RealSense depth scale (meters per unit)
    ):
        self.mode = mode
        self.buffer_size = buffer_size
        self.smoothing_alpha = smoothing_alpha
        self.min_amplitude_mm = min_amplitude_mm
        self.depth_scale = depth_scale

        # Signal buffer for breathing wave analysis
        self.signal_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)

        # Smoothed values
        self.smoothed_value = None
        self.baseline = None

        # Phase detection
        self.last_phase = "hold"
        self.last_peak_time = time.time()
        self.breath_intervals = deque(maxlen=10)

        # Calibration
        self.calibration_frames = 60
        self.calibration_values = []
        self.is_calibrated = False

        # ROI for depth mode
        self.chest_roi = None

        # Adaptive parameters based on distance
        self.current_distance_mm = 0.0
        self.adaptive_threshold = 0.5
        self.adaptive_smoothing = smoothing_alpha

    def update(
        self,
        frame: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        landmarks=None,
    ) -> BreathingState:
        """Process a frame and return current breathing state."""
        current_time = time.time()

        # Get raw measurement based on mode
        if self.mode == DetectionMode.DEPTH and depth_frame is not None:
            raw_value, confidence = self._measure_depth(depth_frame, frame)
        elif self.mode == DetectionMode.POSE and landmarks is not None:
            raw_value, confidence = self._measure_pose(landmarks, frame)
        else:
            return BreathingState(
                signal=0.0, raw_value=0.0, phase="hold",
                breath_rate=0.0, amplitude=0.0, confidence=0.0,
                timestamp=current_time
            )

        # Calibration phase
        if not self.is_calibrated:
            return self._calibrate(raw_value, confidence, current_time)

        # Apply exponential smoothing
        alpha = self.adaptive_smoothing
        if self.smoothed_value is None:
            self.smoothed_value = raw_value
        else:
            self.smoothed_value = alpha * raw_value + (1 - alpha) * self.smoothed_value

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
        """Measure chest height from depth frame."""
        h, w = depth_frame.shape[:2]

        # If no ROI set, use center region
        if self.chest_roi is None:
            roi_w = int(w * 0.3)
            roi_h = int(h * 0.2)
            roi_x = (w - roi_w) // 2
            roi_y = int(h * 0.3)
            self.chest_roi = (roi_x, roi_y, roi_w, roi_h)

        rx, ry, rw, rh = self.chest_roi

        # Clamp ROI to frame bounds
        rx = max(0, min(rx, w - 1))
        ry = max(0, min(ry, h - 1))
        rw = min(rw, w - rx)
        rh = min(rh, h - ry)

        if rw <= 0 or rh <= 0:
            return 0.0, 0.0

        roi = depth_frame[ry:ry+rh, rx:rx+rw]

        # Filter out invalid depth values
        valid_mask = roi > 0
        if not np.any(valid_mask):
            return 0.0, 0.0

        valid_depths = roi[valid_mask]

        # Use median of closest 30% points (chest surface)
        sorted_depths = np.sort(valid_depths)
        top_percent = sorted_depths[:max(1, len(sorted_depths) * 30 // 100)]

        if len(top_percent) == 0:
            return 0.0, 0.0

        chest_depth_raw = np.median(top_percent)
        chest_depth_mm = float(chest_depth_raw) * self.depth_scale * 1000.0

        # Update distance and adaptive parameters
        self.current_distance_mm = chest_depth_mm
        self._update_adaptive_parameters()

        confidence = min(1.0, len(top_percent) / 100)
        return chest_depth_mm, confidence

    def _measure_pose(
        self, landmarks, rgb_frame: np.ndarray
    ) -> Tuple[float, float]:
        """Measure chest position from pose landmarks."""
        if landmarks is None:
            return 0.0, 0.0

        try:
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]
        except (IndexError, AttributeError):
            return 0.0, 0.0

        confidence = (left_shoulder.visibility + right_shoulder.visibility) / 2
        if confidence < 0.5:
            return 0.0, 0.0

        avg_y = (left_shoulder.y + right_shoulder.y) / 2
        h, w = rgb_frame.shape[:2]
        y_pixels = avg_y * h

        return y_pixels, confidence

    def _update_adaptive_parameters(self):
        """Adjust parameters based on distance."""
        d = self.current_distance_mm
        if d <= 0:
            return

        if d < 500:
            self.adaptive_threshold = 0.3
            self.adaptive_smoothing = 0.25
        elif d < 1000:
            self.adaptive_threshold = 0.5
            self.adaptive_smoothing = 0.2
        elif d < 1500:
            self.adaptive_threshold = 0.8
            self.adaptive_smoothing = 0.15
        elif d < 2500:
            self.adaptive_threshold = 1.2
            self.adaptive_smoothing = 0.1
        else:
            self.adaptive_threshold = 2.0
            self.adaptive_smoothing = 0.08

    def _calibrate(
        self, raw_value: float, confidence: float, current_time: float
    ) -> BreathingState:
        """Collect calibration data to establish baseline."""
        if confidence > 0.5:
            self.calibration_values.append(raw_value)

        if len(self.calibration_values) >= self.calibration_frames:
            self.baseline = np.median(self.calibration_values)
            self.smoothed_value = self.baseline
            self.is_calibrated = True
            print(f"[BreathDetector] Calibration complete. Baseline: {self.baseline:.2f}mm")

        progress = len(self.calibration_values) / self.calibration_frames
        return BreathingState(
            signal=0.0, raw_value=raw_value, phase="calibrating",
            breath_rate=0.0, amplitude=progress, confidence=confidence,
            timestamp=current_time
        )

    def _calculate_normalized_signal(self) -> float:
        """Normalize signal to [-1, 1] based on recent range."""
        if len(self.signal_buffer) < 10:
            return 0.0

        # Use recent window for dynamic range
        recent_window = min(90, len(self.signal_buffer))  # ~3 seconds
        recent = list(self.signal_buffer)[-recent_window:]

        min_val = np.min(recent)
        max_val = np.max(recent)
        range_val = max_val - min_val

        # Need minimum range to normalize
        min_range = 1.0 if self.mode == DetectionMode.DEPTH else 2.0
        if range_val < min_range:
            # Use baseline-based normalization
            if self.baseline is not None:
                deviation = self.baseline - self.smoothed_value
                scale = 10.0 if self.mode == DetectionMode.DEPTH else 30.0
                return float(np.clip(deviation / scale, -1, 1))
            return 0.0

        # Normalize current value
        current = self.smoothed_value
        normalized = 2 * (current - min_val) / range_val - 1

        # Invert for depth mode (lower depth = inhale = positive)
        if self.mode == DetectionMode.DEPTH:
            normalized = -normalized

        return float(np.clip(normalized, -1, 1))

    def _detect_phase(self, signal: float) -> str:
        """Detect breathing phase from signal."""
        if len(self.signal_buffer) < 5:
            return "hold"

        # Get recent trend
        recent = list(self.signal_buffer)[-8:]
        if len(recent) < 4:
            return "hold"

        # Calculate derivative (trend)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        derivative = first_half - second_half  # Inverted for depth

        # Threshold based on distance
        threshold = self.adaptive_threshold

        if derivative > threshold:
            new_phase = "inhale"
        elif derivative < -threshold:
            new_phase = "exhale"
        else:
            new_phase = "hold"

        # Track breath cycles for BPM
        if self.last_phase == "inhale" and new_phase == "exhale":
            current_time = time.time()
            interval = current_time - self.last_peak_time
            if 0.5 < interval < 10:
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
        if self.mode == DetectionMode.DEPTH:
            return float(np.clip(amplitude / 15.0, 0, 1))
        return float(np.clip(amplitude / 50.0, 0, 1))

    def reset(self):
        """Reset detector state."""
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.smoothed_value = None
        self.baseline = None
        self.is_calibrated = False
        self.calibration_values = []
        self.breath_intervals.clear()
        self.chest_roi = None

    def set_chest_roi(self, x: int, y: int, w: int, h: int):
        """Manually set chest region of interest."""
        self.chest_roi = (x, y, w, h)
