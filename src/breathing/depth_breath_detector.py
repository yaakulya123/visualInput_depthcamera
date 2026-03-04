#!/usr/bin/env python3
"""
Depth-Based Breathing Detector for Liquid Stillness

Uses ROI mean-depth method: track the average depth of the chest region
within each person's bounding box. Chest rises (depth decreases toward camera)
during inhale, falls (depth increases away from camera) during exhale.

This works because from a top-down view, the chest is the highest point of
a supine person. Breathing causes 5-15mm displacement at 2m range, well
within the RealSense D435's precision.

Algorithm:
  1. Extract chest ROI from depth image (middle portion of person's bbox)
  2. Compute median depth of valid pixels in that ROI
  3. Maintain rolling buffer of depth readings
  4. Bandpass filter (0.1-0.5 Hz) to isolate breathing frequency
  5. Detect peaks/troughs for inhale/exhale phase
  6. Estimate BPM from peak intervals

Reference: "Breathing In-Depth" (Frontiers in Computer Science, 2021)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import time


@dataclass
class DepthBreathingState:
    """Per-person breathing state from depth analysis."""
    signal: float = 0.0           # Normalized breathing signal [-1, 1]
    raw_depth_mm: float = 0.0     # Raw chest depth in mm
    filtered_depth_mm: float = 0.0  # Bandpass-filtered depth
    phase: str = "unknown"        # "inhale", "exhale", "hold", "unknown"
    bpm: float = 0.0              # Estimated breaths per minute
    amplitude_mm: float = 0.0     # Peak-to-trough amplitude in mm
    confidence: float = 0.0       # Detection confidence [0, 1]
    roi_valid_ratio: float = 0.0  # Fraction of valid pixels in chest ROI
    timestamp: float = 0.0


class DepthBreathingDetector:
    """
    Per-person depth-based breathing detector using ROI mean-depth method.

    Usage:
        detector = DepthBreathingDetector()

        # Each frame:
        state = detector.update(depth_image, bbox)
        print(f"Breathing: {state.signal:.2f}, Phase: {state.phase}, BPM: {state.bpm:.1f}")
    """

    def __init__(
        self,
        buffer_size: int = 150,       # ~5 seconds at 30fps
        sample_rate: float = 30.0,    # Expected frame rate
        chest_roi_ratio: float = 0.35, # Middle 35% of bbox height = chest
        chest_roi_top_offset: float = 0.25,  # Start 25% from top of bbox
        min_valid_ratio: float = 0.15, # Need at least 15% valid depth pixels
        smoothing_alpha: float = 0.4,  # EMA smoothing for raw depth
    ):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.chest_roi_ratio = chest_roi_ratio
        self.chest_roi_top_offset = chest_roi_top_offset
        self.min_valid_ratio = min_valid_ratio
        self.smoothing_alpha = smoothing_alpha

        # Rolling buffers
        self._depth_buffer = deque(maxlen=buffer_size)
        self._time_buffer = deque(maxlen=buffer_size)
        self._filtered_buffer = deque(maxlen=buffer_size)

        # Smoothed raw depth (EMA)
        self._smoothed_depth = None

        # Calibration
        self._calibrating = True
        self._calibration_frames = 0
        self._calibration_target = 60  # 2 seconds
        self._baseline_depth = 0.0
        self._depth_range = 1.0  # Will be estimated

        # Phase detection state
        self._last_phase = "unknown"
        self._phase_samples = 0

        # BPM estimation
        self._peak_times = deque(maxlen=20)
        self._trough_times = deque(maxlen=20)
        self._last_was_rising = False
        self._bpm = 0.0

        # Amplitude tracking
        self._recent_max = -np.inf
        self._recent_min = np.inf
        self._amplitude_mm = 0.0

    def update(self, depth_image: np.ndarray, bbox: Tuple[int, int, int, int]) -> DepthBreathingState:
        """
        Process one frame of depth data for this person.

        Args:
            depth_image: Full depth frame as uint16 numpy array (values in mm)
            bbox: Person bounding box (x1, y1, x2, y2) in pixels

        Returns:
            DepthBreathingState with current breathing metrics
        """
        now = time.time()
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h_img, w_img = depth_image.shape[:2]

        # Clamp bbox to image bounds
        x1 = max(0, min(x1, w_img - 1))
        x2 = max(x1 + 1, min(x2, w_img))
        y1 = max(0, min(y1, h_img - 1))
        y2 = max(y1 + 1, min(y2, h_img))

        bbox_h = y2 - y1
        bbox_w = x2 - x1

        # Extract chest ROI (middle portion of bounding box)
        roi_top = y1 + int(bbox_h * self.chest_roi_top_offset)
        roi_bottom = roi_top + int(bbox_h * self.chest_roi_ratio)
        roi_bottom = min(roi_bottom, y2)
        # Narrow horizontally to center 60% to avoid arms
        roi_left = x1 + int(bbox_w * 0.2)
        roi_right = x2 - int(bbox_w * 0.2)

        if roi_bottom <= roi_top or roi_right <= roi_left:
            return self._make_state(now, confidence=0.0)

        chest_roi = depth_image[roi_top:roi_bottom, roi_left:roi_right].astype(np.float32)

        # Filter valid depth pixels (non-zero, reasonable range 200mm-5000mm)
        valid_mask = (chest_roi > 200) & (chest_roi < 5000)
        total_pixels = chest_roi.size
        valid_count = np.count_nonzero(valid_mask)
        valid_ratio = valid_count / max(total_pixels, 1)

        if valid_ratio < self.min_valid_ratio:
            return self._make_state(now, confidence=0.0, roi_valid_ratio=valid_ratio)

        # Compute median depth of valid pixels (robust to outliers)
        chest_depth_mm = float(np.median(chest_roi[valid_mask]))

        # EMA smoothing on raw depth
        if self._smoothed_depth is None:
            self._smoothed_depth = chest_depth_mm
        else:
            self._smoothed_depth = (self.smoothing_alpha * chest_depth_mm +
                                     (1 - self.smoothing_alpha) * self._smoothed_depth)

        # Store in buffers
        self._depth_buffer.append(self._smoothed_depth)
        self._time_buffer.append(now)

        # Calibration phase: collect baseline
        if self._calibrating:
            self._calibration_frames += 1
            if self._calibration_frames >= self._calibration_target:
                self._baseline_depth = np.mean(list(self._depth_buffer))
                self._calibrating = False
            return self._make_state(now, raw_depth_mm=chest_depth_mm,
                                     confidence=0.1, roi_valid_ratio=valid_ratio)

        # Need enough samples for filtering
        if len(self._depth_buffer) < 30:
            return self._make_state(now, raw_depth_mm=chest_depth_mm,
                                     confidence=0.2, roi_valid_ratio=valid_ratio)

        # Simple bandpass filter using moving averages
        # Subtract slow-moving baseline (high-pass) to remove drift
        # Then smooth (low-pass) to remove noise
        data = np.array(list(self._depth_buffer))

        # High-pass: subtract long-term trend (3-second window)
        long_window = min(int(self.sample_rate * 3), len(data))
        if long_window > 1:
            baseline = np.convolve(data, np.ones(long_window) / long_window, mode='same')
            detrended = data - baseline
        else:
            detrended = data - np.mean(data)

        # Low-pass: smooth with short window (~0.3s)
        short_window = max(3, int(self.sample_rate * 0.3))
        if short_window > 1:
            filtered = np.convolve(detrended, np.ones(short_window) / short_window, mode='same')
        else:
            filtered = detrended

        current_filtered = filtered[-1]
        self._filtered_buffer.append(current_filtered)

        # Normalize signal to [-1, 1]
        # Note: DECREASING depth = inhale (chest rises toward camera)
        # So we invert: negative filtered value → positive signal (inhale)
        if len(self._filtered_buffer) > 30:
            filt_arr = np.array(list(self._filtered_buffer))
            filt_max = np.percentile(np.abs(filt_arr), 95)
            if filt_max > 0.5:  # At least 0.5mm variation to be meaningful
                signal = np.clip(-current_filtered / filt_max, -1.0, 1.0)
            else:
                signal = 0.0
        else:
            signal = 0.0

        # Phase detection
        phase = self._detect_phase(signal)

        # BPM estimation
        self._update_bpm(signal, now)

        # Amplitude tracking
        self._update_amplitude(filtered)

        # Confidence based on signal quality
        confidence = self._compute_confidence(valid_ratio, data)

        return self._make_state(
            now,
            signal=signal,
            raw_depth_mm=chest_depth_mm,
            filtered_depth_mm=current_filtered,
            phase=phase,
            bpm=self._bpm,
            amplitude_mm=self._amplitude_mm,
            confidence=confidence,
            roi_valid_ratio=valid_ratio,
        )

    def _detect_phase(self, signal: float) -> str:
        """Detect breathing phase from signal value and derivative."""
        threshold = 0.1

        if signal > threshold:
            phase = "inhale"
        elif signal < -threshold:
            phase = "exhale"
        else:
            phase = "hold"

        self._last_phase = phase
        return phase

    def _update_bpm(self, signal: float, now: float):
        """Estimate BPM from zero-crossing intervals."""
        if len(self._filtered_buffer) < 2:
            return

        # Detect zero crossings (rising edge = start of inhale)
        prev = list(self._filtered_buffer)[-2] if len(self._filtered_buffer) >= 2 else 0
        curr = -signal  # Inverted because we inverted for display
        is_rising = curr > 0 and prev <= 0

        if is_rising and not self._last_was_rising:
            self._peak_times.append(now)

            # Compute BPM from intervals between peaks
            if len(self._peak_times) >= 3:
                intervals = []
                times = list(self._peak_times)
                for i in range(1, len(times)):
                    dt = times[i] - times[i - 1]
                    if 1.5 < dt < 15.0:  # Valid breath interval (4-40 BPM)
                        intervals.append(dt)
                if intervals:
                    avg_interval = np.mean(intervals[-5:])  # Last 5 breaths
                    self._bpm = 60.0 / avg_interval

        self._last_was_rising = is_rising

    def _update_amplitude(self, filtered: np.ndarray):
        """Track breathing amplitude from filtered signal."""
        if len(filtered) < 30:
            return
        recent = filtered[-int(self.sample_rate * 3):]  # Last 3 seconds
        self._recent_max = np.max(recent)
        self._recent_min = np.min(recent)
        self._amplitude_mm = abs(self._recent_max - self._recent_min)

    def _compute_confidence(self, valid_ratio: float, data: np.ndarray) -> float:
        """Compute detection confidence based on signal quality."""
        conf = 0.0

        # Valid pixel ratio contributes
        conf += min(valid_ratio / 0.5, 1.0) * 0.3

        # Buffer fullness
        fullness = len(self._depth_buffer) / self.buffer_size
        conf += fullness * 0.2

        # Amplitude: need at least 2mm to be meaningful breathing
        if self._amplitude_mm > 2.0:
            conf += min(self._amplitude_mm / 10.0, 1.0) * 0.3

        # BPM in reasonable range (8-25 BPM for resting/meditation)
        if 6.0 < self._bpm < 30.0:
            conf += 0.2

        return min(conf, 1.0)

    def _make_state(self, now, signal=0.0, raw_depth_mm=0.0, filtered_depth_mm=0.0,
                    phase="unknown", bpm=0.0, amplitude_mm=0.0,
                    confidence=0.0, roi_valid_ratio=0.0):
        return DepthBreathingState(
            signal=signal,
            raw_depth_mm=raw_depth_mm,
            filtered_depth_mm=filtered_depth_mm,
            phase=phase,
            bpm=bpm,
            amplitude_mm=amplitude_mm,
            confidence=confidence,
            roi_valid_ratio=roi_valid_ratio,
            timestamp=now,
        )

    def reset(self):
        """Reset detector state (e.g., when person reappears)."""
        self._depth_buffer.clear()
        self._time_buffer.clear()
        self._filtered_buffer.clear()
        self._smoothed_depth = None
        self._calibrating = True
        self._calibration_frames = 0
        self._baseline_depth = 0.0
        self._last_phase = "unknown"
        self._peak_times.clear()
        self._trough_times.clear()
        self._bpm = 0.0
        self._amplitude_mm = 0.0
        self._recent_max = -np.inf
        self._recent_min = np.inf
