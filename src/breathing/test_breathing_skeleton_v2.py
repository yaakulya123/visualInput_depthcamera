#!/usr/bin/env python3
"""
Breathing Detection with 3D Skeleton Tracking (V2)

Improved version that uses:
- MediaPipe Pose on RGB (accurate 2D landmarks)
- RealSense aligned depth (true 3D coordinates)
- Chest depth tracking for breathing

This is more accurate than V1 which used YOLO on depth colormaps.

Run with: sudo ./venv/bin/python src/breathing/test_breathing_skeleton_v2.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from src.tracking.realsense_skeleton import (
    RealSenseSkeletonTracker,
    Skeleton3D,
    draw_skeleton_3d
)


@dataclass
class BreathingMetrics:
    """Current breathing state from depth tracking."""
    signal: float           # Normalized [-1, 1], +1 = inhale
    depth_mm: float         # Current chest depth in mm
    phase: str              # "inhale", "exhale", "hold"
    bpm: float              # Breaths per minute
    amplitude_mm: float     # Breathing amplitude
    confidence: float       # Detection confidence


class SkeletonBreathingDetector:
    """
    Breathing detector using skeleton chest tracking.

    Uses the 3D skeleton's chest center depth to track breathing.
    More accurate than pose Y-axis or YOLO approaches.
    """

    def __init__(
        self,
        buffer_size: int = 150,       # ~5 seconds at 30fps
        smoothing_alpha: float = 0.15,
        calibration_frames: int = 90  # 3 seconds
    ):
        self.buffer_size = buffer_size
        self.smoothing_alpha = smoothing_alpha
        self.calibration_frames = calibration_frames

        # Buffers
        self.depth_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)

        # Smoothing
        self.smoothed_depth = None

        # Calibration
        self.baseline_depth = None
        self.calibration_values = []
        self.is_calibrated = False

        # Phase detection
        self.last_phase = "hold"
        self.phase_change_times = deque(maxlen=20)
        self.breath_intervals = deque(maxlen=10)

    def update(self, skeleton: Optional[Skeleton3D]) -> Optional[BreathingMetrics]:
        """Process skeleton and return breathing metrics."""
        current_time = time.time()

        if skeleton is None:
            return None

        # Get chest depth
        chest = skeleton.get_chest_center()
        if chest is None or not chest.is_valid:
            return None

        depth_mm = chest.z * 1000  # Convert to mm
        confidence = chest.visibility

        # Calibration phase
        if not self.is_calibrated:
            return self._calibrate(depth_mm, confidence, current_time)

        # Apply exponential smoothing
        if self.smoothed_depth is None:
            self.smoothed_depth = depth_mm
        else:
            self.smoothed_depth = (
                self.smoothing_alpha * depth_mm +
                (1 - self.smoothing_alpha) * self.smoothed_depth
            )

        # Store in buffer
        self.depth_buffer.append(self.smoothed_depth)
        self.time_buffer.append(current_time)

        # Calculate metrics
        signal = self._calculate_signal()
        phase = self._detect_phase()
        bpm = self._calculate_bpm()
        amplitude = self._calculate_amplitude()

        return BreathingMetrics(
            signal=signal,
            depth_mm=depth_mm,
            phase=phase,
            bpm=bpm,
            amplitude_mm=amplitude,
            confidence=confidence
        )

    def _calibrate(
        self,
        depth_mm: float,
        confidence: float,
        current_time: float
    ) -> BreathingMetrics:
        """Collect calibration data."""
        if confidence > 0.5:
            self.calibration_values.append(depth_mm)

        if len(self.calibration_values) >= self.calibration_frames:
            self.baseline_depth = np.median(self.calibration_values)
            self.smoothed_depth = self.baseline_depth
            self.is_calibrated = True
            print(f"[BreathingDetector] Calibrated! Baseline: {self.baseline_depth:.1f}mm")

        progress = len(self.calibration_values) / self.calibration_frames

        return BreathingMetrics(
            signal=0.0,
            depth_mm=depth_mm,
            phase="calibrating",
            bpm=0.0,
            amplitude_mm=0.0,
            confidence=progress
        )

    def _calculate_signal(self) -> float:
        """Normalize breathing signal to [-1, 1]."""
        if len(self.depth_buffer) < 10:
            return 0.0

        # Use recent window for range
        recent = list(self.depth_buffer)[-90:]  # Last 3 seconds
        min_d = np.min(recent)
        max_d = np.max(recent)
        range_d = max_d - min_d

        # Need minimum range
        if range_d < 2.0:  # Less than 2mm range
            if self.baseline_depth:
                deviation = self.baseline_depth - self.smoothed_depth
                return float(np.clip(deviation / 10.0, -1, 1))
            return 0.0

        # Normalize (lower depth = chest closer = inhale = positive)
        current = self.smoothed_depth
        normalized = -2 * (current - min_d) / range_d + 1

        return float(np.clip(normalized, -1, 1))

    def _detect_phase(self) -> str:
        """Detect breathing phase from signal trend."""
        if len(self.depth_buffer) < 10:
            return "hold"

        recent = list(self.depth_buffer)[-15:]
        if len(recent) < 8:
            return "hold"

        # Calculate trend
        first_half = np.mean(recent[:len(recent) // 2])
        second_half = np.mean(recent[len(recent) // 2:])
        derivative = first_half - second_half  # Positive = depth decreasing = inhale

        # Threshold (mm change)
        threshold = 0.3

        if derivative > threshold:
            new_phase = "inhale"
        elif derivative < -threshold:
            new_phase = "exhale"
        else:
            new_phase = "hold"

        # Track phase transitions for BPM
        if self.last_phase == "inhale" and new_phase == "exhale":
            current_time = time.time()
            if len(self.phase_change_times) > 0:
                interval = current_time - self.phase_change_times[-1]
                if 0.5 < interval < 15:  # Reasonable breath interval
                    self.breath_intervals.append(interval)
            self.phase_change_times.append(current_time)

        self.last_phase = new_phase
        return new_phase

    def _calculate_bpm(self) -> float:
        """Calculate breaths per minute."""
        if len(self.breath_intervals) < 2:
            return 0.0

        avg_interval = np.mean(self.breath_intervals)
        if avg_interval > 0:
            return 60.0 / avg_interval
        return 0.0

    def _calculate_amplitude(self) -> float:
        """Calculate breathing amplitude in mm."""
        if len(self.depth_buffer) < 30:
            return 0.0

        recent = list(self.depth_buffer)[-60:]  # Last 2 seconds
        return float(np.max(recent) - np.min(recent))

    def reset(self):
        """Reset detector state."""
        self.depth_buffer.clear()
        self.time_buffer.clear()
        self.smoothed_depth = None
        self.baseline_depth = None
        self.calibration_values.clear()
        self.is_calibrated = False
        self.breath_intervals.clear()
        self.phase_change_times.clear()


class BreathingSkeletonVisualizer:
    """Visualization for breathing + skeleton tracking."""

    def __init__(self):
        self.signal_history = deque(maxlen=200)
        self.show_skeleton = True
        self.show_waveform = True
        self.show_depth_preview = True

    def render(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        skeleton: Optional[Skeleton3D],
        metrics: Optional[BreathingMetrics],
        fps: float
    ) -> np.ndarray:
        """Render full visualization."""
        display = color_image.copy()
        h, w = display.shape[:2]

        # Draw skeleton
        if self.show_skeleton and skeleton:
            display = draw_skeleton_3d(display, skeleton, thickness=2)

        # Draw breathing indicator
        if metrics:
            self._draw_breathing_indicator(display, metrics)
            self.signal_history.append(metrics.signal)

        # Draw waveform
        if self.show_waveform:
            self._draw_waveform(display)

        # Draw depth preview
        if self.show_depth_preview and depth_image is not None:
            self._draw_depth_preview(display, depth_image)

        # Draw info panel
        self._draw_info(display, metrics, fps)

        return display

    def _draw_breathing_indicator(self, image: np.ndarray, metrics: BreathingMetrics):
        """Draw breathing circle indicator."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Base circle size varies with breathing
        base_radius = 80
        signal = metrics.signal
        radius = int(base_radius + signal * 30)

        # Color based on phase
        if metrics.phase == "calibrating":
            color = (128, 128, 128)
        elif metrics.phase == "inhale":
            color = (0, 255, 200)  # Cyan
        elif metrics.phase == "exhale":
            color = (255, 100, 100)  # Light red
        else:
            color = (200, 200, 200)  # Gray

        # Draw expanding/contracting circle
        cv2.circle(image, center, radius, color, 3)
        cv2.circle(image, center, radius - 10, color, 1)

        # Phase text
        cv2.putText(image, metrics.phase.upper(), (center[0] - 40, center[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_waveform(self, image: np.ndarray):
        """Draw breathing waveform at bottom."""
        h, w = image.shape[:2]
        wave_h = 80
        wave_y = h - wave_h - 20
        wave_w = w - 40

        # Background
        cv2.rectangle(image, (20, wave_y - 20), (20 + wave_w, wave_y + wave_h + 10),
                     (20, 20, 20), -1)
        cv2.rectangle(image, (20, wave_y - 20), (20 + wave_w, wave_y + wave_h + 10),
                     (60, 60, 60), 1)

        if len(self.signal_history) < 2:
            return

        # Draw center line
        center_y = wave_y + wave_h // 2
        cv2.line(image, (20, center_y), (20 + wave_w, center_y), (80, 80, 80), 1)

        # Draw signal
        signals = list(self.signal_history)
        points = []
        for i, sig in enumerate(signals):
            x = 20 + int((i / len(signals)) * wave_w)
            y = center_y - int(sig * (wave_h // 2 - 5))
            points.append((x, y))

        # Color gradient based on latest value
        latest = signals[-1] if signals else 0
        if latest > 0.2:
            color = (0, 255, 200)
        elif latest < -0.2:
            color = (255, 100, 100)
        else:
            color = (200, 200, 200)

        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], color, 2)

        # Labels
        cv2.putText(image, "Breathing Signal", (25, wave_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        cv2.putText(image, "+1", (wave_w + 5, wave_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(image, "-1", (wave_w + 5, wave_y + wave_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    def _draw_depth_preview(self, image: np.ndarray, depth_image: np.ndarray):
        """Draw small depth preview in corner."""
        h, w = image.shape[:2]

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )
        depth_small = cv2.resize(depth_colormap, (w // 5, h // 5))

        y1, y2 = 10, 10 + h // 5
        x1, x2 = w - w // 5 - 10, w - 10
        image[y1:y2, x1:x2] = depth_small

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(image, "Depth", (x1 + 5, y1 + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_info(self, image: np.ndarray, metrics: Optional[BreathingMetrics], fps: float):
        """Draw info panel."""
        h, w = image.shape[:2]

        # Background
        cv2.rectangle(image, (10, 10), (250, 150), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (250, 150), (60, 60, 60), 1)

        y = 30
        cv2.putText(image, "Breathing + Skeleton V2", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25

        cv2.putText(image, f"FPS: {fps:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25

        if metrics:
            if metrics.phase == "calibrating":
                progress = int(metrics.confidence * 100)
                cv2.putText(image, f"Calibrating... {progress}%", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            else:
                cv2.putText(image, f"Depth: {metrics.depth_mm:.0f}mm", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y += 22

                cv2.putText(image, f"Signal: {metrics.signal:+.2f}", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y += 22

                if metrics.bpm > 0:
                    cv2.putText(image, f"BPM: {metrics.bpm:.1f}", (20, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
        else:
            cv2.putText(image, "No skeleton detected", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Controls
        cv2.putText(image, "q:quit r:reset s:save", (20, h - 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)


def main():
    print("\n" + "=" * 60)
    print("  Breathing Detection + Skeleton Tracking V2")
    print("  MediaPipe Pose + RealSense Depth Fusion")
    print("=" * 60)
    print("\nThis version uses:")
    print("  - MediaPipe for accurate 2D landmark detection")
    print("  - RealSense aligned depth for true 3D coordinates")
    print("  - Chest center tracking for breathing detection")
    print("\nControls:")
    print("  q/ESC - Quit")
    print("  r     - Reset/recalibrate")
    print("  s     - Save screenshot")
    print("  w     - Toggle waveform")
    print("  k     - Toggle skeleton")
    print()

    # Create tracker and detector
    tracker = RealSenseSkeletonTracker(
        width=640,
        height=480,
        fps=30,
        model_complexity=1,
        enable_smoothing=True,
        depth_sample_radius=5  # Larger radius for breathing
    )

    if not tracker.start():
        print("\nFailed to start tracker!")
        print("Run with: sudo ./venv/bin/python src/breathing/test_breathing_skeleton_v2.py")
        return

    detector = SkeletonBreathingDetector(
        buffer_size=150,
        smoothing_alpha=0.12,
        calibration_frames=90
    )

    viz = BreathingSkeletonVisualizer()

    cv2.namedWindow("Breathing Skeleton V2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Breathing Skeleton V2", 960, 720)

    print("\nStarted! Stand in front of camera and breathe normally.")
    print("Wait for calibration to complete (~3 seconds).\n")

    try:
        while True:
            skeleton, color_image, depth_image = tracker.get_frame()

            if color_image is None:
                continue

            # Process breathing
            metrics = detector.update(skeleton)

            # Render
            fps = tracker.get_fps()
            display = viz.render(color_image, depth_image, skeleton, metrics, fps)

            cv2.imshow("Breathing Skeleton V2", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                print("Resetting...")
                detector.reset()
                tracker.reset_filters()
                viz.signal_history.clear()
            elif key == ord('s'):
                os.makedirs("outputs", exist_ok=True)
                filename = f"outputs/breathing_v2_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"Saved: {filename}")
            elif key == ord('w'):
                viz.show_waveform = not viz.show_waveform
            elif key == ord('k'):
                viz.show_skeleton = not viz.show_skeleton

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

    print("Done!")


if __name__ == "__main__":
    main()
