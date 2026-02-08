#!/usr/bin/env python3
"""
Test RealSense Skeleton Tracker - MediaPipe + Depth Fusion

This test demonstrates 3D skeleton tracking using:
- MediaPipe Pose for accurate 2D landmark detection
- RealSense D435 depth for true 3D coordinates
- One-Euro filtering for smooth tracking

Run with: sudo ./venv/bin/python src/tracking/test_realsense_skeleton.py

Controls:
  q/ESC - Quit
  r     - Reset filters
  s     - Save screenshot
  d     - Toggle depth overlay
  i     - Show landmark info
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import time
from collections import deque

from src.tracking.realsense_skeleton import (
    RealSenseSkeletonTracker,
    Skeleton3D,
    draw_skeleton_3d
)


class BreathingPhaseDetector:
    """
    Detects breathing phase (inhale/exhale) using derivative analysis.
    Based on: PMC4611185 - Respiratory rate detection algorithm
    """

    def __init__(self):
        # Weighted sliding window (research recommends 4 samples)
        self.weights = [1.0, 0.7, 0.4, 0.1]
        self.raw_buffer = deque(maxlen=10)
        self.smoothed_buffer = deque(maxlen=10)

        # Phase detection
        self.derivative_signs = deque(maxlen=5)
        self.current_phase = "hold"
        self.phase_confidence = 0.0

        # Breath tracking
        self.breath_count = 0
        self.last_phase_change = time.time()
        self.breath_times = deque(maxlen=10)

        # Signal for visualization
        self.normalized_signal = 0.0  # -1 to +1

    def update(self, depth_mm: float) -> str:
        """Update with new depth reading, return phase."""
        self.raw_buffer.append(depth_mm)

        if len(self.raw_buffer) < 4:
            return "calibrating"

        # Apply weighted average (equation from paper)
        weighted_sum = 0
        weight_sum = 0
        for i, w in enumerate(self.weights):
            if i < len(self.raw_buffer):
                idx = -(i + 1)
                weighted_sum += self.raw_buffer[idx] * w
                weight_sum += w
        smoothed = weighted_sum / weight_sum if weight_sum > 0 else depth_mm

        self.smoothed_buffer.append(smoothed)

        if len(self.smoothed_buffer) < 3:
            return "calibrating"

        # Calculate derivative (sign indicates direction)
        derivative = self.smoothed_buffer[-1] - self.smoothed_buffer[-2]

        # Track derivative sign (+ = moving away = exhale, - = moving closer = inhale)
        sign = 1 if derivative > 0.1 else (-1 if derivative < -0.1 else 0)
        self.derivative_signs.append(sign)

        # Need 3 consistent signs for phase detection (paper recommendation)
        if len(self.derivative_signs) >= 3:
            recent_signs = list(self.derivative_signs)[-3:]

            if all(s < 0 for s in recent_signs):  # Consistently moving closer
                new_phase = "inhale"
                self.phase_confidence = 1.0
            elif all(s > 0 for s in recent_signs):  # Consistently moving away
                new_phase = "exhale"
                self.phase_confidence = 1.0
            elif all(s == 0 for s in recent_signs):  # No movement
                new_phase = "hold"
                self.phase_confidence = 0.5
            else:
                new_phase = self.current_phase  # Keep current
                self.phase_confidence = 0.3

            # Track breath count (inhale->exhale transition)
            if self.current_phase == "inhale" and new_phase == "exhale":
                self.breath_count += 1
                self.breath_times.append(time.time())

            self.current_phase = new_phase

        # Calculate normalized signal for visualization
        if len(self.smoothed_buffer) >= 10:
            recent = list(self.smoothed_buffer)[-10:]
            min_d = min(recent)
            max_d = max(recent)
            range_d = max_d - min_d if max_d > min_d else 1.0
            # Invert: closer = inhale = positive
            self.normalized_signal = -((smoothed - min_d) / range_d * 2 - 1)
            self.normalized_signal = max(-1, min(1, self.normalized_signal))

        return self.current_phase

    def get_bpm(self) -> float:
        """Calculate breaths per minute."""
        if len(self.breath_times) < 2:
            return 0.0
        intervals = []
        times = list(self.breath_times)
        for i in range(1, len(times)):
            intervals.append(times[i] - times[i-1])
        if intervals:
            avg_interval = np.mean(intervals)
            return 60.0 / avg_interval if avg_interval > 0 else 0.0
        return 0.0


class SkeletonTestVisualizer:
    """Visualizer for skeleton tracking test."""

    def __init__(self):
        self.show_depth_overlay = True
        self.show_info = True
        self.chest_depth_history = deque(maxlen=150)
        self.start_time = time.time()

        # Breathing phase detector
        self.breath_detector = BreathingPhaseDetector()

    def render(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        skeleton: Skeleton3D,
        fps: float
    ) -> np.ndarray:
        """Render visualization."""
        display = color_image.copy()
        h, w = display.shape[:2]

        chest_z = 0
        phase = "hold"

        # Draw skeleton
        if skeleton:
            display = draw_skeleton_3d(display, skeleton, thickness=2)

            # Get chest info from shoulders
            if len(skeleton.landmarks) >= 13:
                left_sh = skeleton.landmarks[11]
                right_sh = skeleton.landmarks[12]

                if left_sh.visibility > 0.5 and right_sh.visibility > 0.5:
                    # Calculate chest center
                    cx = int((left_sh.x + right_sh.x) / 2 * w)
                    cy = int((left_sh.y + right_sh.y) / 2 * h)
                    chest_z = (left_sh.z + right_sh.z) / 2

                    if chest_z > 0:
                        chest_mm = chest_z * 1000
                        self.chest_depth_history.append(chest_z)

                        # Update breathing phase detector
                        phase = self.breath_detector.update(chest_mm)

                        # Draw small chest marker on skeleton
                        cv2.circle(display, (cx, cy), 8, (0, 255, 255), 2)

        # Draw breathing circle (main visual feedback)
        self._draw_breathing_circle(display, phase, chest_z)

        # Depth overlay (side panel)
        if self.show_depth_overlay and depth_image is not None:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            # Resize depth to quarter size
            depth_small = cv2.resize(depth_colormap, (w // 4, h // 4))
            # Place in corner
            display[10:10 + h // 4, w - w // 4 - 10:w - 10] = depth_small

        # Info panel
        if self.show_info:
            self._draw_info_panel(display, skeleton, fps)

        # Breathing waveform
        if len(self.chest_depth_history) > 2:
            self._draw_breathing_wave(display)

        return display

    def _draw_info_panel(self, image: np.ndarray, skeleton: Skeleton3D, fps: float):
        """Draw info overlay."""
        h, w = image.shape[:2]

        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (280, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

        # Text
        y = 35
        cv2.putText(image, "RealSense + MediaPipe Skeleton", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25

        cv2.putText(image, f"FPS: {fps:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25

        if skeleton:
            # Chest depth
            if len(self.chest_depth_history) > 0:
                chest_z = self.chest_depth_history[-1]
                cv2.putText(image, f"Chest Depth: {chest_z:.3f}m", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y += 25

            # Breathing range
            if len(self.chest_depth_history) > 10:
                min_d = min(self.chest_depth_history)
                max_d = max(self.chest_depth_history)
                range_mm = (max_d - min_d) * 1000
                cv2.putText(image, f"Breath Range: {range_mm:.1f}mm", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                y += 25

            # Current phase
            phase = self.breath_detector.current_phase
            phase_color = (0, 255, 200) if phase == "inhale" else (255, 150, 100) if phase == "exhale" else (200, 200, 200)
            cv2.putText(image, f"Phase: {phase.upper()}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, phase_color, 1)
            y += 25

            valid_count = sum(1 for lm in skeleton.landmarks if lm.visibility > 0.5)
            cv2.putText(image, f"Landmarks: {valid_count}/33", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            cv2.putText(image, "No skeleton detected", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Controls hint
        cv2.putText(image, "q:quit r:reset d:depth i:info s:save", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _draw_breathing_circle(self, image: np.ndarray, phase: str, chest_z: float):
        """Draw animated breathing circle indicator."""
        h, w = image.shape[:2]

        # Position circle in center-right area
        center_x = w - 120
        center_y = h // 2 - 50

        # Base radius varies with breathing signal
        signal = self.breath_detector.normalized_signal
        base_radius = 60
        radius = int(base_radius + signal * 25)  # Expands on inhale

        # Color based on phase
        if phase == "calibrating":
            color = (128, 128, 128)  # Gray
            text_color = (150, 150, 150)
        elif phase == "inhale":
            color = (0, 255, 200)  # Cyan/green
            text_color = (0, 255, 200)
        elif phase == "exhale":
            color = (255, 150, 100)  # Light orange/blue
            text_color = (255, 150, 100)
        else:  # hold
            color = (200, 200, 200)  # Light gray
            text_color = (200, 200, 200)

        # Draw outer glow
        for i in range(3, 0, -1):
            glow_radius = radius + i * 8
            glow_alpha = 0.1 * i
            glow_color = tuple(int(c * glow_alpha) for c in color)
            cv2.circle(image, (center_x, center_y), glow_radius, glow_color, 2)

        # Draw main circle
        cv2.circle(image, (center_x, center_y), radius, color, 3)

        # Draw inner circle
        inner_radius = max(5, radius - 20)
        cv2.circle(image, (center_x, center_y), inner_radius, color, -1)

        # Draw phase text
        phase_text = phase.upper()
        text_size = cv2.getTextSize(phase_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = center_y + radius + 30
        cv2.putText(image, phase_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

        # Draw BPM
        bpm = self.breath_detector.get_bpm()
        if bpm > 0:
            bpm_text = f"{bpm:.1f} BPM"
            bpm_size = cv2.getTextSize(bpm_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(image, bpm_text, (center_x - bpm_size[0] // 2, text_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        # Draw breath count
        count_text = f"Breaths: {self.breath_detector.breath_count}"
        cv2.putText(image, count_text, (center_x - 40, center_y - radius - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _draw_breathing_wave(self, image: np.ndarray):
        """Draw breathing waveform from chest depth history."""
        h, w = image.shape[:2]
        wave_h = 80
        wave_y = h - wave_h - 30
        wave_w = w - 40

        # Background with border
        cv2.rectangle(image, (20, wave_y - 20), (20 + wave_w, wave_y + wave_h + 5),
                     (20, 20, 20), -1)
        cv2.rectangle(image, (20, wave_y - 20), (20 + wave_w, wave_y + wave_h + 5),
                     (60, 60, 60), 1)

        if len(self.chest_depth_history) < 2:
            return

        # Convert to mm and smooth
        depths_mm = [d * 1000 for d in self.chest_depth_history]

        # Apply simple moving average for smoother display
        smoothed = []
        window = 3
        for i in range(len(depths_mm)):
            start = max(0, i - window)
            end = min(len(depths_mm), i + window + 1)
            smoothed.append(np.mean(depths_mm[start:end]))

        min_d = min(smoothed)
        max_d = max(smoothed)
        range_d = max_d - min_d if max_d > min_d else 1.0

        points = []
        for i, d in enumerate(smoothed):
            x = 20 + int((i / len(smoothed)) * wave_w)
            # Invert so closer (inhale) goes up
            norm_d = (d - min_d) / range_d
            y = wave_y + wave_h - int(norm_d * wave_h * 0.85) - 5
            points.append((x, y))

        # Draw center line
        center_y = wave_y + wave_h // 2
        cv2.line(image, (20, center_y), (20 + wave_w, center_y), (50, 50, 50), 1)

        # Draw wave with thicker line
        for i in range(1, len(points)):
            cv2.line(image, points[i - 1], points[i], (0, 200, 255), 2)

        # Labels
        cv2.putText(image, f"Chest Depth | Range: {range_d:.1f}mm", (25, wave_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        cv2.putText(image, "inhale", (wave_w - 20, wave_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        cv2.putText(image, "exhale", (wave_w - 20, wave_y + wave_h - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)


def main():
    print("\n" + "=" * 60)
    print("  RealSense Skeleton Tracker Test")
    print("  MediaPipe Pose + Depth Fusion")
    print("=" * 60)
    print("\nControls:")
    print("  q/ESC - Quit")
    print("  r     - Reset filters")
    print("  s     - Save screenshot")
    print("  d     - Toggle depth overlay")
    print("  i     - Toggle info panel")
    print()

    # Create tracker
    # model_complexity=0 requires download (SSL issues on macOS)
    # model_complexity=1 for balanced (default) - already downloaded
    # model_complexity=2 for most accurate (slowest)
    tracker = RealSenseSkeletonTracker(
        width=640,
        height=480,
        fps=30,
        model_complexity=1,  # Use model that's already downloaded
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_smoothing=True,
        depth_sample_radius=5  # Larger radius for more stable depth
    )

    if not tracker.start():
        print("\nFailed to start tracker!")
        print("Make sure to run with sudo on macOS:")
        print("  sudo ./venv/bin/python src/tracking/test_realsense_skeleton.py")
        return

    # Create visualizer
    viz = SkeletonTestVisualizer()

    # Create window
    cv2.namedWindow("Skeleton Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Skeleton Tracker", 960, 720)

    print("\nTracker started! Position yourself in front of the camera.")
    print("Press 'q' to quit.\n")

    frame_count = 0
    no_skeleton_count = 0

    try:
        while True:
            # Get frame
            skeleton, color_image, depth_image = tracker.get_frame()

            if color_image is None:
                continue

            frame_count += 1

            if skeleton is None:
                no_skeleton_count += 1
                if no_skeleton_count > 30:
                    cv2.putText(color_image, "No person detected - stand in frame",
                               (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                no_skeleton_count = 0

            # Render visualization
            fps = tracker.get_fps()
            display = viz.render(color_image, depth_image, skeleton, fps)

            # Show
            cv2.imshow("Skeleton Tracker", display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                print("Resetting filters and breath detector...")
                tracker.reset_filters()
                viz.chest_depth_history.clear()
                viz.breath_detector = BreathingPhaseDetector()
            elif key == ord('d'):
                viz.show_depth_overlay = not viz.show_depth_overlay
                print(f"Depth overlay: {'ON' if viz.show_depth_overlay else 'OFF'}")
            elif key == ord('i'):
                viz.show_info = not viz.show_info
                print(f"Info panel: {'ON' if viz.show_info else 'OFF'}")
            elif key == ord('s'):
                # Save screenshot
                os.makedirs("outputs", exist_ok=True)
                filename = f"outputs/skeleton_{int(time.time())}.png"
                cv2.imwrite(filename, display)
                print(f"Saved: {filename}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        tracker.stop()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    print("Done!")


if __name__ == "__main__":
    main()
