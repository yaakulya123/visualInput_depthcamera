#!/usr/bin/env python3
"""
Breathing Detection Test - RealSense Depth Mode

Uses the Intel RealSense D435 depth stream for precise Z-axis breathing detection.
Tracks chest rise/fall with 5-10mm precision.

MUST RUN WITH SUDO on macOS:
    sudo python -m src.breathing.test_breathing_depth

Controls:
    q / ESC  - Quit
    r        - Reset/recalibrate
    s        - Save screenshot
    SPACE    - Toggle waveform display
    +/-      - Adjust ROI size
    Arrow keys - Move ROI position
"""

import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from collections import deque

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("ERROR: pyrealsense2 not available")
    print("Install with: pip install pyrealsense2-macosx")
    sys.exit(1)

from src.breathing.breath_detector import BreathingDetector, BreathingState, DetectionMode


class RealSenseBreathingTest:
    """Interactive test for depth-based breathing detection."""

    def __init__(self, resolution=(640, 480), fps=30):
        self.width = resolution[0]
        self.height = resolution[1]
        self.fps = fps

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.depth_scale = 0.001  # Default, will be updated from device

        # Breathing detector in DEPTH mode (depth_scale set after pipeline starts)
        self.detector = None  # Created after we get depth_scale

        # Visualization
        self.show_waveform = True
        self.waveform_history = deque(maxlen=200)

        # ROI adjustment (will be applied when detector is created)
        # Larger default ROI for better long-distance performance
        self.roi_x = self.width // 2 - 80
        self.roi_y = self.height // 3
        self.roi_w = 160
        self.roi_h = 120
        self.roi_step = 10

        # Auto-adjust ROI flag
        self.auto_roi = True

        # Ensure outputs directory exists
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

    def _update_roi(self):
        """Update detector ROI if detector exists."""
        if self.detector is not None:
            self.detector.set_chest_roi(self.roi_x, self.roi_y, self.roi_w, self.roi_h)

    def _auto_adjust_roi(self, distance_mm: float):
        """
        Automatically adjust ROI size based on distance.
        Larger ROI at longer distances for better noise averaging.
        """
        # Target ROI sizes based on distance
        if distance_mm < 500:
            target_w, target_h = 100, 80
        elif distance_mm < 1000:
            target_w, target_h = 140, 100
        elif distance_mm < 1500:
            target_w, target_h = 180, 130
        elif distance_mm < 2500:
            target_w, target_h = 220, 160
        else:
            target_w, target_h = 280, 200

        # Smoothly adjust (don't jump suddenly)
        alpha = 0.02  # Slow adjustment
        self.roi_w = int(self.roi_w + alpha * (target_w - self.roi_w))
        self.roi_h = int(self.roi_h + alpha * (target_h - self.roi_h))

        # Keep ROI centered
        self.roi_x = max(0, min(self.width - self.roi_w, self.width // 2 - self.roi_w // 2))
        self.roi_y = max(0, min(self.height - self.roi_h, self.height // 3))

        self._update_roi()

    def start(self):
        """Start RealSense pipeline."""
        print("\n" + "=" * 60)
        print("  BREATHING DETECTION - RealSense Depth Mode")
        print("=" * 60)

        # Configure depth stream only (color causes bandwidth issues on macOS)
        self.config.enable_stream(rs.stream.depth, self.width, self.height,
                                   rs.format.z16, self.fps)

        try:
            print("  Starting RealSense pipeline...")
            profile = self.pipeline.start(self.config)

            # Get depth scale
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"  Depth scale: {self.depth_scale:.6f} m/unit")
            print(f"  Resolution: {self.width}x{self.height} @ {self.fps}fps")
            print("  Mode: Depth-only (no color stream)")

            # No alignment needed without color
            self.align = None

            # Now create the breathing detector with correct depth scale
            self.detector = BreathingDetector(
                mode=DetectionMode.DEPTH,
                depth_scale=self.depth_scale
            )

            # Set initial ROI on detector
            self.detector.set_chest_roi(self.roi_x, self.roi_y, self.roi_w, self.roi_h)

            print("\n  Controls:")
            print("    q/ESC     - Quit")
            print("    r         - Reset/recalibrate")
            print("    s         - Save screenshot")
            print("    SPACE     - Toggle waveform")
            print("    +/-       - Adjust ROI size")
            print("    Arrows    - Move ROI position")
            print("    a         - Toggle auto-ROI sizing")
            print("-" * 60)

            return True

        except RuntimeError as e:
            print(f"\n  ERROR: {e}")
            if "failed to set power state" in str(e):
                print("\n  This script must be run with sudo on macOS:")
                print("    sudo python -m src.breathing.test_breathing_depth")
            return False

    def stop(self):
        """Stop RealSense pipeline."""
        try:
            self.pipeline.stop()
        except:
            pass

    def run(self):
        """Main test loop."""
        if not self.start():
            return

        cv2.namedWindow("Breathing Detection - Depth", cv2.WINDOW_NORMAL)

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                # Wait for frames with timeout handling
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=3000)
                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        print("  Warning: Frame timeout, retrying...")
                        continue
                    raise

                depth_frame = frames.get_depth_frame()

                if not depth_frame:
                    continue

                frame_count += 1

                # Convert to numpy
                depth_image = np.asanyarray(depth_frame.get_data())

                # Create a grayscale "color" image from depth for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Update detector with depth frame (pass depth colormap as frame for viz)
                state = self.detector.update(
                    frame=depth_colormap,
                    depth_frame=depth_image
                )

                # Store waveform data
                self.waveform_history.append(state.signal)

                # Auto-adjust ROI size based on distance
                if self.auto_roi and state.raw_value > 0:
                    self._auto_adjust_roi(state.raw_value)

                # Draw visualization (use depth colormap since we don't have color)
                display = self.draw_visualization(depth_colormap, depth_image, state)

                # Calculate FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Add FPS overlay
                cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Breathing Detection - Depth", display)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    break

                elif key == ord('r'):
                    print("  Resetting detector...")
                    self.detector.reset()
                    self._update_roi()
                    self.waveform_history.clear()
                    print("  Reset complete. Recalibrating...")

                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(PROJECT_ROOT, "outputs",
                                         f"breathing_depth_{timestamp}.png")
                    cv2.imwrite(path, display)
                    print(f"  Screenshot saved: {path}")

                elif key == ord(' '):
                    self.show_waveform = not self.show_waveform
                    print(f"  Waveform: {'ON' if self.show_waveform else 'OFF'}")

                elif key == ord('+') or key == ord('='):
                    self.roi_w = min(self.roi_w + self.roi_step, self.width // 2)
                    self.roi_h = min(self.roi_h + self.roi_step, self.height // 2)
                    self._update_roi()
                    print(f"  ROI size: {self.roi_w}x{self.roi_h}")

                elif key == ord('-') or key == ord('_'):
                    self.roi_w = max(self.roi_w - self.roi_step, 40)
                    self.roi_h = max(self.roi_h - self.roi_step, 40)
                    self._update_roi()
                    print(f"  ROI size: {self.roi_w}x{self.roi_h}")

                elif key == 81 or key == 2:  # Left arrow
                    self.roi_x = max(0, self.roi_x - self.roi_step)
                    self._update_roi()

                elif key == 83 or key == 3:  # Right arrow
                    self.roi_x = min(self.width - self.roi_w, self.roi_x + self.roi_step)
                    self._update_roi()

                elif key == 82 or key == 0:  # Up arrow
                    self.roi_y = max(0, self.roi_y - self.roi_step)
                    self._update_roi()

                elif key == 84 or key == 1:  # Down arrow
                    self.roi_y = min(self.height - self.roi_h, self.roi_y + self.roi_step)
                    self._update_roi()

                elif key == ord('a'):
                    self.auto_roi = not self.auto_roi
                    print(f"  Auto-ROI: {'ON' if self.auto_roi else 'OFF'}")

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            self.stop()
            cv2.destroyAllWindows()
            print("\n  Done.\n")

    def draw_visualization(self, color_image: np.ndarray,
                           depth_image: np.ndarray,
                           state: BreathingState) -> np.ndarray:
        """Draw the visualization with all overlays."""
        h, w = color_image.shape[:2]

        # Use the color image directly (it's already the depth colormap)
        display = color_image.copy()

        # Draw ROI rectangle
        rx, ry, rw, rh = self.roi_x, self.roi_y, self.roi_w, self.roi_h
        if self.detector and self.detector.chest_roi:
            rx, ry, rw, rh = self.detector.chest_roi
        if rx >= 0 and ry >= 0:
            # Main ROI box
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            # Label
            cv2.putText(display, "CHEST ROI", (rx, ry - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # Show depth value in ROI
            roi_depth = depth_image[ry:ry+rh, rx:rx+rw]
            valid_mask = roi_depth > 0
            if np.any(valid_mask):
                avg_depth_mm = np.mean(roi_depth[valid_mask]) * self.depth_scale * 1000
                cv2.putText(display, f"Depth: {avg_depth_mm:.0f}mm",
                            (rx, ry + rh + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw breathing info panel
        self.draw_info_panel(display, state)

        # Draw waveform if enabled
        if self.show_waveform:
            self.draw_waveform(display, state)

        # Draw pulsing circle based on breathing
        self.draw_breathing_indicator(display, state)

        return display

    def draw_info_panel(self, display: np.ndarray, state: BreathingState):
        """Draw the info panel with breathing metrics."""
        h, w = display.shape[:2]

        # Background panel
        panel_h = 160
        cv2.rectangle(display, (w - 220, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.rectangle(display, (w - 220, 0), (w, panel_h), (100, 100, 100), 1)

        # Title with adaptive indicator
        cv2.putText(display, "DEPTH MODE (Adaptive)", (w - 210, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Metrics
        y = 45
        metrics = [
            f"Signal: {state.signal:+.2f}",
            f"Phase: {state.phase}",
            f"BPM: {state.breath_rate:.1f}",
            f"Conf: {state.confidence:.2f}",
        ]

        for metric in metrics:
            cv2.putText(display, metric, (w - 210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 20

        # Raw depth value (already in mm from detector)
        if state.raw_value > 0:
            cv2.putText(display, f"Dist: {state.raw_value:.0f}mm", (w - 210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y += 20

        # Show adaptive parameters
        if self.detector:
            thresh = self.detector.adaptive_threshold
            smooth = self.detector.adaptive_smoothing
            cv2.putText(display, f"Thresh: {thresh:.1f}mm", (w - 210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
            y += 18
            cv2.putText(display, f"Smooth: {smooth:.2f}", (w - 210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)

    def draw_waveform(self, display: np.ndarray, state: BreathingState):
        """Draw the breathing waveform at the bottom."""
        h, w = display.shape[:2]

        waveform_h = 100
        waveform_y = h - waveform_h - 10

        # Background
        cv2.rectangle(display, (10, waveform_y), (w - 10, h - 10),
                      (20, 20, 20), -1)
        cv2.rectangle(display, (10, waveform_y), (w - 10, h - 10),
                      (80, 80, 80), 1)

        # Center line
        center_y = waveform_y + waveform_h // 2
        cv2.line(display, (10, center_y), (w - 10, center_y),
                 (60, 60, 60), 1)

        # Draw waveform
        if len(self.waveform_history) > 1:
            history = list(self.waveform_history)
            n = len(history)
            plot_w = w - 20

            points = []
            for i, val in enumerate(history):
                px = 10 + int(i * plot_w / max(n - 1, 1))
                # Map signal [-1, 1] to [waveform_y + waveform_h, waveform_y]
                py = center_y - int(val * (waveform_h // 2 - 5))
                points.append((px, py))

            # Draw line
            for j in range(1, len(points)):
                # Color based on phase
                if state.phase == "inhale":
                    color = (100, 255, 100)  # Green
                elif state.phase == "exhale":
                    color = (100, 100, 255)  # Red
                else:
                    color = (200, 200, 200)  # Gray
                cv2.line(display, points[j - 1], points[j], color, 2)

        # Labels
        cv2.putText(display, "+1 (inhale)", (15, waveform_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
        cv2.putText(display, "-1 (exhale)", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)

    def draw_breathing_indicator(self, display: np.ndarray, state: BreathingState):
        """Draw a pulsing circle that responds to breathing."""
        h, w = display.shape[:2]

        # Position in top-left
        cx, cy = 80, 80

        # Base radius + breathing modulation
        base_radius = 30
        breath_mod = int(state.signal * 20)
        radius = base_radius + breath_mod

        # Color based on phase
        if state.phase == "inhale":
            color = (100, 255, 100)  # Green
        elif state.phase == "exhale":
            color = (100, 100, 255)  # Red
        elif state.phase == "calibrating":
            # Pulsing yellow during calibration
            pulse = int(127 + 127 * np.sin(time.time() * 4))
            color = (0, pulse, 255)
        else:
            color = (200, 200, 200)  # Gray

        # Draw circle with glow effect
        for i in range(3):
            r = radius + i * 5
            alpha = 0.3 - i * 0.1
            overlay = display.copy()
            cv2.circle(overlay, (cx, cy), r, color, 2)
            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

        # Solid inner circle
        cv2.circle(display, (cx, cy), radius, color, -1)

        # Phase text
        cv2.putText(display, state.phase.upper(), (cx - 30, cy + radius + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    """Main entry point."""
    # Check if running as root (required on macOS)
    if sys.platform == "darwin" and os.geteuid() != 0:
        print("\n" + "=" * 60)
        print("  WARNING: This script should be run with sudo on macOS")
        print("=" * 60)
        print("\n  Run with:")
        print("    sudo python -m src.breathing.test_breathing_depth")
        print("\n  Attempting to continue anyway...\n")

    test = RealSenseBreathingTest()
    test.run()


if __name__ == "__main__":
    main()
