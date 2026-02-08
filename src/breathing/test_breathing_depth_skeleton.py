#!/usr/bin/env python3
"""
Breathing Detection with Skeleton Tracking - RealSense Depth + YOLO Pose

Combines:
- YOLO11n-pose for skeleton/person detection
- RealSense depth for precise chest breathing measurement
- Auto-positions ROI on detected chest area

MUST RUN WITH SUDO on macOS:
    sudo ./venv/bin/python -m src.breathing.test_breathing_depth_skeleton

Controls:
    q / ESC  - Quit
    r        - Reset/recalibrate
    s        - Save screenshot
    SPACE    - Toggle waveform display
    a        - Toggle auto-ROI (follow skeleton)
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

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics not available")
    print("Install with: pip install ultralytics")
    sys.exit(1)

from src.breathing.breath_detector import BreathingDetector, BreathingState, DetectionMode


# YOLO COCO skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                 # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),       # Arms
    (5, 11), (6, 12),                       # Torso
    (11, 12),                               # Hips
    (11, 13), (13, 15), (12, 14), (14, 16), # Legs
]


class DepthSkeletonBreathingTest:
    """Combined depth breathing + skeleton tracking test."""

    def __init__(self, resolution=(640, 480), fps=30):
        self.width = resolution[0]
        self.height = resolution[1]
        self.fps = fps

        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.depth_scale = 0.001

        # YOLO model
        print("  Loading YOLO pose model...")
        self.yolo = YOLO("yolo11n-pose.pt")
        print("  YOLO loaded.")

        # Breathing detector (created after pipeline starts)
        self.detector = None

        # Visualization
        self.show_waveform = True
        self.waveform_history = deque(maxlen=200)

        # ROI settings
        self.roi_x = self.width // 2 - 80
        self.roi_y = self.height // 3
        self.roi_w = 160
        self.roi_h = 120
        self.auto_roi = True  # Auto-follow skeleton chest

        # Skeleton tracking
        self.current_keypoints = None
        self.current_confidences = None
        self.skeleton_color = (0, 255, 0)  # Green

        # Ensure outputs directory
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

    def start(self):
        """Start RealSense pipeline."""
        print("\n" + "=" * 60)
        print("  BREATHING + SKELETON - Depth Mode")
        print("=" * 60)

        # Configure depth stream only
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

            # Create breathing detector
            self.detector = BreathingDetector(
                mode=DetectionMode.DEPTH,
                depth_scale=self.depth_scale
            )
            self.detector.set_chest_roi(self.roi_x, self.roi_y, self.roi_w, self.roi_h)

            print("\n  Controls:")
            print("    q/ESC     - Quit")
            print("    r         - Reset/recalibrate")
            print("    s         - Save screenshot")
            print("    SPACE     - Toggle waveform")
            print("    a         - Toggle auto-ROI (follow skeleton)")
            print("-" * 60)

            return True

        except RuntimeError as e:
            print(f"\n  ERROR: {e}")
            if "failed to set power state" in str(e):
                print("\n  Run with sudo:")
                print("    sudo ./venv/bin/python -m src.breathing.test_breathing_depth_skeleton")
            return False

    def stop(self):
        """Stop pipeline."""
        try:
            self.pipeline.stop()
        except:
            pass

    def run(self):
        """Main loop."""
        if not self.start():
            return

        cv2.namedWindow("Breathing + Skeleton", cv2.WINDOW_NORMAL)

        frame_count = 0
        start_time = time.time()
        yolo_interval = 3  # Run YOLO every N frames to save CPU

        try:
            while True:
                # Get depth frame
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=3000)
                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        print("  Warning: Frame timeout...")
                        continue
                    raise

                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                frame_count += 1

                # Convert depth to numpy
                depth_image = np.asanyarray(depth_frame.get_data())

                # Create colormap for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET
                )

                # Run YOLO pose detection periodically
                if frame_count % yolo_interval == 0:
                    self._detect_skeleton(depth_colormap)

                # Auto-position ROI on chest if skeleton detected
                if self.auto_roi and self.current_keypoints is not None:
                    self._update_roi_from_skeleton()

                # Update breathing detector
                state = self.detector.update(
                    frame=depth_colormap,
                    depth_frame=depth_image
                )

                self.waveform_history.append(state.signal)

                # Draw visualization
                display = self._draw_visualization(depth_colormap, depth_image, state)

                # FPS
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(display, f"FPS: {fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Breathing + Skeleton", display)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    break
                elif key == ord('r'):
                    print("  Resetting...")
                    self.detector.reset()
                    self.detector.set_chest_roi(self.roi_x, self.roi_y, self.roi_w, self.roi_h)
                    self.waveform_history.clear()
                elif key == ord('s'):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(PROJECT_ROOT, "outputs", f"skeleton_depth_{ts}.png")
                    cv2.imwrite(path, display)
                    print(f"  Saved: {path}")
                elif key == ord(' '):
                    self.show_waveform = not self.show_waveform
                elif key == ord('a'):
                    self.auto_roi = not self.auto_roi
                    print(f"  Auto-ROI: {'ON' if self.auto_roi else 'OFF'}")

        except KeyboardInterrupt:
            print("\n  Interrupted.")
        finally:
            self.stop()
            cv2.destroyAllWindows()
            print("\n  Done.\n")

    def _detect_skeleton(self, frame):
        """Run YOLO pose detection."""
        results = self.yolo(frame, conf=0.5, verbose=False)[0]

        if results.keypoints is not None and len(results.keypoints) > 0:
            # Get first person's keypoints
            kpts = results.keypoints.xy.cpu().numpy()
            confs = results.keypoints.conf.cpu().numpy()

            if len(kpts) > 0:
                self.current_keypoints = kpts[0]  # (17, 2)
                self.current_confidences = confs[0]  # (17,)
                return

        self.current_keypoints = None
        self.current_confidences = None

    def _update_roi_from_skeleton(self):
        """Position ROI on chest based on skeleton keypoints."""
        kpts = self.current_keypoints
        confs = self.current_confidences

        if kpts is None or confs is None:
            return

        # COCO keypoint indices:
        # 5 = left_shoulder, 6 = right_shoulder
        # 11 = left_hip, 12 = right_hip

        # Check if shoulders are visible
        left_shoulder_conf = confs[5] if len(confs) > 5 else 0
        right_shoulder_conf = confs[6] if len(confs) > 6 else 0
        left_hip_conf = confs[11] if len(confs) > 11 else 0
        right_hip_conf = confs[12] if len(confs) > 12 else 0

        # Need at least shoulders
        if left_shoulder_conf < 0.3 and right_shoulder_conf < 0.3:
            return

        # Calculate chest center
        points = []
        if left_shoulder_conf > 0.3:
            points.append(kpts[5])
        if right_shoulder_conf > 0.3:
            points.append(kpts[6])

        if len(points) == 0:
            return

        shoulder_center = np.mean(points, axis=0)

        # If hips visible, chest is between shoulders and hips
        hip_points = []
        if left_hip_conf > 0.3:
            hip_points.append(kpts[11])
        if right_hip_conf > 0.3:
            hip_points.append(kpts[12])

        if len(hip_points) > 0:
            hip_center = np.mean(hip_points, axis=0)
            # Chest is ~1/3 down from shoulders to hips
            chest_center = shoulder_center + 0.3 * (hip_center - shoulder_center)
        else:
            # Just use below shoulders
            chest_center = shoulder_center + np.array([0, 30])

        # Calculate ROI size based on shoulder width
        if left_shoulder_conf > 0.3 and right_shoulder_conf > 0.3:
            shoulder_width = np.linalg.norm(kpts[5] - kpts[6])
            roi_w = int(shoulder_width * 0.8)
            roi_h = int(shoulder_width * 0.6)
        else:
            roi_w = 120
            roi_h = 90

        # Ensure minimum size
        roi_w = max(80, min(roi_w, 250))
        roi_h = max(60, min(roi_h, 180))

        # Smoothly move ROI (don't jump suddenly)
        target_x = int(chest_center[0] - roi_w // 2)
        target_y = int(chest_center[1] - roi_h // 2)

        alpha = 0.15  # Smooth following
        self.roi_x = int(self.roi_x + alpha * (target_x - self.roi_x))
        self.roi_y = int(self.roi_y + alpha * (target_y - self.roi_y))
        self.roi_w = int(self.roi_w + alpha * (roi_w - self.roi_w))
        self.roi_h = int(self.roi_h + alpha * (roi_h - self.roi_h))

        # Clamp to frame
        self.roi_x = max(0, min(self.roi_x, self.width - self.roi_w))
        self.roi_y = max(0, min(self.roi_y, self.height - self.roi_h))

        self.detector.set_chest_roi(self.roi_x, self.roi_y, self.roi_w, self.roi_h)

    def _draw_visualization(self, color_image, depth_image, state):
        """Draw all overlays."""
        display = color_image.copy()

        # Draw skeleton
        self._draw_skeleton(display)

        # Draw ROI
        cv2.rectangle(display, (self.roi_x, self.roi_y),
                      (self.roi_x + self.roi_w, self.roi_y + self.roi_h),
                      (0, 255, 255), 2)
        cv2.putText(display, "CHEST ROI", (self.roi_x, self.roi_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Depth value in ROI
        roi_depth = depth_image[self.roi_y:self.roi_y+self.roi_h,
                                self.roi_x:self.roi_x+self.roi_w]
        valid = roi_depth[roi_depth > 0]
        if len(valid) > 0:
            avg_mm = np.mean(valid) * self.depth_scale * 1000
            cv2.putText(display, f"Depth: {avg_mm:.0f}mm",
                        (self.roi_x, self.roi_y + self.roi_h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Info panel
        self._draw_info_panel(display, state)

        # Waveform
        if self.show_waveform:
            self._draw_waveform(display, state)

        # Breathing indicator
        self._draw_breathing_indicator(display, state)

        return display

    def _draw_skeleton(self, display):
        """Draw skeleton on frame."""
        if self.current_keypoints is None or self.current_confidences is None:
            return

        kpts = self.current_keypoints
        confs = self.current_confidences

        # Draw keypoints
        for i in range(17):
            if confs[i] > 0.3:
                x, y = int(kpts[i][0]), int(kpts[i][1])
                cv2.circle(display, (x, y), 4, self.skeleton_color, -1)

        # Draw connections
        for i, j in SKELETON_CONNECTIONS:
            if confs[i] > 0.3 and confs[j] > 0.3:
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(display, pt1, pt2, self.skeleton_color, 2)

    def _draw_info_panel(self, display, state):
        """Draw info panel."""
        h, w = display.shape[:2]
        panel_h = 180
        cv2.rectangle(display, (w - 220, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.rectangle(display, (w - 220, 0), (w, panel_h), (100, 100, 100), 1)

        cv2.putText(display, "DEPTH + SKELETON", (w - 210, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        y = 45
        metrics = [
            f"Signal: {state.signal:+.2f}",
            f"Phase: {state.phase}",
            f"BPM: {state.breath_rate:.1f}",
            f"Conf: {state.confidence:.2f}",
        ]

        for m in metrics:
            cv2.putText(display, m, (w - 210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 20

        if state.raw_value > 0:
            cv2.putText(display, f"Dist: {state.raw_value:.0f}mm", (w - 210, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            y += 20

        # Adaptive params
        if self.detector:
            cv2.putText(display, f"Thresh: {self.detector.adaptive_threshold:.2f}mm",
                        (w - 210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
            y += 18
            cv2.putText(display, f"Smooth: {self.detector.adaptive_smoothing:.2f}",
                        (w - 210, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
            y += 18

        # Auto-ROI status
        roi_status = "Auto-ROI: ON" if self.auto_roi else "Auto-ROI: OFF"
        color = (100, 255, 100) if self.auto_roi else (150, 150, 150)
        cv2.putText(display, roi_status, (w - 210, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _draw_waveform(self, display, state):
        """Draw breathing waveform."""
        h, w = display.shape[:2]
        wf_h = 100
        wf_y = h - wf_h - 10

        cv2.rectangle(display, (10, wf_y), (w - 10, h - 10), (20, 20, 20), -1)
        cv2.rectangle(display, (10, wf_y), (w - 10, h - 10), (80, 80, 80), 1)

        center_y = wf_y + wf_h // 2
        cv2.line(display, (10, center_y), (w - 10, center_y), (60, 60, 60), 1)

        if len(self.waveform_history) > 1:
            history = list(self.waveform_history)
            n = len(history)
            plot_w = w - 20

            points = []
            for i, val in enumerate(history):
                px = 10 + int(i * plot_w / max(n - 1, 1))
                py = center_y - int(val * (wf_h // 2 - 5))
                points.append((px, py))

            for j in range(1, len(points)):
                if state.phase == "inhale":
                    color = (100, 255, 100)
                elif state.phase == "exhale":
                    color = (100, 100, 255)
                else:
                    color = (200, 200, 200)
                cv2.line(display, points[j-1], points[j], color, 2)

        cv2.putText(display, "+1 (inhale)", (15, wf_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 255, 100), 1)
        cv2.putText(display, "-1 (exhale)", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 255), 1)

    def _draw_breathing_indicator(self, display, state):
        """Draw pulsing circle."""
        cx, cy = 80, 80
        base_r = 30
        breath_mod = int(state.signal * 20)
        radius = base_r + breath_mod

        if state.phase == "inhale":
            color = (100, 255, 100)
        elif state.phase == "exhale":
            color = (100, 100, 255)
        elif state.phase == "calibrating":
            pulse = int(127 + 127 * np.sin(time.time() * 4))
            color = (0, pulse, 255)
        else:
            color = (200, 200, 200)

        cv2.circle(display, (cx, cy), radius, color, -1)
        cv2.putText(display, state.phase.upper(), (cx - 30, cy + radius + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    if sys.platform == "darwin" and os.geteuid() != 0:
        print("\n" + "=" * 60)
        print("  WARNING: Run with sudo on macOS")
        print("=" * 60)
        print("\n  sudo ./venv/bin/python -m src.breathing.test_breathing_depth_skeleton\n")

    test = DepthSkeletonBreathingTest()
    test.run()


if __name__ == "__main__":
    main()
