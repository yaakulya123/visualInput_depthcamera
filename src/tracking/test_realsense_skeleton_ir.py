#!/usr/bin/env python3
"""
RealSense Skeleton Tracker - Using Infrared Stream

This version uses only depth + infrared streams (no color) which is more
reliable on macOS. MediaPipe can still detect poses from grayscale IR images.

Run with: sudo ./venv/bin/python src/tracking/test_realsense_skeleton_ir.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import time
from collections import deque

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 not installed")
    print("Install with: pip install pyrealsense2-macosx")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("Error: mediapipe not installed")
    sys.exit(1)


class OneEuroFilter:
    """Simple One-Euro filter for smoothing."""
    def __init__(self, min_cutoff=0.5, beta=0.5, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        alpha_d = 1.0 / (1.0 + 1.0 / (2 * np.pi * self.d_cutoff * dt))
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = 1.0 / (1.0 + 1.0 / (2 * np.pi * cutoff * dt))
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        self.x_prev = x_hat

        return x_hat


def main():
    print("\n" + "=" * 60)
    print("  RealSense Skeleton Tracker (IR Mode)")
    print("  Using Infrared + Depth streams")
    print("=" * 60)

    # Initialize RealSense
    pipeline = rs.pipeline()
    config = rs.config()

    # Only enable depth and infrared (more reliable on macOS)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

    print("\nStarting RealSense (depth + infrared only)...")

    try:
        profile = pipeline.start(config)
        print("Pipeline started!")
    except RuntimeError as e:
        print(f"\nError: {e}")
        if "failed to set power state" in str(e):
            print("\nTry these steps:")
            print("1. Unplug the RealSense camera")
            print("2. Wait 5 seconds")
            print("3. Plug it back in")
            print("4. Run this script again with sudo")
        return

    # Get depth scale and intrinsics
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale}")

    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

    # Align depth to infrared
    align = rs.align(rs.stream.infrared)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # Smoothing filter for chest depth
    chest_filter = OneEuroFilter(min_cutoff=0.3, beta=0.5)
    chest_depth_history = deque(maxlen=150)

    cv2.namedWindow("Skeleton IR", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Skeleton IR", 960, 720)

    print("\nControls: q=quit, r=reset")
    print("Position yourself in front of the camera.\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            ir_frame = aligned.get_infrared_frame()

            if not depth_frame or not ir_frame:
                continue

            frame_count += 1
            current_time = time.time()

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            ir_image = np.asanyarray(ir_frame.get_data())

            # Convert IR to BGR for display and RGB for MediaPipe
            # Enhance contrast for better pose detection
            ir_enhanced = cv2.equalizeHist(ir_image)
            display = cv2.cvtColor(ir_enhanced, cv2.COLOR_GRAY2BGR)
            rgb_for_mp = cv2.cvtColor(ir_enhanced, cv2.COLOR_GRAY2RGB)

            # Run pose detection
            results = pose.process(rgb_for_mp)

            chest_depth_mm = 0
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Get shoulder positions
                left_shoulder = landmarks[11]
                right_shoulder = landmarks[12]

                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                    # Calculate chest center pixel position
                    h, w = ir_image.shape
                    cx = int((left_shoulder.x + right_shoulder.x) / 2 * w)
                    cy = int((left_shoulder.y + right_shoulder.y) / 2 * h)

                    # Sample depth at chest (with radius for robustness)
                    depths = []
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            px = max(0, min(w - 1, cx + dx))
                            py = max(0, min(h - 1, cy + dy))
                            d = depth_frame.get_distance(px, py)
                            if 0.1 < d < 5.0:
                                depths.append(d)

                    if depths:
                        raw_depth = np.median(depths)
                        # Apply smoothing
                        smoothed = chest_filter(raw_depth, current_time)
                        chest_depth_mm = smoothed * 1000
                        chest_depth_history.append(chest_depth_mm)

                        # Draw chest marker
                        cv2.circle(display, (cx, cy), 20, (0, 255, 255), 3)
                        cv2.putText(display, f"{chest_depth_mm:.0f}mm",
                                   (cx + 25, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.7, (0, 255, 255), 2)

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    display,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                )

            # Draw breathing waveform
            if len(chest_depth_history) > 2:
                wave_h = 60
                wave_y = 400
                wave_w = 600

                cv2.rectangle(display, (20, wave_y), (20 + wave_w, wave_y + wave_h),
                             (30, 30, 30), -1)

                depths = list(chest_depth_history)
                min_d = min(depths)
                max_d = max(depths)
                range_d = max_d - min_d if max_d > min_d else 1

                points = []
                for i, d in enumerate(depths):
                    x = 20 + int((i / len(depths)) * wave_w)
                    norm = (d - min_d) / range_d
                    y = wave_y + wave_h - int(norm * wave_h * 0.8) - int(wave_h * 0.1)
                    points.append((x, y))

                for i in range(1, len(points)):
                    cv2.line(display, points[i-1], points[i], (0, 200, 255), 2)

                cv2.putText(display, f"Range: {range_d:.1f}mm", (25, wave_y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            # Info panel
            fps = frame_count / (current_time - start_time) if current_time > start_time else 0
            cv2.rectangle(display, (10, 10), (200, 80), (0, 0, 0), -1)
            cv2.putText(display, f"FPS: {fps:.1f}", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"Chest: {chest_depth_mm:.0f}mm", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Depth preview in corner
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            depth_small = cv2.resize(depth_colormap, (160, 120))
            display[10:130, 470:630] = depth_small

            cv2.imshow("Skeleton IR", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                chest_depth_history.clear()
                chest_filter = OneEuroFilter(min_cutoff=0.3, beta=0.5)
                print("Reset!")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        pose.close()
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames at {fps:.1f} FPS")


if __name__ == "__main__":
    main()
