#!/usr/bin/env python3
"""
Simple RealSense Skeleton Test - Minimal version to avoid crashes

Run with: sudo ./venv/bin/python src/tracking/test_skeleton_simple.py
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
    print("Error: pip install pyrealsense2-macosx")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("Error: pip install mediapipe")
    sys.exit(1)


class BreathingDetector:
    """Simple breathing phase detector."""

    def __init__(self):
        self.buffer = deque(maxlen=20)
        self.smoothed = deque(maxlen=10)
        self.phase = "calibrating"
        self.signal = 0.0
        self.breath_count = 0
        self.last_phase = "hold"

    def update(self, depth_mm):
        self.buffer.append(depth_mm)

        if len(self.buffer) < 5:
            self.phase = "calibrating"
            return

        # Simple moving average
        avg = np.mean(list(self.buffer)[-5:])
        self.smoothed.append(avg)

        if len(self.smoothed) < 3:
            return

        # Calculate trend
        recent = list(self.smoothed)
        derivative = recent[-1] - recent[-3]

        # Detect phase (negative derivative = moving closer = inhale)
        if derivative < -0.3:
            new_phase = "inhale"
        elif derivative > 0.3:
            new_phase = "exhale"
        else:
            new_phase = "hold"

        # Count breaths
        if self.last_phase == "inhale" and new_phase == "exhale":
            self.breath_count += 1

        self.last_phase = new_phase
        self.phase = new_phase

        # Normalized signal for circle size
        if len(self.buffer) >= 10:
            buf = list(self.buffer)
            min_d, max_d = min(buf), max(buf)
            range_d = max_d - min_d if max_d > min_d else 1
            self.signal = -((avg - min_d) / range_d * 2 - 1)
            self.signal = max(-1, min(1, self.signal))


def main():
    print("\n" + "=" * 50)
    print("  Simple Skeleton + Breathing Test")
    print("=" * 50)

    # Initialize RealSense with just depth + color
    pipeline = rs.pipeline()
    config = rs.config()

    print("\nStarting RealSense...")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
        print("RealSense started!")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("\nTry: unplug camera, wait 5 sec, replug, run with sudo")
        return

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Align depth to color
    align = rs.align(rs.stream.color)

    # MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5)

    # Breathing detector
    breath = BreathingDetector()
    depth_history = deque(maxlen=150)

    cv2.namedWindow("Skeleton", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Skeleton", 960, 720)

    print("\nRunning! Press 'q' to quit, 'r' to reset\n")

    frame_count = 0
    start_time = time.time()

    consecutive_failures = 0
    max_failures = 10

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                consecutive_failures = 0  # Reset on success
            except RuntimeError as e:
                consecutive_failures += 1
                print(f"Frame timeout ({consecutive_failures}/{max_failures})...")
                if consecutive_failures >= max_failures:
                    print("\nToo many frame failures. Camera may need reset.")
                    print("Unplug camera, wait 10 sec, replug, run again.")
                    break
                continue

            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            frame_count += 1

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]

            # Run pose detection
            rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            chest_depth = 0
            display = color_image.copy()

            if results.pose_landmarks:
                # Draw skeleton
                mp_draw.draw_landmarks(
                    display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
                )

                # Get shoulder positions for chest
                lm = results.pose_landmarks.landmark
                left_sh = lm[11]
                right_sh = lm[12]

                if left_sh.visibility > 0.5 and right_sh.visibility > 0.5:
                    cx = int((left_sh.x + right_sh.x) / 2 * w)
                    cy = int((left_sh.y + right_sh.y) / 2 * h)

                    # Sample depth at chest
                    depths = []
                    for dy in range(-5, 6):
                        for dx in range(-5, 6):
                            px = max(0, min(w-1, cx + dx))
                            py = max(0, min(h-1, cy + dy))
                            d = depth_frame.get_distance(px, py)
                            if 0.1 < d < 5.0:
                                depths.append(d)

                    if depths:
                        chest_depth = np.median(depths) * 1000  # to mm
                        depth_history.append(chest_depth)
                        breath.update(chest_depth)

                        # Draw chest marker
                        cv2.circle(display, (cx, cy), 10, (0, 255, 255), 2)

            # Draw breathing circle
            circle_x, circle_y = w - 100, h // 2
            base_radius = 50
            radius = int(base_radius + breath.signal * 20)

            # Color by phase
            if breath.phase == "inhale":
                color = (0, 255, 200)
            elif breath.phase == "exhale":
                color = (100, 150, 255)
            elif breath.phase == "calibrating":
                color = (128, 128, 128)
            else:
                color = (200, 200, 200)

            cv2.circle(display, (circle_x, circle_y), radius, color, 3)
            cv2.circle(display, (circle_x, circle_y), max(5, radius - 15), color, -1)

            # Phase text
            cv2.putText(display, breath.phase.upper(), (circle_x - 35, circle_y + radius + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(display, f"Breaths: {breath.breath_count}", (circle_x - 35, circle_y - radius - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Draw waveform
            if len(depth_history) > 2:
                wave_h, wave_y, wave_w = 70, h - 90, w - 40
                cv2.rectangle(display, (20, wave_y - 15), (20 + wave_w, wave_y + wave_h), (20, 20, 20), -1)

                depths = list(depth_history)
                min_d, max_d = min(depths), max(depths)
                range_d = max_d - min_d if max_d > min_d else 1

                points = []
                for i, d in enumerate(depths):
                    x = 20 + int((i / len(depths)) * wave_w)
                    norm = (d - min_d) / range_d
                    y = wave_y + wave_h - int(norm * wave_h * 0.9) - 5
                    points.append((x, y))

                for i in range(1, len(points)):
                    cv2.line(display, points[i-1], points[i], (0, 200, 255), 2)

                cv2.putText(display, f"Chest Depth | Range: {range_d:.1f}mm", (25, wave_y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            # Info panel
            fps = frame_count / (time.time() - start_time)
            cv2.rectangle(display, (10, 10), (220, 120), (0, 0, 0), -1)
            cv2.putText(display, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(display, f"Depth: {chest_depth:.0f}mm", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, f"Phase: {breath.phase}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(display, "q:quit r:reset", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            # Depth preview
            depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_small = cv2.resize(depth_color, (160, 120))
            display[10:130, w-170:w-10] = depth_small

            cv2.imshow("Skeleton", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                breath = BreathingDetector()
                depth_history.clear()
                print("Reset!")

    except KeyboardInterrupt:
        pass
    finally:
        pose.close()
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
