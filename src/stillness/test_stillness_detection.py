#!/usr/bin/env python3
"""
Stillness Detection Test - Visual Demo for Liquid Stillness

This script tests the stillness/jitter detector and displays:
1. Live camera feed with pose overlay
2. Jitter score meter (vertical bar)
3. Regional motion breakdown (arms, torso, legs)
4. Stillness duration timer
5. Quality indicator (settling → focused → deep focus → transcendent)

Controls:
  - 'q' or ESC: Quit
  - 'r': Reset/recalibrate
  - 's': Save screenshot
  - '+'/'-': Adjust sensitivity

Run: python src/stillness/test_stillness_detection.py
"""

import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stillness.stillness_detector import (
    StillnessDetector, StillnessState, BodyRegion, create_stillness_detector
)


class StillnessVisualizer:
    """Real-time visualization of stillness detection."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Colors (BGR format)
        self.colors = {
            "still": (0, 200, 0),        # Green
            "fidgeting": (0, 200, 200),  # Yellow
            "moving": (0, 140, 255),     # Orange
            "restless": (0, 80, 255),    # Red-orange
            "calibrating": (255, 255, 0),# Cyan
        }

        # Quality colors
        self.quality_colors = {
            "settling": (200, 200, 200),    # Gray
            "focused": (200, 200, 0),       # Cyan
            "deep_focus": (200, 150, 0),    # Blue
            "transcendent": (0, 215, 255),  # Gold
        }

        # Jitter history for graph
        self.jitter_history = []
        self.max_history = 150  # ~5 seconds at 30fps

    def draw_overlay(
        self,
        frame: np.ndarray,
        state: StillnessState,
        quality: str,
        landmarks=None,
        mp_drawing=None,
        mp_pose=None
    ) -> np.ndarray:
        """Draw all visualizations on frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Draw pose landmarks if available
        if landmarks and mp_drawing and mp_pose:
            # Color landmarks based on motion state
            landmark_color = self.colors.get(state.motion_type, (255, 255, 255))
            mp_drawing.draw_landmarks(
                output,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=landmark_color, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

        # 2. Draw jitter meter (left side)
        output = self._draw_jitter_meter(output, state)

        # 3. Draw metrics panel (top-left)
        output = self._draw_metrics_panel(output, state, quality)

        # 4. Draw regional motion bars (bottom-left)
        output = self._draw_regional_motion(output, state)

        # 5. Draw stillness timer (center-bottom)
        output = self._draw_stillness_timer(output, state, quality)

        # 6. Draw jitter history graph (bottom)
        output = self._draw_jitter_graph(output, state)

        return output

    def _draw_jitter_meter(
        self, frame: np.ndarray, state: StillnessState
    ) -> np.ndarray:
        """Draw vertical jitter meter on left side."""
        h, w = frame.shape[:2]

        # Meter dimensions
        meter_x = 40
        meter_y = 100
        meter_w = 30
        meter_h = h - 250

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (meter_x - 5, meter_y - 30),
                     (meter_x + meter_w + 60, meter_y + meter_h + 30),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Meter background
        cv2.rectangle(frame, (meter_x, meter_y),
                     (meter_x + meter_w, meter_y + meter_h),
                     (50, 50, 50), -1)

        # Meter fill (bottom to top, inverted so 0 = low, 1 = high)
        fill_h = int(state.jitter_score * meter_h)

        # Gradient fill based on jitter level
        for i in range(fill_h):
            y = meter_y + meter_h - i
            ratio = i / meter_h

            # Color gradient: green → yellow → red
            if ratio < 0.3:
                color = (0, 200, 0)  # Green
            elif ratio < 0.6:
                blend = (ratio - 0.3) / 0.3
                color = (0, int(200 - 100 * blend), int(200 * blend))
            else:
                blend = (ratio - 0.6) / 0.4
                color = (0, int(100 - 100 * blend), 200)

            cv2.line(frame, (meter_x, y), (meter_x + meter_w, y), color, 1)

        # Meter border
        cv2.rectangle(frame, (meter_x, meter_y),
                     (meter_x + meter_w, meter_y + meter_h),
                     (200, 200, 200), 2)

        # Labels
        cv2.putText(frame, "JITTER", (meter_x - 5, meter_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Score value
        cv2.putText(frame, f"{state.jitter_score:.2f}",
                   (meter_x + meter_w + 5, meter_y + meter_h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Threshold markers
        low_y = meter_y + int(meter_h * 0.9)  # 0.1 threshold
        high_y = meter_y + int(meter_h * 0.4)  # 0.6 threshold

        cv2.line(frame, (meter_x - 5, low_y), (meter_x + meter_w + 5, low_y),
                (0, 200, 0), 1)
        cv2.line(frame, (meter_x - 5, high_y), (meter_x + meter_w + 5, high_y),
                (0, 80, 255), 1)

        return frame

    def _draw_metrics_panel(
        self, frame: np.ndarray, state: StillnessState, quality: str
    ) -> np.ndarray:
        """Draw metrics panel in top-left corner."""
        h, w = frame.shape[:2]

        # Panel background
        panel_x = 100
        panel_y = 10
        panel_w = 280
        panel_h = 160

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_w, panel_y + panel_h),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Title
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = panel_y + 25
        cv2.putText(frame, "STILLNESS DETECTOR", (panel_x + 10, y),
                   font, 0.6, (0, 255, 255), 2)

        # Motion type with color
        y += 30
        type_color = self.colors.get(state.motion_type, (255, 255, 255))
        cv2.putText(frame, f"Status: {state.motion_type.upper()}",
                   (panel_x + 10, y), font, 0.5, type_color, 1)

        # Jitter score
        y += 25
        cv2.putText(frame, f"Jitter Score: {state.jitter_score:.3f}",
                   (panel_x + 10, y), font, 0.5, (255, 255, 255), 1)

        # Raw vs smoothed motion
        y += 25
        cv2.putText(frame, f"Raw: {state.raw_motion:.4f} | Smooth: {state.smoothed_motion:.4f}",
                   (panel_x + 10, y), font, 0.4, (180, 180, 180), 1)

        # Stillness duration
        y += 25
        if state.stillness_duration > 0:
            cv2.putText(frame, f"Still for: {state.stillness_duration:.1f}s",
                       (panel_x + 10, y), font, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Still for: --",
                       (panel_x + 10, y), font, 0.5, (150, 150, 150), 1)

        # Quality indicator
        y += 25
        quality_color = self.quality_colors.get(quality, (200, 200, 200))
        cv2.putText(frame, f"Quality: {quality.upper()}",
                   (panel_x + 10, y), font, 0.5, quality_color, 1)

        return frame

    def _draw_regional_motion(
        self, frame: np.ndarray, state: StillnessState
    ) -> np.ndarray:
        """Draw regional motion breakdown bars."""
        h, w = frame.shape[:2]

        # Position
        panel_x = w - 200
        panel_y = 10
        bar_w = 150
        bar_h = 15
        spacing = 22

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x - 10, panel_y - 5),
                     (panel_x + bar_w + 40, panel_y + 7 * spacing + 10),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Title
        cv2.putText(frame, "BODY REGIONS", (panel_x, panel_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw bars for each region
        regions = [
            ("Head", "head"),
            ("Torso", "torso"),
            ("L Arm", "left_arm"),
            ("R Arm", "right_arm"),
            ("L Leg", "left_leg"),
            ("R Leg", "right_leg"),
        ]

        y = panel_y + 35
        for label, key in regions:
            motion = state.regional_motion.get(key, 0)
            fill = min(1.0, motion * 20)  # Scale for visibility

            # Bar background
            cv2.rectangle(frame, (panel_x, y), (panel_x + bar_w, y + bar_h),
                         (50, 50, 50), -1)

            # Bar fill with color gradient
            fill_w = int(fill * bar_w)
            if fill < 0.3:
                color = (0, 200, 0)
            elif fill < 0.6:
                color = (0, 200, 200)
            else:
                color = (0, 100, 255)

            cv2.rectangle(frame, (panel_x, y), (panel_x + fill_w, y + bar_h),
                         color, -1)

            # Label
            cv2.putText(frame, label, (panel_x - 60, y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            y += spacing

        return frame

    def _draw_stillness_timer(
        self, frame: np.ndarray, state: StillnessState, quality: str
    ) -> np.ndarray:
        """Draw stillness timer in center bottom."""
        h, w = frame.shape[:2]

        if state.stillness_duration > 0:
            # Calculate position
            timer_y = h - 180

            # Quality-based styling
            if quality == "transcendent":
                # Golden glow effect
                text = f"{state.stillness_duration:.1f}s"
                color = (0, 215, 255)  # Gold
                font_scale = 2.0

                # Glow effect
                for i in range(3, 0, -1):
                    alpha = 0.1 * i
                    overlay = frame.copy()
                    cv2.putText(overlay, text, (w // 2 - 80, timer_y),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale + i * 0.1,
                               color, 4 + i * 2)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            elif quality == "deep_focus":
                color = (200, 150, 0)
                font_scale = 1.5
            elif quality == "focused":
                color = (200, 200, 0)
                font_scale = 1.2
            else:
                color = (200, 200, 200)
                font_scale = 1.0

            # Timer text
            text = f"{state.stillness_duration:.1f}s"
            cv2.putText(frame, text, (w // 2 - 60, timer_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # Quality label
            cv2.putText(frame, quality.upper(), (w // 2 - 80, timer_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       self.quality_colors.get(quality, (200, 200, 200)), 1)

        return frame

    def _draw_jitter_graph(
        self, frame: np.ndarray, state: StillnessState
    ) -> np.ndarray:
        """Draw jitter history graph at bottom."""
        h, w = frame.shape[:2]

        # Update history
        self.jitter_history.append(state.jitter_score)
        if len(self.jitter_history) > self.max_history:
            self.jitter_history.pop(0)

        if len(self.jitter_history) < 2:
            return frame

        # Graph dimensions
        graph_x = 100
        graph_y = h - 100
        graph_w = w - 220
        graph_h = 60

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x - 5, graph_y - graph_h - 10),
                     (graph_x + graph_w + 5, graph_y + 10),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Draw graph line
        points = []
        for i, val in enumerate(self.jitter_history):
            x = graph_x + int(i * graph_w / self.max_history)
            y = graph_y - int(val * graph_h)
            points.append((x, y))

        # Draw filled area
        if len(points) > 1:
            pts = np.array(points + [(points[-1][0], graph_y), (points[0][0], graph_y)], np.int32)

            # Color based on average jitter
            avg_jitter = np.mean(self.jitter_history[-30:]) if len(self.jitter_history) >= 30 else 0.5
            if avg_jitter < 0.2:
                fill_color = (0, 80, 0)
                line_color = (0, 200, 0)
            elif avg_jitter < 0.5:
                fill_color = (0, 80, 80)
                line_color = (0, 200, 200)
            else:
                fill_color = (0, 40, 80)
                line_color = (0, 100, 255)

            cv2.fillPoly(frame, [pts], fill_color)

            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], line_color, 2)

        # Label
        cv2.putText(frame, "JITTER HISTORY", (graph_x, graph_y - graph_h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame


def preview_camera(idx, duration=2):
    """Show a quick preview of a camera."""
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window_name = f"Preview: Camera {idx}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    print(f"    Showing preview for Camera {idx}... (press any key to close)")

    import time
    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()
        if ret:
            # Add text overlay
            cv2.putText(frame, f"Camera {idx} Preview", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, "Press any key to close", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) != -1:
                break

    cap.release()
    cv2.destroyWindow(window_name)
    return True


def select_camera():
    """Select camera interactively with preview."""
    print("\n  Scanning for cameras...")
    available = []

    for idx in range(10):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Identify camera type (Camera 0 is usually built-in Mac camera)
                if idx == 0:
                    cam_type = "💻 Built-in Camera (likely Mac)"
                elif w == 640 and h == 480:
                    cam_type = "📱 Secondary Camera (likely iPhone)"
                else:
                    cam_type = "📷 External Camera"

                available.append((idx, w, h, cam_type))
                print(f"    [{idx}] {cam_type} - {w}x{h}")
            cap.release()

    if not available:
        return None

    if len(available) == 1:
        return available[0][0]

    # Check for saved selection
    try:
        with open("outputs/selected_camera.txt", "r") as f:
            idx = int(f.read().strip())
            if any(cam[0] == idx for cam in available):
                print(f"\n  Using saved camera: {idx}")
                # Offer to preview saved selection
                preview_choice = input("  Preview this camera? [y/N]: ").strip().lower()
                if preview_choice == 'y':
                    preview_camera(idx, duration=2)
                    confirm = input("  Keep this selection? [Y/n]: ").strip().lower()
                    if confirm == 'n':
                        # Delete saved selection and continue to selection
                        import os
                        try:
                            os.remove("outputs/selected_camera.txt")
                        except:
                            pass
                    else:
                        return idx
                else:
                    return idx
    except:
        pass

    # Multiple cameras - let user preview and choose
    print("\n  Multiple cameras detected!")
    print("  You can preview each camera to see which is which.")

    while True:
        try:
            print("\n  Options:")
            print("    [number] - Preview & select that camera")
            print("    [p] - Preview all cameras one by one")
            print("    [q] - Quit")

            choice = input("\n  Your choice: ").strip().lower()

            if choice == 'q':
                return None
            elif choice == 'p':
                # Preview all cameras
                print("\n  Previewing all cameras...")
                for cam in available:
                    idx, w, h, cam_type = cam
                    print(f"\n  Camera {idx}: {cam_type}")
                    preview_camera(idx, duration=3)
                print("\n  Preview complete!")
            else:
                # Try to parse as camera index
                try:
                    idx = int(choice)
                    if any(cam[0] == idx for cam in available):
                        # Preview this camera
                        print(f"\n  Previewing camera {idx}...")
                        preview_camera(idx, duration=2)

                        # Confirm selection
                        confirm = input(f"\n  Use camera {idx}? [Y/n]: ").strip().lower()
                        if confirm != 'n':
                            with open("outputs/selected_camera.txt", "w") as f:
                                f.write(str(idx))
                            print(f"  ✅ Selected camera {idx}")
                            return idx
                    else:
                        print(f"  Invalid index. Choose from: {[cam[0] for cam in available]}")
                except ValueError:
                    print("  Please enter a number, 'p' to preview all, or 'q' to quit")
        except KeyboardInterrupt:
            return None


def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("  LIQUID STILLNESS - Stillness Detection Test")
    print("="*60)
    print("\nInitializing...")

    # Import MediaPipe
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        print("  ✅ MediaPipe loaded")
    except (ImportError, AttributeError) as e:
        print(f"  ❌ MediaPipe error: {e}")
        print("  Run: pip install mediapipe==0.10.9")
        return

    # Select camera
    camera_index = select_camera()
    if camera_index is None:
        print("  Cancelled.")
        return

    # Initialize camera
    print(f"\n  Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"  ❌ Could not open camera {camera_index}")
        print("  The selected camera is no longer available.")

        # Clear bad selection
        import os
        try:
            os.remove("outputs/selected_camera.txt")
            print("  Cleared saved selection.")
        except:
            pass

        # Prompt for new camera
        print("\n  Let's select a different camera...")
        camera_index = select_camera()

        if camera_index is None:
            print("  Cancelled.")
            return

        # Try opening new camera
        print(f"\n  Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("  ❌ Still could not open camera")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Camera opened: {actual_w}x{actual_h}")

    # Initialize pose detector
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("  ✅ Pose detector ready")

    # Initialize stillness detector
    detector = create_stillness_detector(sensitivity="normal")
    visualizer = StillnessVisualizer(actual_w, actual_h)
    print("  ✅ Stillness detector ready")

    print("\n" + "-"*60)
    print("  CONTROLS:")
    print("    q/ESC  - Quit")
    print("    c      - Change camera (switch on-the-fly!)")
    print("    r      - Reset/recalibrate")
    print("    s      - Save screenshot")
    print("    +/-    - Adjust sensitivity")
    print("-"*60)
    print("\n  Stand still and wait for calibration...")

    # Sensitivity levels
    sensitivities = ["low", "normal", "high"]
    current_sensitivity = 1  # Start with "normal"

    # Main loop
    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get pose
            results = pose.process(rgb_frame)
            landmarks = results.pose_landmarks

            # Update stillness detector
            state = detector.update(frame, landmarks=landmarks)
            quality = detector.get_stillness_quality()

            # Draw visualization
            output = visualizer.draw_overlay(
                frame, state, quality,
                landmarks=landmarks,
                mp_drawing=mp_drawing,
                mp_pose=mp_pose
            )

            # FPS counter
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
                cv2.putText(output, f"FPS: {fps:.1f}", (actual_w - 100, actual_h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Sensitivity indicator
            sens_text = f"Sensitivity: {sensitivities[current_sensitivity]}"
            cv2.putText(output, sens_text, (actual_w - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Liquid Stillness - Stillness Test", output)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):  # Change camera
                print("\n  Changing camera...")
                cap.release()
                cv2.destroyAllWindows()

                # Show camera selection
                camera_index = select_camera()
                if camera_index is None:
                    print("  Camera change cancelled. Exiting...")
                    break

                # Reopen with new camera
                print(f"  Opening camera {camera_index}...")
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    print("  ❌ Could not open new camera")
                    break

                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                # Reset detector
                detector.reset()
                visualizer.jitter_history.clear()
                print("  ✅ Camera changed! Recalibrating...")
            elif key == ord('r'):
                print("  Recalibrating...")
                detector.reset()
                visualizer.jitter_history.clear()
            elif key == ord('s'):
                filename = f"outputs/stillness_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, output)
                print(f"  Screenshot saved: {filename}")
            elif key == ord('+') or key == ord('='):
                current_sensitivity = min(2, current_sensitivity + 1)
                detector = create_stillness_detector(sensitivities[current_sensitivity])
                print(f"  Sensitivity: {sensitivities[current_sensitivity]}")
            elif key == ord('-'):
                current_sensitivity = max(0, current_sensitivity - 1)
                detector = create_stillness_detector(sensitivities[current_sensitivity])
                print(f"  Sensitivity: {sensitivities[current_sensitivity]}")

    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()
        print("  ✅ Cleanup complete")


if __name__ == "__main__":
    main()
