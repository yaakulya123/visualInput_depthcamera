#!/usr/bin/env python3
"""
Breathing Detection Test - Visual Demo for Liquid Stillness

This script runs the breathing detector and displays:
1. Live camera feed with pose overlay
2. Real-time breathing waveform
3. Current breathing metrics (signal, phase, BPM)
4. Visual feedback (color changes with breath)

Controls:
  - 'q' or ESC: Quit
  - 'r': Reset/recalibrate
  - 's': Save screenshot
  - SPACE: Toggle waveform display

Run: python src/breathing/test_breathing_detection.py
"""

import cv2
import numpy as np
import sys
import time
from collections import deque
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.breathing.breath_detector import BreathingDetector, DetectionMode, BreathingState


class BreathingVisualizer:
    """Real-time visualization of breathing detection."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        # Waveform display
        self.waveform_history = deque(maxlen=300)  # ~10 seconds at 30fps
        self.show_waveform = True

        # Colors (BGR format)
        self.colors = {
            "inhale": (255, 200, 100),    # Light blue
            "exhale": (100, 150, 255),    # Orange-ish
            "hold": (200, 200, 200),      # Gray
            "calibrating": (0, 255, 255), # Yellow
        }

        # Visual feedback intensity
        self.breath_color = np.array([180, 120, 60])  # Base teal color

    def draw_overlay(
        self,
        frame: np.ndarray,
        state: BreathingState,
        landmarks=None,
        mp_drawing=None,
        mp_pose=None
    ) -> np.ndarray:
        """Draw all visualizations on frame."""
        output = frame.copy()
        h, w = output.shape[:2]

        # 1. Draw pose landmarks if available
        if landmarks and mp_drawing and mp_pose:
            mp_drawing.draw_landmarks(
                output,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

            # Highlight shoulder region (breathing detection area)
            self._draw_shoulder_roi(output, landmarks, w, h)

        # 2. Draw breathing color overlay
        output = self._draw_breath_overlay(output, state)

        # 3. Draw metrics panel
        output = self._draw_metrics_panel(output, state)

        # 4. Draw waveform
        if self.show_waveform:
            output = self._draw_waveform(output, state)

        # 5. Draw breathing indicator circle
        output = self._draw_breath_circle(output, state)

        return output

    def _draw_shoulder_roi(
        self, frame: np.ndarray, landmarks, width: int, height: int
    ):
        """Draw rectangle around shoulder region."""
        try:
            left_shoulder = landmarks.landmark[11]
            right_shoulder = landmarks.landmark[12]

            # Get pixel coordinates
            lx = int(left_shoulder.x * width)
            ly = int(left_shoulder.y * height)
            rx = int(right_shoulder.x * width)
            ry = int(right_shoulder.y * height)

            # Draw line between shoulders
            cv2.line(frame, (lx, ly), (rx, ry), (0, 255, 255), 2)

            # Draw detection zone rectangle
            padding = 50
            x1 = min(lx, rx) - padding
            y1 = min(ly, ry) - padding
            x2 = max(lx, rx) + padding
            y2 = max(ly, ry) + padding

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(
                frame, "CHEST ROI", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
        except (IndexError, AttributeError):
            pass

    def _draw_breath_overlay(
        self, frame: np.ndarray, state: BreathingState
    ) -> np.ndarray:
        """Apply color overlay based on breathing state."""
        h, w = frame.shape[:2]

        # Create overlay based on breath signal
        overlay = frame.copy()

        # Calculate color intensity from signal
        intensity = (state.signal + 1) / 2  # Map [-1,1] to [0,1]

        # Inhale = brighter/lighter, Exhale = darker
        if state.phase == "inhale":
            color = (255, 220, 180)  # Warm light
        elif state.phase == "exhale":
            color = (120, 100, 80)   # Cool dark
        else:
            color = (180, 160, 140)  # Neutral

        # Apply subtle tint to edges
        gradient = np.zeros_like(frame, dtype=np.float32)

        # Radial gradient from center
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        normalized_dist = dist_from_center / max_dist

        # Apply color at edges
        alpha = normalized_dist * 0.3 * (0.5 + intensity * 0.5)
        for c in range(3):
            gradient[:, :, c] = alpha * color[c]

        # Blend
        output = cv2.addWeighted(
            frame, 1.0,
            gradient.astype(np.uint8), 0.5,
            0
        )

        return output

    def _draw_metrics_panel(
        self, frame: np.ndarray, state: BreathingState
    ) -> np.ndarray:
        """Draw metrics panel in top-left corner."""
        h, w = frame.shape[:2]

        # Panel background
        panel_h = 180
        panel_w = 300
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        y = 35

        # Title
        cv2.putText(frame, "BREATHING DETECTOR", (20, y), font, 0.6, (0, 255, 255), 2)
        y += 30

        # Phase with color
        phase_color = self.colors.get(state.phase, (255, 255, 255))
        cv2.putText(frame, f"Phase: {state.phase.upper()}", (20, y), font, 0.5, phase_color, 1)
        y += 25

        # Signal
        signal_bar_w = 150
        signal_x = 100
        cv2.putText(frame, "Signal:", (20, y), font, 0.5, color, 1)

        # Signal bar background
        cv2.rectangle(frame, (signal_x, y - 12), (signal_x + signal_bar_w, y + 2), (50, 50, 50), -1)

        # Signal bar fill
        fill_w = int((state.signal + 1) / 2 * signal_bar_w)
        bar_color = (0, 255, 0) if state.signal > 0 else (0, 100, 255)
        cv2.rectangle(frame, (signal_x, y - 12), (signal_x + fill_w, y + 2), bar_color, -1)

        # Center line
        cv2.line(frame, (signal_x + signal_bar_w // 2, y - 15),
                 (signal_x + signal_bar_w // 2, y + 5), (255, 255, 255), 1)
        y += 25

        # BPM
        cv2.putText(frame, f"Breath Rate: {state.breath_rate:.1f} BPM", (20, y), font, 0.5, color, 1)
        y += 25

        # Amplitude
        cv2.putText(frame, f"Amplitude: {state.amplitude:.2f}", (20, y), font, 0.5, color, 1)
        y += 25

        # Confidence
        conf_color = (0, 255, 0) if state.confidence > 0.7 else (0, 165, 255) if state.confidence > 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Confidence: {state.confidence:.2f}", (20, y), font, 0.5, conf_color, 1)

        return frame

    def _draw_waveform(
        self, frame: np.ndarray, state: BreathingState
    ) -> np.ndarray:
        """Draw breathing waveform at bottom of frame."""
        h, w = frame.shape[:2]

        # Add to history
        self.waveform_history.append(state.signal)

        if len(self.waveform_history) < 2:
            return frame

        # Waveform area
        wave_h = 100
        wave_y = h - wave_h - 20
        wave_x = 20
        wave_w = w - 40

        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (wave_x - 5, wave_y - 10),
                     (wave_x + wave_w + 5, wave_y + wave_h + 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Center line
        center_y = wave_y + wave_h // 2
        cv2.line(frame, (wave_x, center_y), (wave_x + wave_w, center_y), (100, 100, 100), 1)

        # Draw waveform
        history = list(self.waveform_history)
        points = []

        for i, val in enumerate(history):
            x = wave_x + int(i * wave_w / len(history))
            y = center_y - int(val * wave_h / 2 * 0.9)
            points.append((x, y))

        if len(points) > 1:
            # Draw filled area
            pts = np.array(points + [(points[-1][0], center_y), (points[0][0], center_y)], np.int32)
            cv2.fillPoly(frame, [pts], (60, 100, 60))

            # Draw line
            for i in range(1, len(points)):
                color = (0, 255, 100)
                cv2.line(frame, points[i-1], points[i], color, 2)

        # Labels
        cv2.putText(frame, "INHALE", (wave_x, wave_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(frame, "EXHALE", (wave_x, wave_y + wave_h + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def _draw_breath_circle(
        self, frame: np.ndarray, state: BreathingState
    ) -> np.ndarray:
        """Draw pulsing circle that represents breathing."""
        h, w = frame.shape[:2]

        # Circle position (top-right)
        cx = w - 80
        cy = 80

        # Size based on signal
        base_radius = 40
        pulse = (state.signal + 1) / 2  # [0, 1]
        radius = int(base_radius * (0.7 + pulse * 0.6))

        # Color based on phase
        if state.phase == "inhale":
            color = (255, 200, 100)  # Light blue
        elif state.phase == "exhale":
            color = (100, 150, 200)  # Warm
        else:
            color = (150, 150, 150)  # Gray

        # Draw glow effect
        for r in range(radius + 20, radius, -2):
            alpha = (radius + 20 - r) / 20 * 0.3
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), r, color, -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Main circle
        cv2.circle(frame, (cx, cy), radius, color, -1)
        cv2.circle(frame, (cx, cy), radius, (255, 255, 255), 2)

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

            # Wait 1ms, if key pressed break
            if cv2.waitKey(1) != -1:
                break

    cap.release()
    cv2.destroyWindow(window_name)
    return True


def select_camera_interactive():
    """
    Let user select camera interactively with preview.
    Returns camera index or None.
    """
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
        print("  ❌ No cameras found!")
        return None

    if len(available) == 1:
        print("\n  Only one camera found, using it automatically.")
        return available[0][0]

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
                            # Save selection
                            try:
                                with open("outputs/selected_camera.txt", "w") as f:
                                    f.write(str(idx))
                            except:
                                pass
                            print(f"  ✅ Selected camera {idx}")
                            return idx
                    else:
                        print(f"  Invalid index. Choose from: {[cam[0] for cam in available]}")
                except ValueError:
                    print("  Please enter a number, 'p' to preview all, or 'q' to quit")
        except KeyboardInterrupt:
            return None


def main():
    """Main entry point for breathing detection test."""
    print("\n" + "="*60)
    print("  LIQUID STILLNESS - Breathing Detection Test")
    print("="*60)
    print("\nInitializing...")

    # Try to import MediaPipe
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        print("  ✅ MediaPipe loaded")
    except ImportError:
        print("  ❌ MediaPipe not found. Install with: pip install mediapipe")
        return
    except AttributeError:
        print("  ❌ MediaPipe version incompatible")
        print("  Run: pip install mediapipe==0.10.9")
        return

    # Determine camera index
    camera_index = None

    # Check for saved selection
    try:
        with open("outputs/selected_camera.txt", "r") as f:
            camera_index = int(f.read().strip())
            print(f"  Using saved camera: {camera_index}")
    except:
        pass

    # If no saved selection, prompt user
    if camera_index is None:
        camera_index = select_camera_interactive()

    if camera_index is None:
        print("  Cancelled.")
        return

    # Initialize camera
    print(f"\n  Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("  ❌ Could not open camera {camera_index}")
        print("  The saved camera is no longer available.")

        # Clear bad selection
        import os
        try:
            os.remove("outputs/selected_camera.txt")
            print("  Cleared saved selection.")
        except:
            pass

        # Prompt for new camera
        print("\n  Let's select a different camera...")
        camera_index = select_camera_interactive()

        if camera_index is None:
            print("  Cancelled.")
            return

        # Try opening new camera
        print(f"\n  Opening camera {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print("  ❌ Still could not open camera")
            return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  ✅ Camera opened: {actual_w}x{actual_h}")

    # Initialize pose detector
    print("  Initializing pose detector...")
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("  ✅ Pose detector ready")

    # Initialize breathing detector
    detector = BreathingDetector(mode=DetectionMode.POSE)
    visualizer = BreathingVisualizer(actual_w, actual_h)
    print("  ✅ Breathing detector ready")

    print("\n" + "-"*60)
    print("  CONTROLS:")
    print("    q/ESC  - Quit")
    print("    c      - Change camera (switch on-the-fly!)")
    print("    r      - Reset/recalibrate")
    print("    s      - Save screenshot")
    print("    SPACE  - Toggle waveform")
    print("-"*60)
    print("\n  Stand in front of camera. Calibrating...")

    # Main loop
    fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ❌ Failed to read frame")
                break

            # Flip for mirror effect (optional, comment out for top-down camera)
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get pose landmarks
            results = pose.process(rgb_frame)
            landmarks = results.pose_landmarks

            # Update breathing detector
            state = detector.update(frame, landmarks=landmarks)

            # Draw visualization
            output = visualizer.draw_overlay(
                frame, state,
                landmarks=landmarks,
                mp_drawing=mp_drawing,
                mp_pose=mp_pose
            )

            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()
                cv2.putText(output, f"FPS: {fps:.1f}", (actual_w - 100, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display
            cv2.imshow("Liquid Stillness - Breathing Test", output)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('c'):  # Change camera
                print("\n  Changing camera...")
                cap.release()
                cv2.destroyAllWindows()

                # Show camera selection
                camera_index = select_camera_interactive()
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
                visualizer.waveform_history.clear()
                print("  ✅ Camera changed! Recalibrating...")
            elif key == ord('r'):
                print("  Recalibrating...")
                detector.reset()
                visualizer.waveform_history.clear()
            elif key == ord('s'):
                filename = f"outputs/breath_screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, output)
                print(f"  Screenshot saved: {filename}")
            elif key == ord(' '):
                visualizer.show_waveform = not visualizer.show_waveform

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
    finally:
        cap.release()
        pose.close()
        cv2.destroyAllWindows()
        print("\n  ✅ Cleanup complete")


if __name__ == "__main__":
    main()
